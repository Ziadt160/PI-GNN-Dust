"""Train and score the model roster on AI-DUST's own metrics.

    python -u experiments/benchmark_aidust.py --models M2_SRC --epochs 2 --smoke
    python -u experiments/benchmark_aidust.py --groups A B --epochs 8
    python -u experiments/benchmark_aidust.py --core --epochs 25

Fairness rules enforced here, because a benchmark that gets these wrong measures
bookkeeping rather than modelling:

1. **Identical evaluation points.**  Every model is scored on the same
   ``(timestep, cell)`` set: test timesteps subsampled by ``--eval-stride``, all
   251 cells.  Full-domain models emit those cells in one pass; patch models are
   run once per centre cell.  No model is scored on a set another did not see.

2. **Equal compute per epoch, not equal samples.**  A full-domain step predicts
   251 targets from 251 node computations (1 per target); a 7x7 patch predicts 1
   target from 49 (49 per target).  Equal *sample* counts would hand one topology
   ~49x the compute.  We instead fix a node-computation budget per epoch and
   report how many targets each model actually saw.

3. **Train-only normalisation**, fitted per model on its own feature view.

4. **Selection on validation**, never on test; test is touched once, at the end.

`SaudiFields` is cached across models -- only ``cadence`` and ``divergence``
change the preprocessed fields, so at most two passes are ever built.
"""
import argparse
import copy
import json
import zlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from torch_geometric.loader import DataLoader                       # noqa: E402

from src.data.saudi_dataset import DataConfig, SaudiFields          # noqa: E402
from src.data import topologies as topo                             # noqa: E402
from src.data.patch_frames import PatchFrameBuilder, predict_frames  # noqa: E402
from src.data.rollout import Rollout                                # noqa: E402
from src.metrics import aidust_metrics as M                         # noqa: E402
from src.models import registry                                     # noqa: E402
from src.models.unified import build_model                          # noqa: E402
from src.physics.gated_loss import BenchmarkLoss                    # noqa: E402
from src.utils.graph import build_masked_graph, build_dust_graph    # noqa: E402

DEFAULT_NODE_BUDGET = 2_000_000
NORM_SAMPLE_FRAMES = 120


# --------------------------------------------------------------------------- #
# Field cache
# --------------------------------------------------------------------------- #
_FIELDS = {}


def get_fields(data_cfg: DataConfig, verbose=True):
    """Cache by the switches that change preprocessing; all others are assembly-time."""
    key = (data_cfg.cadence, data_cfg.divergence, data_cfg.reference_mode,
           data_cfg.drag_partition)
    if key not in _FIELDS:
        _FIELDS[key] = SaudiFields(data_cfg, verbose=verbose)
    return _FIELDS[key]


def make_data_cfg(model_cfg):
    return DataConfig(**model_cfg.data)


# --------------------------------------------------------------------------- #
# Loader construction
# --------------------------------------------------------------------------- #
def _stratum_weights(fields, idxs):
    """Per-(step, cell) weights emulating AI-DUST stratification on a dense grid.

    The patch topology can *sample* strata; the full-domain topology predicts every
    cell every step, so the equivalent is to re-weight the loss by inverse stratum
    frequency.  Same intent -- stop the clean, non-emitting majority from dominating.
    """
    pm_phys = np.expm1(fields.pm10[idxs])
    active = fields.emis_active[idxs].astype(int)
    conc_bin = np.digitize(pm_phys, topo.CONC_BINS)
    stratum = active * (len(topo.CONC_BINS) + 1) + conc_bin
    counts = np.bincount(stratum.ravel(), minlength=stratum.max() + 1).astype("float64")
    inv = np.zeros_like(counts)
    inv[counts > 0] = 1.0 / counts[counts > 0]
    w = inv[stratum]
    return (w / w.mean()).astype("float32")


def build_split(fields, model_cfg, data_cfg, years, node_budget, seed, stride=1):
    """Dataset + loader for one split, sized to the shared compute budget."""
    idxs = fields.split_indices(*years)
    if stride > 1:
        idxs = idxs[::stride]

    if data_cfg.topology == "full":
        edge_index, pos, _ = build_masked_graph(fields.mask, neighbors=data_cfg.neighbors)
        per_sample_nodes = fields.n_nodes
        if node_budget:
            n = max(1, node_budget // per_sample_nodes)
            if n < len(idxs):
                rng = np.random.default_rng(seed)
                idxs = np.sort(rng.choice(idxs, n, replace=False))
        ds = topo.FullDomainDataset(fields, idxs, edge_index, pos, cfg=data_cfg)
        weights = (_stratum_weights(fields, idxs) if model_cfg.stratified else None)
        return ds, weights, idxs

    # patch topology
    p = data_cfg.patch
    edge_index, pos = build_dust_graph(p, p, neighbors=data_cfg.neighbors)
    gather = topo.build_patch_gather(fields, p)
    n_pairs = max(1, node_budget // (p * p)) if node_budget else len(idxs) * 64
    sampler = topo.stratified_pairs if model_cfg.stratified else topo.uniform_pairs
    pairs = sampler(fields, idxs, n_pairs, seed=seed)
    ds = topo.PatchDataset(fields, pairs, edge_index, pos, gather, p, cfg=data_cfg)
    return ds, None, idxs


def fit_norm_stats(ds, n_frames=NORM_SAMPLE_FRAMES):
    """Feature mean/std from a subsample of the training split only."""
    take = np.linspace(0, len(ds) - 1, min(n_frames, len(ds))).astype(int)
    xs = [ds[int(i)].x.numpy() for i in take]
    stacked = np.concatenate(xs, axis=0)
    return stacked.mean(0), stacked.std(0)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, loss_fn, opt, device, weights_lookup=None, train=True):
    model.train(train)
    total, nb, gate = 0.0, 0, 0.0
    for batch in loader:
        batch = batch.to(device)
        if train:
            opt.zero_grad()
        pred = model(batch)
        target = batch.y
        loss, info = loss_fn(pred, target, c_ref=batch.c_ref)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        total += float(loss.detach())
        gate += info.get("gate_fired_frac", 0.0)
        nb += 1
    return total / max(nb, 1), gate / max(nb, 1)


@torch.no_grad()
def collect_predictions(model, ds, device, batch_size, n_steps, n_nodes, n_targets,
                        topology):
    """Run the model over the evaluation set and reshape to ``[T, N, targets]``."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    for batch in loader:
        batch = batch.to(device)
        preds.append(model(batch).cpu().numpy())
        trues.append(batch.y.cpu().numpy())
    p = np.concatenate(preds).reshape(n_steps, n_nodes, n_targets)
    t = np.concatenate(trues).reshape(n_steps, n_nodes, n_targets)
    return p, t


def aggregate_seeds(runs):
    """Mean over seed replicates, with sd on the headline metrics.

    Reporting a single-seed number for a benchmark whose run-to-run sd is ~0.033 TS
    invites reading noise as signal, so replicated runs carry their spread.
    """
    base = dict(runs[0])
    numeric = [k for k, v in runs[0].items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for k in numeric:
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float)) and r[k] == r[k]]
        if vals:
            base[k] = float(np.mean(vals))
    for k in ("TS", "RMSE_ugm3", "R_spatial_mean", "TS_day1"):
        vals = [r[k] for r in runs if isinstance(r.get(k), (int, float)) and r[k] == r[k]]
        base[k + "_sd"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    base["n_seeds"] = len(runs)
    return base


def train_and_score(model_cfg, args, device):
    # Seed per model, not once per process.  Seeding only in main() makes each
    # model's weight init depend on how many models ran before it, so the same
    # config scores differently depending on its position in the roster --
    # measured at sd 0.033 TS, max 0.056, which is larger than most of the
    # ablation effects being measured.  Deriving the seed from the model name
    # makes every run reproducible and roster-order independent.
    model_seed = args.seed + (zlib.crc32(model_cfg.name.encode()) % 100_000)
    torch.manual_seed(model_seed)
    np.random.seed(model_seed % (2 ** 32 - 1))

    data_cfg = make_data_cfg(model_cfg)
    fields = get_fields(data_cfg, verbose=args.verbose)
    n_targets = 2 if data_cfg.multitask else 1

    train_ds, train_w, _ = build_split(fields, model_cfg, data_cfg, data_cfg.train_years,
                                       args.node_budget, args.seed)
    val_ds, _, _ = build_split(fields, model_cfg, data_cfg, data_cfg.val_years,
                               args.node_budget // 4, args.seed + 1)

    # Test points are identical for every model: the same strided timesteps, all cells.
    test_idxs = fields.split_indices(*data_cfg.test_years)[::args.eval_stride]

    mean, std = fit_norm_stats(train_ds)
    model = build_model(model_cfg, train_ds.n_channels, n_targets, data_cfg.topology)
    model.set_norm_stats(mean, std)
    model.to(device)

    bs = args.batch_patch if data_cfg.topology == "patch" else args.batch_full
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False)

    loss_fn = BenchmarkLoss(model_cfg.loss, device=device).to(device)
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)

    n_params = sum(p.numel() for p in model.parameters())
    targets_seen = len(train_ds) * (fields.n_nodes if data_cfg.topology == "full" else 1)
    print(f"    in_ch={train_ds.n_channels} params={n_params:,} "
          f"train_samples={len(train_ds):,} targets/epoch={targets_seen:,} bs={bs}",
          flush=True)

    best, best_state, t0 = float("inf"), None, time.time()
    for ep in range(args.epochs):
        tr, gate = run_epoch(model, train_loader, loss_fn, opt, device, train=True)
        va, _ = run_epoch(model, val_loader, loss_fn, opt, device, train=False)
        if va < best:
            best = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if args.verbose or ep == args.epochs - 1:
            msg = f"    ep {ep+1:3d}/{args.epochs} train {tr:.4f} val {va:.4f}"
            if model_cfg.loss == "gated":
                msg += f" gate_fired {gate*100:.1f}%"
            print(msg, flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    train_time = time.time() - t0

    if data_cfg.topology == "patch":
        # Whole-frame assembly: ~2.9k gathers instead of ~733k Data objects.
        builder = PatchFrameBuilder(fields, data_cfg,
                                    topo.build_patch_gather(fields, data_cfg.patch),
                                    data_cfg.patch, neighbors=data_cfg.neighbors,
                                    device=device)
        pred, true = predict_frames(model, builder, test_idxs)
    else:
        edge_index, pos, _ = build_masked_graph(fields.mask, neighbors=data_cfg.neighbors)
        test_ds = topo.FullDomainDataset(fields, test_idxs, edge_index, pos, cfg=data_cfg)
        pred, true = collect_predictions(
            model, test_ds, device, bs * 2, len(test_idxs), fields.n_nodes, n_targets,
            data_cfg.topology)

    # Cache raw predictions so a metric change never requires retraining again.
    # (The first version of this benchmark stored only the metric dicts, which
    # meant adding ETS/HSS/FGE forced a full re-run.)
    np.savez_compressed(
        os.path.join(pred_dir, f"{model_cfg.name.replace('/', '_')}.npz"),
        pred=pred.astype("float32"), true=true.astype("float32"),
        test_idxs=np.asarray(test_idxs), emis_active=fields.emis_active[test_idxs],
    )

    active = fields.emis_active[test_idxs]
    row = M.evaluate(pred[..., 0], true[..., 0], emission_active=active,
                     label=model_cfg.name)
    row.update({"note": model_cfg.note, "params": n_params, "train_s": round(train_time, 1),
                "targets_per_epoch": targets_seen, "topology": data_cfg.topology,
                "in_channels": train_ds.n_channels, "val_loss": best})
    if n_targets == 2:
        pm25 = M.evaluate(pred[..., 1], true[..., 1], label=model_cfg.name + "|PM2.5")
        row["pm25_RMSE_ugm3"] = pm25["RMSE_ugm3"]
        row["pm25_R2_log"] = pm25["R2_log"]

    pred_dir = os.path.join("results", "preds")
    os.makedirs(pred_dir, exist_ok=True)

    if args.rollout_steps > 0:
        roll_starts = np.linspace(test_idxs[0], test_idxs[-1] - args.rollout_steps - 1,
                                  args.rollout_starts).astype(int)
        if data_cfg.topology == "patch":
            roll = Rollout(fields, data_cfg, model, device, "patch",
                           patch_builder=builder)
        else:
            roll = Rollout(fields, data_cfg, model, device, "full",
                           edge_index=edge_index, pos=pos)
        rp, rt = roll.run(roll_starts, args.rollout_steps)
        # Keep [K, S, N] for the PM10 channel: spatial R is then across cells and
        # temporal R across initialisation times, both meaningful.
        per_step, per_day = M.evaluate_rollout(rp[..., 0], rt[..., 0],
                                               steps_per_day=8, label=model_cfg.name)
        # Cache the rollout too: scoring lead-day forecasts against station
        # observations is the only way our numbers become comparable to the
        # published 24 h / 48 h threat scores.
        np.savez_compressed(
            os.path.join(pred_dir, f"{model_cfg.name.replace('/', '_')}__rollout.npz"),
            pred=rp[..., 0].astype("float32"), true=rt[..., 0].astype("float32"),
            starts=np.asarray(roll_starts), steps_per_day=8)
        row["rollout"] = {"per_day": [{k: d[k] for k in
                                       ("lead_day", "TS", "R_spatial_mean",
                                        "RMSE_ugm3", "RE_mean")} for d in per_day]}
        day1 = per_day[0] if per_day else {}
        row["TS_day1"] = day1.get("TS", float("nan"))
        row["RMSE_day1"] = day1.get("RMSE_ugm3", float("nan"))
        if len(per_day) >= 10:
            row["TS_day10"] = per_day[9]["TS"]

    # Persistence on exactly these evaluation points, as the floor.
    persist = np.stack([fields.pm10[test_idxs]], -1)[..., 0]
    row["_persistence"] = M.evaluate(persist, true[..., 0], emission_active=active,
                                     label="persistence")
    return row, model


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--core", action="store_true", help="the 6-model minimum roster")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-full", type=int, default=16)
    ap.add_argument("--batch-patch", type=int, default=512)
    ap.add_argument("--node-budget", type=int, default=DEFAULT_NODE_BUDGET)
    ap.add_argument("--eval-stride", type=int, default=4)
    ap.add_argument("--rollout-steps", type=int, default=0,
                    help="free-running lead steps (80 = 10 days at 3h, AI-DUST Fig.2-3); "
                         "0 disables. Single-step scores are NOT comparable to the paper.")
    ap.add_argument("--rollout-starts", type=int, default=48,
                    help="number of rollout initialisation times, spread over the test set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1,
                    help="train each model with N seeds and report mean +/- sd; "
                         "run-to-run sd is ~0.033 TS, so single-seed deltas below "
                         "~0.07 are not separable from noise")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="results/aidust_benchmark.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny budget, 1 epoch -- shape/plumbing check only")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.node_budget, args.eval_stride = 1, 40_000, 64
        args.rollout_steps = min(args.rollout_steps, 8)
        args.rollout_starts = min(args.rollout_starts, 4)

    names = registry.CORE if args.core else args.models
    models = registry.select(names=names, groups=args.groups)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device} | models={len(models)} | epochs={args.epochs} "
          f"| node_budget={args.node_budget:,} | eval_stride={args.eval_stride}\n",
          flush=True)

    rows, persistence_row = [], None
    for i, cfg in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {cfg.name} -- {cfg.note}", flush=True)
        runs = []
        for rep in range(max(1, args.seeds)):
            rep_args = copy.copy(args)
            rep_args.seed = args.seed + 1000 * rep
            try:
                r, _ = train_and_score(cfg, rep_args, device)
                runs.append(r)
            except Exception as exc:                   # keep the sweep alive
                import traceback
                print(f"    FAILED (seed {rep}): {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
        if not runs:
            continue
        row = aggregate_seeds(runs) if len(runs) > 1 else runs[0]
        persistence_row = row.pop("_persistence", persistence_row)
        rows.append(row)
        sd = f" +/-{row['TS_sd']:.3f}" if row.get("n_seeds", 1) > 1 else ""
        seeds_note = f", {row['n_seeds']} seeds" if row.get("n_seeds", 1) > 1 else ""
        print(f"    -> TS={row['TS']:.3f}{sd} R_spat={row['R_spatial_mean']:.3f} "
              f"RMSE={row['RMSE_ugm3']:.1f} ug/m3 ({row['train_s']:.0f}s{seeds_note})\n",
              flush=True)

    table_rows = ([persistence_row] if persistence_row else []) + rows
    print("\n" + "=" * 110)
    print("AI-DUST metric set | test 2021-2024 | PM10 | physical units")
    fields_any = next(iter(_FIELDS.values()), None)
    if fields_any is not None:
        print("emission drivers: " +
              "; ".join(f"{k}={v}" for k, v in fields_any.drivers_meta.items()))
    print("=" * 110)
    print(M.format_table(table_rows))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"rows": table_rows,
                   "drivers": fields_any.drivers_meta if fields_any else {},
                   "args": vars(args)}, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
