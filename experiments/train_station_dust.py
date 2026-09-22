"""Forecast observed dust events from ERA5 physics -- the target that exists.

Why this replaced the CAMS-target models
----------------------------------------
Validated against 234,512 matched station-hours, CAMS PM10 under-represents real
dust storms by 17-56x and never exceeded 1,000 ug/m^3 across 499 WMO-duststorm
hours.  Models trained on it learn to rank dustiness, not detect storms
(TS 0.11, POD 0.20 at 3 h).  Airport visibility does contain dust storms, is
observed rather than modelled, and there are 3.56M observations of it.

  KEPT     full-domain masked graph + `PhysicsFluxLayer` -- dust emitted upwind
           arrives 12-24 h later, so transport still has to be modelled
  KEPT     ERA5 surface drivers and the Shao emission scheme
  CHANGED  target: station-observed dust occurrence, not CAMS PM10
  CHANGED  loss: class-weighted BCE for a 0.07-3% base rate, not MSE
  CHANGED  CAMS PM10 demoted from target to auxiliary input

Horizon-matched meteorology
---------------------------
**One model per horizon, each fed meteorology at its own target time.**  The first
version shared one model across horizons and supplied wind only at t+3 h, so the
24 h head had to predict a dust storm a day ahead while seeing only the wind three
hours after initialisation -- it scored TS 0.025.  Operational dust forecasts are
driven by NWP wind *at the valid time*; AI-DUST feeds meteorology at ``t`` and
``t+dt`` from AIFS.  Here ERA5 at ``t+h`` stands in for a perfect wind forecast,
the same idealisation AI-DUST's evaluation makes.

Other design notes
------------------
**Forward on the whole graph, supervise at the stations.**  Message passing can
carry dust from unobserved upwind cells; the loss is masked to cells containing a
station.  The learned function is of gridded drivers, so it applies everywhere.

**Capacity is deliberately small.**  With ~26 supervised nodes the first run
overfit hard (val bottomed at epoch 5, then rose 20%).  Hidden width down, dropout
up, early stopping on validation loss.

**`--loso` holds stations out of training entirely.**  Generalising to cells with
no station is an assumption in this design, not a result; this tests it.

Run:  python -u experiments/train_station_dust.py --epochs 25
      python -u experiments/train_station_dust.py --epochs 25 --loso
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.data.saudi_dataset import DataConfig, SaudiFields        # noqa: E402
from src.data.station_target import EVENT_THRESHOLDS, DustTarget  # noqa: E402
from src.data.topologies import FeatureAssembler                  # noqa: E402
from src.layers.flux import PhysicsFluxLayer                      # noqa: E402
from src.metrics.aidust_metrics import contingency_scores         # noqa: E402
from src.models.dust_transport import DustTransportGNN, hops_for_horizon  # noqa: E402
from src.models.ordinal import (                                  # noqa: E402
    SORDLoss, cumulative_probabilities, grades_from_labels, inverse_frequency_weights)
from src.utils.graph import build_masked_graph                    # noqa: E402

HORIZON_STEPS = {"3h": 1, "24h": 8, "48h": 16}     # at the 3-hourly cadence
PATIENCE = 12


class StationDustModel(nn.Module):
    """Flux-GNN over the full domain, one logit per threshold per node."""

    def __init__(self, in_channels, n_out, hidden=48, hops=2, dropout=0.3):
        super().__init__()
        self.flux_in = PhysicsFluxLayer(in_channels, hidden)
        self.flux_hidden = nn.ModuleList(
            PhysicsFluxLayer(hidden, hidden) for _ in range(hops - 1))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)
        self.register_buffer("mean", torch.zeros(in_channels))
        self.register_buffer("std", torch.ones(in_channels))

    def set_stats(self, m, s):
        with torch.no_grad():
            self.mean.copy_(torch.as_tensor(m, dtype=torch.float32))
            self.std.copy_(torch.clamp(torch.as_tensor(s, dtype=torch.float32), min=1e-6))

    def forward(self, x, edge_index, pos):
        pm10, u, v = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        xn = (x - self.mean) / self.std
        h = torch.relu(self.flux_in(xn, edge_index, pos, u, v, pm10))
        for fl in self.flux_hidden:
            h = h + torch.relu(fl(h, edge_index, pos, u, v, pm10))
        return self.head(self.drop(h))


def contingency(pred_evt, obs_evt):
    """Full categorical score set for one threshold.

    Delegates to :func:`src.metrics.aidust_metrics.contingency_scores` so the
    trainer and the benchmark cannot drift apart -- this used to be a second,
    slightly different implementation that reported neither SEDI nor the base
    rate, which is how a TS of 0.135 at a 1% base rate came to be quoted beside
    published numbers measured at 13%.

    ``events`` is kept as an alias of ``events_observed`` so the JSONs written
    before this change stay readable by the same code.
    """
    pred_evt = np.asarray(pred_evt, dtype=bool)
    obs_evt = np.asarray(obs_evt, dtype=bool)
    nc = int(np.sum(pred_evt & obs_evt))
    nf = int(np.sum(pred_evt & ~obs_evt))
    no = int(np.sum(~pred_evt & obs_evt))
    cn = int(pred_evt.size - nc - nf - no)
    sc = contingency_scores(nc, nf, no, cn)
    sc["events"] = sc["events_observed"]
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loso", action="store_true")
    ap.add_argument("--buffer-rings", type=int, default=1,
                    help="concentric rings of cells reserved as INPUT-ONLY buffer, "
                         "measured by successive 8-neighbour erosion from the domain "
                         "edge. The core inside them is what gets predicted. Sizing: "
                         "at 24 h dust crosses ~7 cells, so a deep buffer is what "
                         "gives the core genuine upwind inflow. On the 251-cell Saudi "
                         "domain 3 rings leave only 4 stations in the core; the wide "
                         "domain is where a deep buffer becomes affordable.")
    ap.add_argument("--interior-only", action="store_true",
                    help="supervise and score only on cells whose 8 neighbours are all "
                         "valid. Boundary cells have a truncated stencil and, with the "
                         "upwind source region masked out, no inflow information at all")
    ap.add_argument("--transport", action="store_true",
                    help="use the DustTransportGNN: every node emits and advects, "
                         "hops scaled to the forecast horizon, readout at interior "
                         "cells only. The default 2-hop model has a 152 km reach, "
                         "while 24 h transport is 500-700 km")
    ap.add_argument("--ordinal", action="store_true",
                    help="predict one ORDINAL severity grade (clear/dusty/storm/"
                         "severe) with a SORD loss instead of three independent "
                         "binary heads. The three threshold probabilities come "
                         "back as tail sums, monotone by construction, so the "
                         "scores stay comparable. Fixes the `severe` head, which "
                         "at a 0.070%% base rate never fires and scores TS 0.000")
    ap.add_argument("--levels", action="store_true",
                    help="add the ERA5 pressure-level stability block (t_500, "
                         "t_700-500, Ri(850-700), low-level jet, shear). Every "
                         "feature this project has used so far is a surface field, "
                         "yet the one comparable study that publishes a feature "
                         "ranking finds pressure-level quantities in its top 3 at "
                         "every horizon. Needs download_era5_levels.py to have "
                         "finished; a partial archive is refused, not padded")
    ap.add_argument("--upwind-obs", action="store_true",
                    help="reveal observations at ALL stations as input, including "
                         "boundary ones, while still scoring only interior cells. "
                         "An interior storm is preceded by a northern station's "
                         "storm 48.8%% of the time within 24 h (21.8%% from the south)")
    ap.add_argument("--daily", action="store_true",
                    help="score daily dust-days (any storm hour in the day) rather than "
                         "3-hourly station-hours. This is the unit AI-DUST Fig.3 and WMO "
                         "SDS-WAS report; it raises the base rate 4.5x (0.384%% -> 1.744%%)")
    ap.add_argument("--obs-input", action="store_true",
                    help="feed the CURRENT observed dust state at station cells as "
                         "extra input channels (a light form of data assimilation)")
    ap.add_argument("--obs-dropout", type=float, default=0.5,
                    help="probability a station's observation is hidden during "
                         "training, so the model must also work where none exists")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--metar-dir", default=None)
    # Splits are configurable so the evaluation period can be moved inside the
    # years for which real ERA5 emission drivers exist. Using measured drivers for
    # training and proxies at test would introduce a train/test distribution shift,
    # which is worse than using proxies throughout -- hence one split, one source.
    ap.add_argument("--train-years", type=int, nargs=2, default=[2003, 2018])
    ap.add_argument("--val-years", type=int, nargs=2, default=[2019, 2020])
    ap.add_argument("--test-years", type=int, nargs=2, default=[2021, 2024])
    ap.add_argument("--min-driver-coverage", type=float, default=0.98,
                    help="lower this only when the split lies inside the downloaded "
                         "driver years, so train and test share one driver source")
    ap.add_argument("--out", default="results/station_dust.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    # Checkpoints and saved predictions are namespaced by the run, so an ablation
    # arm cannot silently overwrite the arm it is being compared against.
    run_tag = os.path.splitext(os.path.basename(args.out))[0]

    cfg = DataConfig(cadence="3h", window=3, future_met=True, courant=True,
                     relief=True, emission_features=True, transport_feature=True,
                     divergence=True, multitask=True, topology="full", neighbors=8,
                     min_driver_coverage=args.min_driver_coverage,
                     pressure_levels=args.levels)
    fields = SaudiFields(cfg, verbose=True)
    asm = FeatureAssembler(fields, cfg)
    edge_index, pos, _ = build_masked_graph(fields.mask, neighbors=cfg.neighbors)
    edge_index, pos = edge_index.to(device), pos.to(device)

    tgt = DustTarget(args.metar_dir) if args.metar_dir else DustTarget()
    labels, valid, extra = tgt.align(pd.to_datetime(fields.times, utc=True))
    node_idx, kept = tgt.node_indices(fields.mask)
    names = extra["threshold_names"]
    cols = [extra["station_ids"].index(s) for s in kept]
    labels, valid = labels[:, cols], valid[:, cols]

    # Boundary vs interior is a statement about which cells can be PREDICTED, not
    # about which observations are useful.  Boundary cells have a truncated stencil
    # and (before the domain extension) no inflow, so they make poor targets -- but
    # they are where imported dust arrives first, and 42% of storms are imported.
    # Measured: an interior storm is preceded by a northern station's storm 48.8%
    # of the time within 24 h, against 21.8% from the south, so the lead is
    # directional transport rather than widespread dust.
    #
    # `score_sel` therefore restricts the LOSS and the SCORING to interior cells,
    # while every station stays available as an observational input.
    # Ring depth: how many successive 8-neighbour erosions a cell survives.
    # depth 1 is the outer boundary, higher is deeper inside the domain.
    h, w = fields.mask.shape
    depth = np.zeros_like(fields.mask, dtype=int)
    cur = fields.mask.copy()
    d = 0
    while cur.any():
        d += 1
        depth[cur] = d
        pad = np.pad(cur, 1, constant_values=False)
        nb = sum(pad[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc].astype(int)
                 for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
        cur = cur & (nb == 8)
    interior = depth > args.buffer_rings
    by_id = {s_["id"]: s_ for s_ in tgt.stations}
    interior_sel = np.array([k for k, sid in enumerate(kept)
                             if interior[by_id[sid]["row"], by_id[sid]["col"]]])
    boundary = [sid for k, sid in enumerate(kept) if k not in set(interior_sel.tolist())]

    if args.interior_only:
        print(f"\ninterior-only: {int(interior.sum())}/{int(fields.mask.sum())} cells "
              f"have a full 8-neighbour stencil")
        if args.upwind_obs:
            print(f"  scoring {len(interior_sel)} interior stations; the "
                  f"{len(boundary)} boundary stations stay as UPWIND INPUTS: "
                  f"{', '.join(boundary)}")
        else:
            print(f"  dropped {len(boundary)} boundary stations entirely: "
                  f"{', '.join(boundary)}")
            labels, valid = labels[:, interior_sel], valid[:, interior_sel]
            node_idx = node_idx[interior_sel]
            kept = [kept[k] for k in interior_sel]
            interior_sel = np.arange(len(kept))
    else:
        interior_sel = np.arange(len(kept))

    # Ordinal grade = how many nested thresholds the observation crosses, so
    # 0 clear / 1 dusty / 2 dust_storm / 3 severe. Derived from the same labels
    # the binary path uses, which is what keeps the two directly comparable.
    grades = grades_from_labels(labels, EVENT_THRESHOLDS)
    n_grades = len(EVENT_THRESHOLDS) + 1

    print(f"\nstations in graph: {len(kept)} -> {len(set(node_idx))} distinct cells")
    print("observed base rates:")
    print(DustTarget.summarise(labels, valid, names), flush=True)

    max_h = max(HORIZON_STEPS.values())

    def split(y0, y1):
        sel = np.flatnonzero((fields.years >= y0) & (fields.years <= y1))
        return np.arange(max(int(sel[0]), cfg.window - 1),
                         min(int(sel[-1]), fields.n_steps - max_h - 1))

    tr = split(*args.train_years)
    va = split(*args.val_years)
    te = split(*args.test_years)
    print(f"split years: train {args.train_years} val {args.val_years} "
          f"test {args.test_years}")
    print(f"\nsamples: train {len(tr):,} | val {len(va):,} | test {len(te):,}")

    held = (set(rng.permutation(len(kept))[:max(1, len(kept) // 3)].tolist())
            if args.loso else set())
    scored = set(interior_sel.tolist())
    train_st = np.array([i for i in sorted(scored) if i not in held])
    eval_st = (np.array([i for i in sorted(held) if i in scored]) if held
               else np.array(sorted(scored)))
    obs_st = np.arange(len(kept)) if args.upwind_obs else train_st
    print(f"train stations {len(train_st)} | eval stations {len(eval_st)}"
          f"{' (UNSEEN in training)' if held else ''}\n", flush=True)

    # ------------------------------------------------------------------ #
    # Current observed dust state as an input (optional).
    #
    # The classifier lost the residual head's implicit persistence anchor, and it
    # shows: at 3 h it scores below "it is dusty now, so it will be dusty in 3 h"
    # (TS 0.151 vs 0.240).  The missing ingredient is that the model never sees the
    # station's own current visibility -- only gridded CAMS and ERA5.
    #
    # Channels are written at station cells only and left zero elsewhere, with a
    # has-observation flag so the network can tell "no dust reported" from "no
    # station here".  During training each station's observation is hidden with
    # probability `--obs-dropout`, which forces the model to keep a working
    # gridded-only pathway; otherwise it would lean entirely on the observation and
    # be useless at the 225 cells that have none.  Only time `t` is ever read, so
    # no future information enters.
    # ------------------------------------------------------------------ #
    N_NODES = fields.n_nodes
    log_beta = np.nan_to_num(np.clip(extra["log_beta"][:, cols], -1.0, 4.5),
                             nan=0.0).astype("float32")
    dust_code_now = extra["dust_code"][:, cols]
    n_obs_ch = 3 if args.obs_input else 0

    def obs_channels(idx, stations, drop_p=0.0, gen=None):
        """[N, 3] = (log extinction, has-observation, dust code) at time `idx`."""
        ch = np.zeros((N_NODES, 3), dtype="float32")
        if not args.obs_input:
            return ch[:, :0]
        ok = valid[idx][stations].copy()
        if drop_p > 0 and gen is not None:
            ok &= gen.random(ok.shape) > drop_p
        if not ok.any():
            return ch
        nodes = node_idx[stations][ok]
        # Several stations can share a cell; keep the worst visibility (max beta).
        np.maximum.at(ch[:, 0], nodes, log_beta[idx][stations][ok])
        ch[nodes, 1] = 1.0
        np.maximum.at(ch[:, 2], nodes, dust_code_now[idx][stations][ok].astype("float32"))
        return ch

    results = {}
    for hname, step in HORIZON_STEPS.items():
        print(f"===== horizon {hname}: meteorology supplied at t+{hname} =====",
              flush=True)

        def feats(idxs, _step=step, drop_p=0.0, gen=None, stations=None):
            base = np.stack([asm.assemble(int(i), _step) for i in idxs])
            if not args.obs_input:
                return base
            st = np.arange(len(kept)) if stations is None else stations
            obs = np.stack([obs_channels(int(i), st, drop_p, gen) for i in idxs])
            return np.concatenate([base, obs], axis=-1)

        n_ch = asm.n_channels + n_obs_ch
        # One softmax over grades, or one sigmoid per threshold.
        n_out = n_grades if args.ordinal else len(names)
        sub = feats(tr[::max(1, len(tr) // 80)]).reshape(-1, n_ch)
        if args.transport:
            lead_h = step * 3
            k_hops = hops_for_horizon(lead_h)
            emis_cols = [i for i, c in enumerate(asm.channels)
                         if c in ("ustar_+1", "log_emission_+1", "emis_tendency_+1",
                                  "speed_+1", "rel_elev")]
            model = DustTransportGNN(n_ch, n_out, emis_cols, args.hidden,
                                     k_hops, args.dropout).to(device)
            print(f"  DustTransportGNN: {k_hops} hops = {k_hops*76} km reach "
                  f"(dust travels ~{5.8*lead_h*3.6:.0f} km at p90 wind over {lead_h} h)",
                  flush=True)
        else:
            model = StationDustModel(n_ch, n_out, args.hidden,
                                     args.hops, args.dropout).to(device)
        model.set_stats(sub.mean(0), sub.std(0))

        if args.ordinal:
            gw = inverse_frequency_weights(
                grades[:, train_st][valid[:, train_st]], n_grades, cap=200.0)
            sord = SORDLoss(n_grades, class_weight=gw).to(device)
            print(f"  ordinal head: {n_grades} grades, SORD loss, "
                  f"class weights {np.round(gw, 2).tolist()}", flush=True)

            def loss_of(out, sel, stations, _step):
                y = torch.from_numpy(
                    grades[sel + _step][:, stations]).to(device)
                m = torch.from_numpy(valid[sel + _step][:, stations]).to(device)
                return sord(out, y, m), int(m.sum())
        else:
            pw = []
            for k in range(len(names)):
                lab = labels[:, train_st, k][valid[:, train_st]]
                r = max(lab.mean(), 1e-6)
                pw.append(min((1 - r) / r, 200.0))
            lossfn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pw, dtype=torch.float32, device=device),
                reduction="none")

            def loss_of(out, sel, stations, _step):
                y = torch.from_numpy(
                    labels[sel + _step][:, stations].astype("float32")).to(device)
                m = torch.from_numpy(
                    valid[sel + _step][:, stations]).to(device).unsqueeze(-1).expand_as(y)
                return (lossfn(out, y) * m).sum() / m.sum().clamp(min=1), int(m.sum())

        def event_probs(out):
            """Model output -> P(event) per threshold, in `names` order."""
            if args.ordinal:
                return cumulative_probabilities(out, EVENT_THRESHOLDS, names)
            return torch.sigmoid(out)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        def run(idxs, stations, train, _step=step):
            model.train(train)
            order = rng.permutation(len(idxs)) if train else np.arange(len(idxs))
            tot, nb, nlab = 0.0, 0, 0
            for b0 in range(0, len(order), args.batch):
                sel = idxs[order[b0:b0 + args.batch]]
                if not len(sel):
                    continue
                x = torch.from_numpy(
                    feats(sel, _step, args.obs_dropout if train else 0.0,
                          rng, obs_st)).to(device)
                B, N, C = x.shape
                offs = (torch.arange(B, device=device) * N).view(1, -1, 1)
                ei = (edge_index.unsqueeze(1) + offs).reshape(2, -1)
                out = model(x.reshape(B * N, C), ei, pos.repeat(B, 1)).view(B, N, -1)
                out = out[:, node_idx[stations], :]
                loss, n_valid = loss_of(out, sel, stations, _step)
                nlab += n_valid
                if train:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tot += float(loss.detach()); nb += 1
            # A silent zero here means the station record does not cover this split.
            if nlab == 0:
                raise RuntimeError(
                    "no valid station labels in this split -- does the METAR record "
                    "cover these years? run download_metar_saudi.py for 2003-2024.")
            return tot / max(nb, 1)

        best, state, bad, t0 = float("inf"), None, 0, time.time()
        for ep in range(args.epochs):
            trl = run(tr, train_st, True)
            with torch.no_grad():
                vl = run(va, train_st, False)
            if vl < best - 1e-4:
                best, bad = vl, 0
                state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            print(f"  ep {ep+1:3d}/{args.epochs}  train {trl:.4f}  val {vl:.4f}"
                  f"{'  *' if bad == 0 else ''}", flush=True)
            if bad >= PATIENCE:
                print(f"  early stop (no val gain in {PATIENCE} epochs)", flush=True)
                break
        if state:
            model.load_state_dict(state)

        torch.save({"state": model.state_dict(), "n_ch": n_ch, "names": names,
                    "n_out": n_out, "ordinal": bool(args.ordinal),
                    "channels": list(asm.channels), "tag": run_tag},
                   os.path.join("results", f"station_model_{run_tag}_{hname}.pt"))

        # Score twice: with the current observation available, and with it fully
        # withheld.  The second is what the model would deliver at the 225 cells
        # that have no station, so the gap between them is the generalisation cost.
        model.eval()
        modes = [("obs", 0.0), ("no-obs", 1.0)] if args.obs_input else [("gridded", 0.0)]
        for mode, dp in modes:
            P, Y, Mk = [], [], []
            with torch.no_grad():
                for b0 in range(0, len(te), args.batch):
                    sel = te[b0:b0 + args.batch]
                    x = torch.from_numpy(
                        feats(sel, step, dp, np.random.default_rng(0),
                              np.arange(len(kept)))).to(device)
                    B, N, C = x.shape
                    offs = (torch.arange(B, device=device) * N).view(1, -1, 1)
                    ei = (edge_index.unsqueeze(1) + offs).reshape(2, -1)
                    out = model(x.reshape(B * N, C), ei, pos.repeat(B, 1)).view(B, N, -1)
                    P.append(event_probs(out[:, node_idx[eval_st], :]).cpu().numpy())
                    Y.append(labels[sel + step][:, eval_st])
                    Mk.append(valid[sel + step][:, eval_st])
            P = np.concatenate(P); Y = np.concatenate(Y).astype(bool)
            Mk = np.concatenate(Mk)[..., None].repeat(len(names), -1)

            # Persist the raw 3-hourly scores before any aggregation, so a
            # re-score at a different threshold, a different operating point or a
            # new metric costs seconds instead of a retrain.  Every number in the
            # results table is recoverable from this file; nothing here is.
            pred_path = os.path.join("results", "preds",
                                     f"station_{run_tag}_{hname}_{mode}.npz")
            os.makedirs(os.path.dirname(pred_path), exist_ok=True)
            np.savez_compressed(
                pred_path, prob=P.astype("float32"), obs=Y, valid=Mk,
                times=fields.times[te + step].astype("datetime64[s]").astype("int64"),
                thresholds=np.array(names), stations=np.array([kept[i] for i in eval_st]),
                horizon=hname, mode=mode, daily=bool(args.daily))

            if args.daily:
                # A dust DAY at a station: any storm hour within the calendar day.
                # This is the unit AI-DUST (Fig. 3) and WMO SDS-WAS actually report;
                # scoring 3-hourly station-hours is a 4.5x rarer event and is not
                # comparable to their published threat scores.
                vt = pd.to_datetime(fields.times[te + step], utc=True)
                day = pd.Index(vt.date)
                uniq, inv = np.unique(day.values, return_inverse=True)
                agg = lambda a, how: np.stack(
                    [how(a[inv == d], axis=0) for d in range(len(uniq))])
                P = agg(P, np.max)          # day's forecast = its peak probability
                Y = agg(Y, np.any)          # day is a dust day if any hour was
                Mk = agg(Mk, np.any)

            for k, nm in enumerate(names):
                p, y = P[..., k][Mk[..., k]], Y[..., k][Mk[..., k]]
                if y.sum() < 5:
                    continue
                thr = np.quantile(p, 1.0 - y.mean())     # frequency matching
                results[f"{hname}|{nm}|{mode}"] = contingency(p >= thr, y)
        print(f"  trained in {(time.time()-t0)/60:.1f} min\n", flush=True)

    print("=== dust-event forecast vs OBSERVED station visibility (test 2021-2024) ===")
    print("  obs    = current station visibility available (nowcast-assisted)")
    print("  no-obs = observation withheld: what the 225 station-less cells would get")
    print("           -- this is the column comparable to published forecast scores\n")
    def fmt(v, w=8, d=3):
        return f"{v:>{w}.{d}f}" if isinstance(v, float) and v == v else f"{'--':>{w}}"

    print(f"{'horizon':<9}{'event':<12}{'mode':<8}{'base':>8}{'TS':>8}{'ETS':>8}"
          f"{'SEDI':>8}{'POD':>8}{'FAR':>8}{'events':>9}")
    for key, sc in results.items():
        parts = key.split("|")
        h, nm, mode = (parts + ["-"])[:3]
        print(f"{h:<9}{nm:<12}{mode:<8}"
              f"{sc['base_rate']*100:>7.3f}%{fmt(sc['TS'])}{fmt(sc['ETS'])}"
              f"{fmt(sc['SEDI'])}{fmt(sc['POD'])}{fmt(sc['FAR'])}{sc['events']:>9,}")

    print("\nRead DOWN the SEDI column, ACROSS the TS column.")
    print("  TS, ETS, POD and FAR all fall with the base rate whether or not the")
    print("  forecast got worse, so they compare models at ONE threshold and")
    print("  nothing else. Verified in experiments/verify_metrics.py: at constant")
    print("  forecast quality TS drops 11.4x from the `dusty` to the `severe`")
    print("  grade while SEDI does not move at all.")
    print("\nreference points, with their base rates, since TS without one is meaningless:")
    print("  Sarafian 2023 (Israel, 24 h, stations):  TS 0.553 implied, base rate 13%")
    print("    -- the one genuinely comparable published result; compare at `dusty`")
    print("  AI-DUST (East Asia, day 1, stations):    TS 0.33,  base rate unstated")
    print("  WMO SDS-WAS operational:                 TS 0.31 @24h -- UNTRACEABLE,")
    print("    could not be sourced primarily; Middle East sub-region POD runs")
    print("    0.19-0.48 against >0.68 over the Sahara, so the regional bar is")
    print("    probably lower. Do not quote 0.31 without chasing the AEMET report.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"results": results, "stations": kept, "loso": sorted(held),
                   "args": vars(args)}, fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
