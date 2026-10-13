"""Which inputs actually drive the dust-storm forecast?

Method: permutation importance on the trained model.  Each input channel is
shuffled across the time axis -- destroying its association with the target while
leaving the spatial structure, the graph and every other channel intact -- and the
loss of skill is measured.  A channel the model relies on collapses performance
when scrambled; a channel it ignores does nothing.

This is model-agnostic and needs no retraining, which matters because retraining
30 ablations would cost hours and confound the answer with run-to-run variance
(measured at 0.033 TS, larger than most effects of interest).

Scored with **average precision** (area under precision-recall) rather than ROC
AUC or a thresholded score.  At a 1% base rate, ROC AUC is dominated by the
overwhelming negative class and flatters everything; average precision tracks
performance on the positives, which is the whole question.

Two hypotheses this tests directly:

* If **CAMS PM10** ranks low, that independently corroborates the finding that the
  reanalysis carries little dust-storm information (it is 15-49x too low during
  observed storms).
* If **wind at the target time** dominates everything else, the model is
  effectively a wind threshold with extra steps, and the graph transport and
  emission machinery are not earning their place.

Run:  python -u experiments/station_dust/feature_importance.py --horizon 24h
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.data.saudi_dataset import DataConfig, SaudiFields        # noqa: E402
from src.data.station_target import DustTarget                    # noqa: E402
from src.data.topologies import FeatureAssembler                  # noqa: E402
from src.utils.graph import build_masked_graph                    # noqa: E402
from experiments.station_dust.train_station_dust import StationDustModel, HORIZON_STEPS  # noqa: E402

OBS_CHANNELS = ["obs_log_beta", "obs_has_observation", "obs_dust_code"]


def average_precision(scores, labels):
    """Area under the precision-recall curve -- the right summary at a 1% base rate."""
    order = np.argsort(-scores)
    y = labels[order].astype(float)
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    return float((precision * y).sum() / n_pos) if n_pos else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", default="24h", choices=list(HORIZON_STEPS))
    ap.add_argument("--event", default="dust_storm")
    ap.add_argument("--stride", type=int, default=2, help="subsample test timesteps")
    ap.add_argument("--repeats", type=int, default=3, help="permutation repeats")
    ap.add_argument("--out", default="results/feature_importance.json")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DataConfig(cadence="3h", window=3, future_met=True, courant=True,
                     relief=True, emission_features=True, transport_feature=True,
                     divergence=True, multitask=True, topology="full", neighbors=8)
    fields = SaudiFields(cfg, verbose=False)
    asm = FeatureAssembler(fields, cfg)
    edge_index, pos, _ = build_masked_graph(fields.mask, neighbors=8)
    edge_index, pos = edge_index.to(dev), pos.to(dev)

    tgt = DustTarget()
    labels, valid, extra = tgt.align(pd.to_datetime(fields.times, utc=True))
    node_idx, kept = tgt.node_indices(fields.mask)
    names = extra["threshold_names"]
    cols = [extra["station_ids"].index(s) for s in kept]
    labels, valid = labels[:, cols], valid[:, cols]

    # interior cells only, matching the trained model
    h, w = fields.mask.shape
    pad = np.pad(fields.mask, 1, constant_values=False)
    nbr = sum(pad[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc].astype(int)
              for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
    interior = fields.mask & (nbr == 8)
    by = {s["id"]: s for s in tgt.stations}
    sel = [k for k, sid in enumerate(kept) if interior[by[sid]["row"], by[sid]["col"]]]
    labels, valid, node_idx = labels[:, sel], valid[:, sel], node_idx[sel]

    step = HORIZON_STEPS[args.horizon]
    k_ev = names.index(args.event)
    te = np.flatnonzero((fields.years >= 2021) & (fields.years <= 2024))
    te = te[(te >= cfg.window - 1) & (te < fields.n_steps - 17)][::args.stride]

    ck = torch.load(os.path.join("results", f"station_model_{args.horizon}.pt"),
                    map_location=dev, weights_only=False)
    model = StationDustModel(ck["n_ch"], len(names), 48, 2, 0.3).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()

    channels = list(asm.channels) + OBS_CHANNELS[:ck["n_ch"] - asm.n_channels]
    assert len(channels) == ck["n_ch"], (len(channels), ck["n_ch"])

    # Materialise the test features once; permutation then costs no reassembly.
    X = np.stack([asm.assemble(int(i), step) for i in te])          # [T, N, C]
    if ck["n_ch"] > asm.n_channels:
        X = np.concatenate([X, np.zeros((*X.shape[:2], ck["n_ch"] - asm.n_channels),
                                        "float32")], axis=-1)
    Y = labels[te + step][..., k_ev].astype(bool)
    M = valid[te + step]

    @torch.no_grad()
    def score(x):
        out = []
        for b0 in range(0, len(x), 64):
            xb = torch.from_numpy(x[b0:b0 + 64]).to(dev)
            B, N, C = xb.shape
            offs = (torch.arange(B, device=dev) * N).view(1, -1, 1)
            ei = (edge_index.unsqueeze(1) + offs).reshape(2, -1)
            o = model(xb.reshape(B * N, C), ei, pos.repeat(B, 1)).view(B, N, -1)
            out.append(torch.sigmoid(o[:, node_idx, k_ev]).cpu().numpy())
        p = np.concatenate(out)
        return average_precision(p[M], Y[M])

    base = score(X)
    base_rate = Y[M].mean()
    print(f"horizon {args.horizon} | event {args.event} | {M.sum():,} station-times "
          f"| base rate {base_rate*100:.2f}%")
    print(f"baseline average precision = {base:.4f}  "
          f"(a random forecast scores {base_rate:.4f})\n")

    rng = np.random.default_rng(0)
    results = []
    for c, name in enumerate(channels):
        drops = []
        for _ in range(args.repeats):
            Xp = X.copy()
            Xp[:, :, c] = Xp[rng.permutation(len(Xp)), :, c]   # shuffle across time
            drops.append(base - score(Xp))
        results.append({"channel": name, "drop": float(np.mean(drops)),
                        "sd": float(np.std(drops))})
        print(f"  {name:<22} drop {np.mean(drops):+.4f}  (sd {np.std(drops):.4f})",
              flush=True)

    results.sort(key=lambda r: -r["drop"])
    print(f"\n{'rank':<5}{'feature':<24}{'AP drop':>10}{'% of baseline':>15}")
    for i, r in enumerate(results, 1):
        print(f"{i:<5}{r['channel']:<24}{r['drop']:>+10.4f}{r['drop']/base*100:>14.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"horizon": args.horizon, "event": args.event,
                   "baseline_ap": base, "base_rate": float(base_rate),
                   "importances": results}, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
