"""Can this model actually be used? The POD/FAR trade-off across every threshold.

Every score reported so far sits at ONE operating point -- the frequency-matched
threshold, where the model forecasts exactly as many events as are observed.  That
point maximises nothing in particular; it just removes the calibration constant so
models can be compared fairly.

Deployability is a different question.  A warning system does not care about the
frequency-matched threshold: it cares whether *some* threshold gives an acceptable
combination of detection and false alarms.  Lowering the threshold raises POD and
raises FAR together, and the whole curve -- not one point on it -- says whether a
usable setting exists.

Reference points for judging the curve:
  * A public warning needs high POD; missing storms is a safety failure.
  * Above roughly 80-90% FAR, operational forecasters stop acting on alerts
    (the "cry wolf" effect), so high POD bought at any FAR is not a real option.
  * WMO SDS-WAS operational models sit near TS 0.31 at 24 h.

Run:  python -u experiments/operating_points.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.data.saudi_dataset import DataConfig, SaudiFields        # noqa: E402
from src.data.station_target import DustTarget                    # noqa: E402
from src.data.topologies import FeatureAssembler                  # noqa: E402
from src.utils.graph import build_masked_graph                    # noqa: E402
from experiments.train_station_dust import StationDustModel, HORIZON_STEPS  # noqa: E402


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DataConfig(cadence="3h", window=3, future_met=True, courant=True,
                     relief=True, emission_features=True, transport_feature=True,
                     divergence=True, multitask=True, topology="full", neighbors=8)
    fields = SaudiFields(cfg, verbose=False)
    asm = FeatureAssembler(fields, cfg)
    edge_index, pos, _ = build_masked_graph(fields.mask, neighbors=8)
    edge_index, pos = edge_index.to(device), pos.to(device)

    tgt = DustTarget()
    labels, valid, extra = tgt.align(pd.to_datetime(fields.times, utc=True))
    node_idx, kept = tgt.node_indices(fields.mask)
    names = extra["threshold_names"]
    cols = [extra["station_ids"].index(s) for s in kept]
    labels, valid = labels[:, cols], valid[:, cols]
    n_obs_ch = 3          # the trained models carry the observation channels

    sel_years = (fields.years >= 2021) & (fields.years <= 2024)
    te = np.flatnonzero(sel_years)
    te = te[(te >= cfg.window - 1) & (te < fields.n_steps - 17)]

    out = {}
    for hname, step in HORIZON_STEPS.items():
        ckpt = torch.load(os.path.join("results", f"station_model_{hname}.pt"),
                          map_location=device, weights_only=False)
        model = StationDustModel(ckpt["n_ch"], len(names), 48, 2, 0.3).to(device)
        model.load_state_dict(ckpt["state"]); model.eval()

        # Observation channels withheld: the deployable, station-free configuration.
        P, Y, M = [], [], []
        with torch.no_grad():
            for b0 in range(0, len(te), 64):
                s = te[b0:b0 + 64]
                base = np.stack([asm.assemble(int(i), step) for i in s])
                x = np.concatenate(
                    [base, np.zeros((*base.shape[:2], n_obs_ch), "float32")], -1)
                x = torch.from_numpy(x).to(device)
                B, N, C = x.shape
                offs = (torch.arange(B, device=device) * N).view(1, -1, 1)
                ei = (edge_index.unsqueeze(1) + offs).reshape(2, -1)
                o = model(x.reshape(B * N, C), ei, pos.repeat(B, 1)).view(B, N, -1)
                P.append(torch.sigmoid(o[:, node_idx, :]).cpu().numpy())
                Y.append(labels[s + step]); M.append(valid[s + step])
        P = np.concatenate(P); Y = np.concatenate(Y).astype(bool)
        M = np.concatenate(M)

        k = names.index("dust_storm")
        p, y = P[..., k][M], Y[..., k][M]
        base_rate = y.mean()

        print(f"\n===== {hname} dust storm (visibility < 1 km) — observation withheld =====")
        print(f"  {len(y):,} forecast points, {int(y.sum())} observed storms "
              f"(base rate {base_rate*100:.3f}%)")
        print(f"  {'target POD':>11}{'threshold':>11}{'POD':>8}{'FAR':>8}{'TS':>8}"
              f"{'alerts/yr':>11}{'verdict':>26}")
        rows = []
        for target_pod in (0.50, 0.60, 0.70, 0.80, 0.90):
            # Threshold that achieves this detection rate.
            thr = np.quantile(p[y], 1.0 - target_pod)
            pe = p >= thr
            nc = int((pe & y).sum()); nf = int((pe & ~y).sum()); no = int((~pe & y).sum())
            pod = nc / max(nc + no, 1); far = nf / max(nc + nf, 1)
            ts = nc / max(nc + nf + no, 1)
            alerts = (nc + nf) / (len(y) / len(kept)) * (365 * 24 / 3) / (365 * 4 / 4)
            alerts_per_station_year = (nc + nf) / len(kept) / 4
            verdict = ("unusable: cry-wolf" if far > 0.90 else
                       "marginal triage only" if far > 0.80 else
                       "possible triage" if far > 0.60 else "usable")
            rows.append(dict(target_pod=target_pod, thr=float(thr), POD=pod, FAR=far,
                             TS=ts, alerts_per_station_year=alerts_per_station_year))
            print(f"  {target_pod:>11.0%}{thr:>11.3f}{pod:>8.3f}{far:>8.3f}{ts:>8.3f}"
                  f"{alerts_per_station_year:>11.0f}{verdict:>26}")
        out[hname] = rows

    with open("results/operating_points.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nwrote results/operating_points.json")


if __name__ == "__main__":
    main()
