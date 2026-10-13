"""Recompute the metric table from cached predictions -- no retraining.

`benchmark_aidust.py` writes every model's raw test predictions to
``results/preds/<model>.npz``.  This script reads them back and rebuilds the
results table with whatever metrics `src.metrics.aidust_metrics` currently
defines, so adding a score (ETS, HSS, FGE, a new threshold) costs seconds
instead of a full sweep.

  python experiments/saudi_pm10/rescore.py
  python experiments/saudi_pm10/rescore.py --preds results/preds --out results/rescored.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src.metrics import aidust_metrics as M            # noqa: E402
from src.metrics import literature                     # noqa: E402


MERGED = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/saudi_merged.nc"


def persistence_baseline(test_idxs, true, active):
    """True one-step persistence: the state at t, scored against the target at t+1.

    Not obtainable by shifting the cached arrays -- test samples are strided by
    ``--eval-stride`` (4 steps = 12 h apart), so shifting them would measure a 12 h
    persistence against models doing a 3 h step, and understate the floor.  The
    input state is read back from the source field instead.
    """
    import xarray as xr
    ds = xr.open_dataset(MERGED)
    mask = ds.valid_mask.values.astype(bool)
    pm = ds.pm10.values.reshape(ds.sizes["valid_time"], -1)[:, np.flatnonzero(mask.ravel())]
    ds.close()
    persist = np.log1p(np.clip(pm[np.asarray(test_idxs)] * 1e9, 0, None)).astype("float32")
    return M.evaluate(persist, true, emission_active=active, label="persistence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds")
    ap.add_argument("--out", default="results/rescored.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.preds, "*.npz")))
    if not files:
        sys.exit(f"no cached predictions in {args.preds} -- run benchmark_aidust.py first")

    rows, persistence = [], None
    for f in files:
        z = np.load(f)
        name = os.path.splitext(os.path.basename(f))[0]
        pred, true = z["pred"][..., 0], z["true"][..., 0]
        active = z["emis_active"]
        rows.append(M.evaluate(pred, true, emission_active=active, label=name))
        if persistence is None:
            persistence = persistence_baseline(z["test_idxs"], true, active)

    table = ([persistence] if persistence else []) + rows
    M.add_skill_scores(table)
    head, rest = table[:1], sorted(table[1:],
                                   key=lambda r: -(r.get("SS_TS", -9) or -9))
    table = head + rest
    print(M.format_table(table))

    print("\n\n" + "=" * 96)
    print(literature.format_comparison(table))

    with open(args.out, "w") as fh:
        json.dump({"rows": table, "references": literature.REFERENCES},
                  fh, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
