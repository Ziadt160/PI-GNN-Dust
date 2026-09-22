"""Does vertical structure help, and does an ordinal head help? Measure, don't assume.

Two changes were added on the strength of published evidence rather than our own:

  --levels    ERA5 pressure-level stability (t_500, t_700-500, Ri, shear, jet).
              Xu et al. 2026 find pressure-level quantities in their top 3 SHAP
              features at every horizon, and vertical mixing as the dominant
              mechanism. We had no vertical information at all.
  --ordinal   one SORD-trained severity grade instead of three independent binary
              heads. Sarafian et al. 2023 use this for the same task; our own
              `severe` head scores TS 0.000 at a 0.070% base rate.

Neither is credited until it is measured HERE, on our data, against our baseline.
This driver runs the arms, holds everything else fixed, and prints the deltas.

Seeding
-------
The handover records a measured sd of 0.033 TS between identical configurations
when the process was seeded once. Most effects in this project are smaller than
that, so a single-seed comparison cannot distinguish a real gain from noise.
Each arm is therefore run over several seeds and reported as mean +- sd, and an
effect inside the noise band is called inside the noise band.

Run (after download_era5_levels.py has finished):
    python -u experiments/ablation_levels.py --epochs 60 --seeds 3
    python -u experiments/ablation_levels.py --epochs 6 --seeds 1 --arms base ordinal
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

# Everything the arms share. Kept in one place so an arm cannot differ from
# another by an accident of the command line.
COMMON = ["--obs-input", "--upwind-obs", "--interior-only", "--daily",
          "--transport", "--hidden", "64", "--buffer-rings", "2"]

ARMS = {
    "base":           [],
    "levels":         ["--levels"],
    "ordinal":        ["--ordinal"],
    "levels+ordinal": ["--levels", "--ordinal"],
}

# The row to read. `dusty` is the grade whose base rate (~13% daily) matches the
# published studies; `severe` is the one the ordinal head exists to rescue.
REPORT = [("24h", "dusty", "no-obs"),
          ("24h", "dust_storm", "no-obs"),
          ("24h", "severe", "no-obs"),
          ("48h", "dusty", "no-obs")]


EXPECTED_HORIZONS = {"3h", "24h", "48h"}


def arm_complete(path):
    """Has this arm finished EVERY horizon, or only some of them?

    `train_station_dust.py` now writes its results file after each horizon so a
    late failure cannot discard earlier work. The cost is that "the file exists"
    no longer means "the arm is done" -- a partial file appears within minutes of
    the run starting. Skipping on existence alone would silently drop the 48 h
    horizon of any arm that was interrupted.

    Files written before `complete` was added fall back to deriving the finished
    horizons from the result keys.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return False       # half-written or unreadable: redo it
    done = set(d.get("complete") or
               {k.split("|")[0] for k in d.get("results", {})})
    return EXPECTED_HORIZONS <= done


def run_arm(arm, seed, epochs, out_dir, python):
    out = os.path.join(out_dir, f"abl_{arm.replace('+', '_')}_s{seed}.json")
    if arm_complete(out):
        print(f"  skip (complete): {os.path.basename(out)}", flush=True)
        return out
    if os.path.exists(out):
        print(f"  redo (partial): {os.path.basename(out)} -- "
              f"resuming from its checkpoints", flush=True)
    cmd = ([python, "-u", os.path.join("experiments", "train_station_dust.py"),
            "--epochs", str(epochs), "--seed", str(seed), "--out", out]
           + COMMON + ARMS[arm])
    print(f"  {arm} seed {seed}: {' '.join(cmd[3:])}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"arm {arm} seed {seed} failed (exit {proc.returncode})")
    return out


def load(path):
    with open(path) as fh:
        return json.load(fh)["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--metric", default="SEDI",
                    help="SEDI compares across thresholds; TS only within one")
    ap.add_argument("--out-dir", default=os.path.join("results", "ablation"))
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; choose from {list(ARMS)}")
    os.makedirs(args.out_dir, exist_ok=True)

    scores = {}
    for arm, seed in itertools.product(args.arms, range(args.seeds)):
        path = run_arm(arm, seed, args.epochs, args.out_dir, args.python)
        scores.setdefault(arm, []).append(load(path))

    print(f"\n=== ablation: {args.metric}, mean +- sd over {args.seeds} seed(s) ===")
    print("SEDI is base-rate independent, so the `severe` column is readable here")
    print("even though its TS is not. A delta smaller than the seed sd is noise.\n")

    head = f"{'arm':<16}" + "".join(f"{f'{h} {n}':>22}" for h, n, _ in REPORT)
    print(head)
    print("-" * len(head))

    table = {}
    for arm in args.arms:
        cells = []
        for h, nm, mode in REPORT:
            key = f"{h}|{nm}|{mode}"
            vals = [r[key][args.metric] for r in scores[arm]
                    if key in r and r[key].get(args.metric) is not None
                    and r[key][args.metric] == r[key][args.metric]]
            if not vals:
                cells.append(f"{'--':>22}")
                table[(arm, key)] = (np.nan, np.nan)
                continue
            m, s = float(np.mean(vals)), float(np.std(vals))
            table[(arm, key)] = (m, s)
            cells.append(f"{m:>15.3f} +-{s:<5.3f}")
        print(f"{arm:<16}" + "".join(cells))

    if "base" in args.arms and len(args.arms) > 1:
        print("\ndelta vs base, with the seed sd of the base arm as the noise floor:")
        for arm in args.arms:
            if arm == "base":
                continue
            parts = []
            for h, nm, mode in REPORT:
                key = f"{h}|{nm}|{mode}"
                m, _ = table[(arm, key)]
                b, bs = table[("base", key)]
                if not (m == m and b == b):
                    parts.append(f"{h}/{nm}: --")
                    continue
                d = m - b
                verdict = "inside noise" if abs(d) <= max(bs, 1e-9) else "REAL"
                parts.append(f"{h}/{nm}: {d:+.3f} ({verdict})")
            print(f"  {arm:<16}" + " | ".join(parts))

    summary = os.path.join(args.out_dir, "summary.json")
    with open(summary, "w") as fh:
        json.dump({"metric": args.metric, "epochs": args.epochs,
                   "seeds": args.seeds, "report_rows": REPORT,
                   "mean_sd": {f"{a}|{k}": v for (a, k), v in table.items()}},
                  fh, indent=2)
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
