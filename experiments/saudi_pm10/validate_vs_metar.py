"""Score the models against real observations instead of CAMS reanalysis.

Everything in `benchmark_aidust.py` is scored against CAMS, which is a model.  This
script scores the same cached predictions against **airport visibility**, the
observation operational dust forecasting is actually verified against.

The threshold problem this exposes
----------------------------------
Our CAMS-based criterion (PM10 >= 200 ug/m^3) fires on **8.5%** of cell-times.
Real dust storms, by the WMO definition (visibility < 1 km), occur in **0.27%** of
station-hours -- a factor of ~31.  The 200 ug/m^3 cut on a 0.75 deg reanalysis is
therefore not selecting dust storms; over Arabia it selects "somewhat dusty",
which is the normal state.  Any TS quoted against it measures the wrong event.

So the model threshold is not fixed here.  It is **frequency-matched**: for each
event definition, the model's threshold is set to the quantile that reproduces the
observed event rate.  That removes an arbitrary calibration constant and makes
TS/POD/FAR answer the question that matters -- *when the model says dust, is there
dust?* -- rather than rewarding a model for saying dust constantly.

Three observed-event definitions are reported, because they are not the same event:

    vis < 1.0 km    WMO duststorm           0.27% of obs   the strict definition
    vis <= 5.0 km   dust-degraded visibility  ~2-5%        closer to our base rate
    wx code         DS/SS/BLDU/BLSA/DU/SA    2.01%         observer-reported dust

Codes are reported but trusted less: a published audit found ~50% of coded DS
events had visibility above 1 km.

Usage:  python experiments/saudi_pm10/validate_vs_metar.py
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src.metrics import aidust_metrics as M         # noqa: E402
from src import paths                               # noqa: E402
from src import paths                               # noqa: E402

MERGED = paths.SAUDI_MERGED
METAR_DIR = paths.METAR_DIR
MILES_TO_KM = 1.609344
DUST_CODE_RE = "DS|SS|BLDU|BLSA|DU|SA|PO"

EVENT_DEFS = {
    "vis<1km":  ("visibility", 1.0),
    "vis<5km":  ("visibility", 5.0),
    "wxcode":   ("code", None),
}


def load_stations():
    with open(os.path.join(METAR_DIR, "stations.json")) as fh:
        return json.load(fh)


def load_metar(station_id):
    """Hourly visibility (km) and dust-code flag, indexed by UTC time."""
    path = os.path.join(METAR_DIR, f"{station_id}.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, na_values=["M", "T"])
    t = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    vis = pd.to_numeric(d["vsby"], errors="coerce") * MILES_TO_KM
    code = d["wxcodes"].fillna("").astype(str).str.contains(DUST_CODE_RE, regex=True)
    out = pd.DataFrame({"vis_km": vis.values, "dust_code": code.values}, index=t)
    return out[~out.index.isna()].sort_index()


def model_times(test_idxs, cadence_factor=1):
    """Absolute UTC timestamps of the cached evaluation samples.

    The prediction target is the state at ``t+1``, so the verification time is one
    model step after the sample's own index.
    """
    ds = xr.open_dataset(MERGED)
    tv = ds.valid_time.values
    ds.close()
    n_trim = (len(tv) // cadence_factor) * cadence_factor
    times = tv[:n_trim:cadence_factor]
    return pd.to_datetime(times[np.asarray(test_idxs) + 1], utc=True)


def node_index_map():
    """(row, col) -> node index, matching SaudiFields' node ordering."""
    ds = xr.open_dataset(MERGED)
    mask = ds.valid_mask.values.astype(bool)
    ds.close()
    n_cols = mask.shape[1]
    node_flat = np.flatnonzero(mask.ravel())
    return {(int(f // n_cols), int(f % n_cols)): i for i, f in enumerate(node_flat)}, mask


def match_observations(stations, times, tolerance="90min"):
    """Align each station's observations onto the model verification times.

    Returns ``obs[station] = DataFrame(vis_km, dust_code)`` reindexed to `times`,
    nearest-neighbour within `tolerance`; unmatched times are NaN and dropped later.
    """
    out = {}
    for s in stations:
        m = load_metar(s["id"])
        if m is None or m.empty:
            continue
        aligned = m.reindex(times, method="nearest",
                            tolerance=pd.Timedelta(tolerance))
        out[s["id"]] = aligned
    return out


def evaluate_model(pred_log, node_of_cell, stations, obs, event_kind, threshold):
    """TS/ETS/POD/FAR for one model against one observed-event definition.

    The model threshold is frequency-matched to the observed event rate, so the
    comparison is not contingent on a PM10-to-visibility conversion.
    """
    model_vals, obs_evt = [], []
    used = 0
    for s in stations:
        key = (s["row"], s["col"])
        if key not in node_of_cell or s["id"] not in obs:
            continue
        n = node_of_cell[key]
        o = obs[s["id"]]
        if event_kind == "visibility":
            ev = (o["vis_km"] < threshold).to_numpy()
            valid = o["vis_km"].notna().to_numpy()
        else:
            ev = o["dust_code"].astype("boolean").fillna(False).to_numpy(dtype=bool)
            valid = o["dust_code"].notna().to_numpy()
        if valid.sum() == 0:
            continue
        model_vals.append(pred_log[valid, n])
        obs_evt.append(ev[valid])
        used += 1

    if not model_vals:
        return None
    mv = M.to_physical(np.concatenate(model_vals))
    oe = np.concatenate(obs_evt)
    rate = oe.mean()
    if rate <= 0 or rate >= 1:
        return None

    # Frequency matching: forecast an event as often as one is observed.
    model_thr = float(np.quantile(mv, 1.0 - rate))
    # The observed side is already boolean (from visibility or a code), so the
    # contingency table is built directly rather than through threat_score, which
    # expects a concentration on both sides.
    pred_evt = mv >= model_thr
    nc = int(np.sum(pred_evt & oe))
    nf = int(np.sum(pred_evt & ~oe))
    no = int(np.sum(~pred_evt & oe))
    cn = int(oe.size - nc - nf - no)
    n_tot = oe.size
    nc_rand = (nc + nf) * (nc + no) / n_tot if n_tot else 0.0
    den = nc + nf + no
    return {
        "stations": used, "n": n_tot, "obs_rate": rate, "model_thr_ugm3": model_thr,
        "TS": nc / den if den else np.nan,
        "ETS": (nc - nc_rand) / (den - nc_rand) if (den - nc_rand) > 0 else np.nan,
        "POD": nc / (nc + no) if (nc + no) else np.nan,
        "FAR": nf / (nc + nf) if (nc + nf) else np.nan,
        "events": nc + no,
    }


def score_rollout_vs_stations(path, node_of_cell, stations, kind, thr,
                              cadence_factor=1, max_day=2):
    """Score free-running lead-day forecasts against station observations.

    This is the comparison that makes our numbers mean the same thing as the
    published ones.  WMO SDS-WAS reports TS at 24/48/72 h and AI-DUST at day 1-10;
    a 3 h single-step TS is not the same quantity and must not be set beside them.

    The cached rollout is ``[K, S, N]`` -- lead step, initialisation, node.  For
    lead day ``d`` the verification time of start ``s`` at step ``k`` is
    ``time[start_s + k + 1]``, so each (start, step) pair is matched to the station
    observation nearest that time.
    """
    z = np.load(path)
    pred, starts = z["pred"], z["starts"]
    steps_per_day = int(z.get("steps_per_day", 8))
    n_steps = pred.shape[0]

    ds = xr.open_dataset(MERGED)
    tv = ds.valid_time.values
    ds.close()
    times_all = pd.to_datetime(tv[:(len(tv) // cadence_factor) * cadence_factor:
                                  cadence_factor], utc=True)

    metar = {s["id"]: load_metar(s["id"]) for s in stations}
    out = {}
    for day in range(1, min(max_day, n_steps // steps_per_day) + 1):
        ks = range((day - 1) * steps_per_day, day * steps_per_day)
        vals, evts = [], []
        for k in ks:
            vt = times_all[np.clip(starts + k + 1, 0, len(times_all) - 1)]
            for s in stations:
                key = (s["row"], s["col"])
                m = metar.get(s["id"])
                if key not in node_of_cell or m is None or m.empty:
                    continue
                a = m.reindex(vt, method="nearest", tolerance=pd.Timedelta("90min"))
                if kind == "visibility":
                    ev, ok = (a["vis_km"] < thr).to_numpy(), a["vis_km"].notna().to_numpy()
                else:
                    ev = a["dust_code"].astype("boolean").fillna(False).to_numpy(dtype=bool)
                    ok = a["dust_code"].notna().to_numpy()
                if ok.sum():
                    vals.append(pred[k, ok, node_of_cell[key]])
                    evts.append(ev[ok])
        if not vals:
            continue
        mv = M.to_physical(np.concatenate(vals))
        oe = np.concatenate(evts)
        rate = oe.mean()
        if rate <= 0 or rate >= 1:
            continue
        t = float(np.quantile(mv, 1.0 - rate))
        pe = mv >= t
        nc = int(np.sum(pe & oe)); nf = int(np.sum(pe & ~oe)); no = int(np.sum(~pe & oe))
        den = nc + nf + no
        rand = (nc + nf) * (nc + no) / oe.size if oe.size else 0.0
        out[f"day{day}"] = {
            "TS": nc / den if den else np.nan,
            "ETS": (nc - rand) / (den - rand) if (den - rand) > 0 else np.nan,
            "POD": nc / (nc + no) if (nc + no) else np.nan,
            "FAR": nf / (nc + nf) if (nc + nf) else np.nan,
            "events": nc + no, "n": int(oe.size), "obs_rate": float(rate),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds")
    ap.add_argument("--out", default="results/metar_validation.json")
    ap.add_argument("--rollout", action="store_true",
                    help="also score cached lead-day rollouts at the stations")
    ap.add_argument("--rollout-event", default="wxcode",
                    choices=list(EVENT_DEFS), help="event definition for lead-day scoring")
    args = ap.parse_args()

    stations = load_stations()
    node_of_cell, _ = node_index_map()
    files = sorted(f for f in glob.glob(os.path.join(args.preds, "*.npz"))
                   if not f.endswith("__rollout.npz"))
    if not files:
        sys.exit("no cached predictions -- run benchmark_aidust.py first")

    z0 = np.load(files[0])
    times = model_times(z0["test_idxs"])
    obs = match_observations(stations, times)
    print(f"model verification times: {len(times)} "
          f"({times[0].date()} .. {times[-1].date()})")
    print(f"stations with matched observations: {len(obs)}\n", flush=True)

    results = {}
    for name, (kind, thr) in EVENT_DEFS.items():
        print(f"=== observed event: {name} ===")
        print(f"{'model':<22}{'TS':>8}{'ETS':>8}{'POD':>7}{'FAR':>7}"
              f"{'obs_rate':>10}{'thr ug/m3':>11}{'events':>8}")
        rows = []
        for f in files:
            z = np.load(f)
            label = os.path.splitext(os.path.basename(f))[0]
            r = evaluate_model(z["pred"][..., 0], node_of_cell, stations, obs, kind, thr)
            if r is None:
                continue
            r["label"] = label
            rows.append(r)
        # Persistence on the same points, as the floor.
        rows.sort(key=lambda r: -(r["ETS"] if r["ETS"] == r["ETS"] else -9))
        for r in rows:
            print(f"{r['label'][:22]:<22}{r['TS']:>8.3f}{r['ETS']:>8.3f}{r['POD']:>7.3f}"
                  f"{r['FAR']:>7.3f}{r['obs_rate']*100:>9.2f}%{r['model_thr_ugm3']:>11.1f}"
                  f"{r['events']:>8}")
        results[name] = rows
        print()

    if args.rollout:
        kind, thr = EVENT_DEFS[args.rollout_event]
        rfiles = sorted(glob.glob(os.path.join(args.preds, "*__rollout.npz")))
        if rfiles:
            print(f"=== LEAD-DAY forecasts vs stations (event: {args.rollout_event}) ===")
            print("    this is the only column comparable to the published numbers:")
            print("    WMO SDS-WAS 0.31 @24h / 0.23 @48h  |  AI-DUST 0.33 @day1\n")
            print(f"{'model':<22}{'TS d1':>8}{'ETS d1':>8}{'POD d1':>8}{'FAR d1':>8}"
                  f"{'TS d2':>8}{'events':>8}")
            roll = {}
            for f in rfiles:
                label = os.path.basename(f).replace("__rollout.npz", "")
                r = score_rollout_vs_stations(f, node_of_cell, stations, kind, thr)
                if not r:
                    continue
                roll[label] = r
                d1 = r.get("day1", {}); d2 = r.get("day2", {})
                gv = lambda d, k: (f"{d[k]:.3f}" if isinstance(d.get(k), float)
                                   and d[k] == d[k] else "  --")
                print(f"{label[:22]:<22}{gv(d1,'TS'):>8}{gv(d1,'ETS'):>8}"
                      f"{gv(d1,'POD'):>8}{gv(d1,'FAR'):>8}{gv(d2,'TS'):>8}"
                      f"{d1.get('events',0):>8}")
            results["_rollout_vs_stations"] = roll
            print()

    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
