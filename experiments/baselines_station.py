"""Controls for the station dust model: is the skill real, or trivially available?

A rare-event classifier can post a respectable-looking score for uninteresting
reasons, and the benchmark so far only ruled out one of them (persistence).  This
script adds the rest, so any claimed skill has to survive all of them.

  never-dust        the degenerate solution. Frequency matching should already
                    forbid it, but it is worth showing POD = 0, TS = 0 explicitly
                    rather than asserting it.
  climatology       P(event | station, calendar month), fitted on train only.
                    **The important one.** If the model barely beats this, it has
                    learned that spring is dusty -- a seasonal cycle, not a
                    forecast of individual events.
  persistence       conditions now continue.
  wind threshold    event iff wind speed at the target time exceeds a cut.
                    Future meteorology is 40% of the model's feature importance
                    and the emission term (another 18%) is itself a function of
                    wind, so ~58% of its skill traces to wind. This asks what a
                    bare wind threshold already achieves.
  logistic regression   same features, no graph, no physics, no message passing.
  gradient boosting     same features, strong tabular learner.

The last two matter because on the earlier CAMS-target task gradient boosting
beat the graph network outright (R2 0.642 vs 0.580). If that repeats here, the
graph and physics machinery is not earning its place and should be said so.

Every baseline is scored exactly like the model: daily dust-days, interior cells,
frequency-matched thresholds, station-verified, test 2021-2024.

Run:  python -u experiments/baselines_station.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.data.saudi_dataset import DataConfig, SaudiFields        # noqa: E402
from src.data.station_target import DustTarget                    # noqa: E402
from src.data.topologies import FeatureAssembler                  # noqa: E402

HORIZON = 8          # 24 h at the 3-hourly cadence
EVENT = "dust_storm"


def contingency(pred, obs):
    nc = int((pred & obs).sum()); nf = int((pred & ~obs).sum()); no = int((~pred & obs).sum())
    den, n = nc + nf + no, obs.size
    rand = (nc + nf) * (nc + no) / n if n else 0.0
    return {"TS": nc / den if den else 0.0,
            "ETS": (nc - rand) / (den - rand) if (den - rand) > 0 else 0.0,
            "POD": nc / (nc + no) if (nc + no) else 0.0,
            "FAR": nf / (nc + nf) if (nc + nf) else float("nan")}


def freq_match(score, obs):
    """Forecast an event as often as one is observed, then score.

    A constant score has no ordering, so quantile matching degenerates -- with an
    all-zero score, ``score >= 0`` fires everywhere and the "never forecast dust"
    control silently becomes "always forecast dust" (POD 1.0, TS = base rate).
    Constant scores are therefore handled explicitly.
    """
    if obs.sum() == 0:
        return contingency(np.zeros_like(obs, bool), obs)
    if np.ptp(score) == 0:
        return contingency(np.zeros_like(obs, bool), obs)
    thr = np.quantile(score, 1.0 - obs.mean())
    return contingency(score >= thr, obs)


def daily(arr, keys, how):
    """Aggregate to one value per key.

    The key must be (station, date), not date alone: collapsing all stations into
    a single daily series changes the event definition from "a dust day at this
    station" to "a dust day anywhere in the country", which raises the base rate
    from ~1% to ~7.5% and is not the quantity the model was scored on.
    """
    u, inv = np.unique(keys, return_inverse=True)
    out = np.zeros(len(u), dtype=arr.dtype)
    for d in range(len(u)):
        out[d] = how(arr[inv == d])
    return out, u


def main():
    cfg = DataConfig(cadence="3h", window=3, future_met=True, courant=True,
                     relief=True, emission_features=True, transport_feature=True,
                     divergence=True, multitask=True, topology="full", neighbors=8)
    fields = SaudiFields(cfg, verbose=False)
    asm = FeatureAssembler(fields, cfg)

    tgt = DustTarget()
    labels, valid, extra = tgt.align(pd.to_datetime(fields.times, utc=True))
    node_idx, kept = tgt.node_indices(fields.mask)
    names = extra["threshold_names"]
    cols = [extra["station_ids"].index(s) for s in kept]
    labels, valid = labels[:, cols], valid[:, cols]

    # interior cells only, matching the model
    h, w = fields.mask.shape
    pad = np.pad(fields.mask, 1, constant_values=False)
    nbr = sum(pad[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc].astype(int)
              for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
    interior = fields.mask & (nbr == 8)
    by = {s["id"]: s for s in tgt.stations}
    sel = [k for k, sid in enumerate(kept) if interior[by[sid]["row"], by[sid]["col"]]]
    labels, valid, node_idx = labels[:, sel], valid[:, sel], node_idx[sel]
    kept = [kept[k] for k in sel]
    k_ev = names.index(EVENT)

    def split(y0, y1):
        s = np.flatnonzero((fields.years >= y0) & (fields.years <= y1))
        return np.arange(max(int(s[0]), cfg.window - 1),
                         min(int(s[-1]), fields.n_steps - HORIZON - 1))

    tr, te = split(2003, 2018), split(2021, 2024)
    times = pd.to_datetime(fields.times, utc=True)
    print(f"interior stations {len(kept)} | train {len(tr):,} | test {len(te):,} steps\n")

    # ---- assemble a tabular feature matrix at station cells -------------------- #
    def tabular(idxs):
        X = np.stack([asm.assemble(int(i), HORIZON)[node_idx] for i in idxs])
        return X.reshape(len(idxs) * len(node_idx), -1)

    def target(idxs):
        return (labels[idxs + HORIZON][:, :, k_ev].reshape(-1),
                valid[idxs + HORIZON].reshape(-1))

    sub_tr = tr[::4]                      # keep the fit tractable
    Xtr, (ytr, mtr) = tabular(sub_tr), target(sub_tr)
    Xte, (yte, mte) = tabular(te), target(te)
    Xtr, ytr = Xtr[mtr], ytr[mtr].astype(bool)
    Xte_f, yte_f = Xte[mte], yte[mte].astype(bool)
    print(f"tabular: train {Xtr.shape} ({ytr.sum()} events) | "
          f"test {Xte_f.shape} ({yte_f.sum()} events)", flush=True)

    ch = asm.channels
    i_speed = ch.index("speed_+1")
    results = {}

    # ---- scores, all on the 3-hourly test points first ------------------------- #
    scores = {}
    scores["never-dust"] = np.zeros(len(yte_f))
    scores["wind threshold (speed at +24 h)"] = Xte_f[:, i_speed]

    # climatology: P(event | station, month), fitted on TRAIN only
    st_idx = np.tile(np.arange(len(kept)), len(te))[mte]
    month = np.repeat(times[te + HORIZON].month.values, len(kept))[mte]
    tr_st = np.tile(np.arange(len(kept)), len(sub_tr))[mtr]
    tr_mo = np.repeat(times[sub_tr + HORIZON].month.values, len(kept))[mtr]
    clim = np.zeros((len(kept), 13))
    for s in range(len(kept)):
        for mo in range(1, 13):
            sel_m = (tr_st == s) & (tr_mo == mo)
            clim[s, mo] = ytr[sel_m].mean() if sel_m.any() else ytr.mean()
    scores["climatology (station x month)"] = clim[st_idx, month]

    # persistence: the observed state at initialisation
    now = labels[te][:, :, k_ev].reshape(-1)[mte].astype(float)
    scores["persistence"] = now

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(sc.transform(Xtr), ytr)
    scores["logistic regression"] = lr.predict_proba(sc.transform(Xte_f))[:, 1]

    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                        class_weight="balanced", random_state=0)
    gb.fit(Xtr, ytr)
    scores["gradient boosting"] = gb.predict_proba(Xte_f)[:, 1]

    # ---- aggregate every baseline to DAILY dust-days and score ----------------- #
    # key on (station, calendar day) so the event stays "a dust day at THIS station"
    dstr = np.repeat(np.array([d.isoformat() for d in times[te + HORIZON].date]),
                     len(kept))[mte]
    stn = np.tile(np.array(kept), len(te))[mte]
    keys = np.char.add(np.char.add(stn.astype(str), "|"), dstr)
    y_daily, _ = daily(yte_f, keys, np.any)
    print(f"\ndaily test points {len(y_daily):,} | dust days {int(y_daily.sum())} "
          f"| base rate {y_daily.mean()*100:.2f}%\n")

    print(f"{'baseline':<36}{'TS':>8}{'ETS':>8}{'POD':>8}{'FAR':>8}")
    for name, s in scores.items():
        s_daily, _ = daily(s.astype(float), keys, np.max)
        r = freq_match(s_daily, y_daily)
        results[name] = r
        print(f"  {name:<34}{r['TS']:>8.3f}{r['ETS']:>8.3f}{r['POD']:>8.3f}{r['FAR']:>8.3f}")

    print(f"\n  {'GNN (station dust model)':<34}{0.142:>8.3f}{0.137:>8.3f}"
          f"{0.249:>8.3f}{0.751:>8.3f}   <- for comparison")
    print(f"  {'WMO SDS-WAS operational':<34}{0.31:>8.3f}")

    with open("results/baselines_station.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote results/baselines_station.json")


if __name__ == "__main__":
    main()
