"""Evaluation in the AI-DUST paper's own terms -- physical units, not log-space R^2.

Everything this repo has reported so far (R^2 / MSE on log1p values) is
incomparable with the dust-forecasting literature.  AI-DUST scores with:

    TS            threat score at a dust threshold           (their Eq. 5, Figs 3/5/8/9)
    spatial R     Pearson across cells, per timestep          (Fig. 4a)
    temporal R    Pearson across time, per cell               (Figs 2, 6a-d)
    RMSE          ug/m^3                                      (Fig. 4b)
    RE            relative error %                            (Figs 2, 6e-h)
    lead curves   free-running rollout to 80 steps            (Fig. 2)
    stratified    emission-active vs inactive, reported apart (Figs 1, 2a/2b)

All inputs to :func:`evaluate` are **log1p(ug/m^3)**; conversion to physical units
happens inside, once, so no caller can forget it.

Shapes are ``[T, N]`` -- time by cell.  ``T`` is the evaluation sample axis and
``N`` the 251 valid land cells, which stand in for AI-DUST's monitoring stations.
"""
import numpy as np

# AI-DUST's SDS criterion: a station reports a dust event when PM10 > 200 ug/m^3.
DUST_THRESHOLD = 200.0

# RE is unstable as the denominator goes to zero; AI-DUST evaluates at stations
# during dust episodes.  We mirror that by masking out near-clean observations and
# reporting what fraction was excluded.
RE_MIN_OBS = 10.0


def to_physical(log_conc):
    """log1p(ug/m^3) -> ug/m^3, clamped non-negative."""
    return np.expm1(np.clip(np.asarray(log_conc, dtype="float64"), 0.0, None))


def _pearson(a, b, axis):
    """Pearson r along `axis`, returning NaN where either series is constant."""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    a_c = a - a.mean(axis=axis, keepdims=True)
    b_c = b - b.mean(axis=axis, keepdims=True)
    num = (a_c * b_c).sum(axis=axis)
    den = np.sqrt((a_c ** 2).sum(axis=axis) * (b_c ** 2).sum(axis=axis))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


# Multi-threshold verification, as WMO SDS-WAS does rather than a single cut.
# 200 ug/m^3 is AI-DUST's SDS criterion; the others bracket it.
TS_THRESHOLDS = (100.0, 200.0, 500.0, 1000.0)


def extremal_dependence(hit_rate, false_alarm_rate, symmetric=True):
    """EDI / SEDI -- the rare-event scores that do not depend on the base rate.

    Ferro & Stephenson (2011), Wea. Forecasting 26, 699-713.  Both are functions
    of ``H`` (hit rate = POD) and ``F`` (false-alarm RATE) alone, so unlike TS,
    ETS and HSS they do not shrink simply because the event is rare::

        EDI  = (ln F - ln H) / (ln F + ln H)
        SEDI = [ln F - ln H - ln(1-F) + ln(1-H)] / [ln F + ln H + ln(1-F) + ln(1-H)]

    Range -1..1, 0 = no skill.  SEDI is the symmetric form: it is unchanged when
    events and non-events swap roles, and is harder to hedge by over-forecasting.

    **Why this matters here.**  Our three visibility grades differ in base rate by
    a factor of ~35 (``dusty`` ~7%, ``dust_storm`` ~1%, ``severe`` ~0.2%).  TS
    falls across that range even for a forecast of constant quality, which is
    exactly what makes our ``severe`` TS of 0.000 uninformative and our
    ``dust_storm`` TS look far worse than the published numbers it was set beside.
    SEDI is the score that survives the comparison.

    Undefined at the boundaries: ``H`` or ``F`` equal to 0 or 1 sends a logarithm
    to infinity.  We return NaN rather than clipping, because a clipped value is a
    number the caller cannot distinguish from a measurement.  ``F`` reaching 0 is
    common at tiny base rates when the forecast never fires; that is not skill, it
    is an empty contingency table, and NaN says so.

    Note ``F`` is the false-alarm RATE ``NF / (NF + CN)``, not the false-alarm
    RATIO ``FAR = NF / (NC + NF)`` reported elsewhere in this module.  The two are
    routinely confused and are not interchangeable.
    """
    h = float(hit_rate)
    f = float(false_alarm_rate)
    if not (0.0 < h < 1.0) or not (0.0 < f < 1.0):
        return np.nan
    lh, lf = np.log(h), np.log(f)
    if not symmetric:
        den = lf + lh
        return (lf - lh) / den if den != 0 else np.nan
    l1h, l1f = np.log1p(-h), np.log1p(-f)
    den = lf + lh + l1f + l1h
    return (lf - lh - l1f + l1h) / den if den != 0 else np.nan


def contingency_scores(nc, nf, no, cn):
    """Every categorical score this repo reports, from one 2x2 table.

    ``nc`` hits, ``nf`` false alarms, ``no`` misses, ``cn`` correct negatives.
    Kept as the single implementation so the benchmark and the station trainer
    cannot drift apart.

        TS   = NC / (NC + NF + NO)                             (AI-DUST Eq. 5, CSI)
        ETS  = (NC - NC_rand) / (NC + NO + NF - NC_rand),  NC_rand = (NC+NF)(NC+NO)/N
               range -1/3..1, 0 = no skill beyond chance       (Gilbert)
        HSS  = 2(NC*CN - NF*NO) / [(NC+NO)(NO+CN) + (NC+NF)(NF+CN)]
               range -1..1, 0 = no skill beyond chance         (Heidke)
        SEDI see :func:`extremal_dependence` -- the only one here that is
             base-rate independent, and the only one to quote across thresholds

    TS is **not equitable** -- it credits hits that would occur by chance.  On the
    CAMS-target benchmark that correction was small (at an 8.5% base rate chance
    hits are ~9% of actual hits, so ETS sat within ~0.02 of TS for every model),
    which is why ETS was described here as near-redundant.  That conclusion does
    **not** carry to the station target: at a 0.2% base rate the base-rate
    dependence of TS itself dominates, and SEDI is the score to use.
    """
    nc, nf, no, cn = int(nc), int(nf), int(no), int(cn)
    n = nc + nf + no + cn
    denom = nc + nf + no

    nc_rand = (nc + nf) * (nc + no) / n if n > 0 else 0.0
    ets_den = denom - nc_rand
    hss_den = ((nc + no) * (no + cn) + (nc + nf) * (nf + cn))

    pod = (nc / (nc + no)) if (nc + no) > 0 else np.nan          # hit rate H
    f_rate = (nf / (nf + cn)) if (nf + cn) > 0 else np.nan       # false-alarm rate F

    return {
        "TS": (nc / denom) if denom > 0 else np.nan,
        "ETS": ((nc - nc_rand) / ets_den) if ets_den > 0 else np.nan,
        "HSS": (2.0 * (nc * cn - nf * no) / hss_den) if hss_den > 0 else np.nan,
        "SEDI": extremal_dependence(pod, f_rate, symmetric=True),
        "EDI": extremal_dependence(pod, f_rate, symmetric=False),
        "POD": pod,                                                # hit rate
        "FAR": (nf / (nc + nf)) if (nc + nf) > 0 else np.nan,      # false-alarm ratio
        "F_rate": f_rate,                                          # false-alarm rate
        "BIAS_freq": ((nc + nf) / (nc + no)) if (nc + no) > 0 else np.nan,
        "hits": nc, "false_alarms": nf, "misses": no, "correct_negatives": cn,
        "n": n,
        "events_observed": nc + no, "base_rate": (nc + no) / n if n else np.nan,
    }


def threat_score(pred, obs, threshold=DUST_THRESHOLD):
    """Categorical verification at one concentration threshold.

    Thin wrapper: thresholds both fields, then :func:`contingency_scores`.
    """
    pred_evt = np.asarray(pred) >= threshold
    obs_evt = np.asarray(obs) >= threshold
    nc = int(np.sum(pred_evt & obs_evt))
    nf = int(np.sum(pred_evt & ~obs_evt))
    no = int(np.sum(~pred_evt & obs_evt))
    cn = int(pred_evt.size - nc - nf - no)
    return contingency_scores(nc, nf, no, cn)


def fractional_error(pred, obs):
    """FGE and MFB -- the WMO SDS-WAS / air-quality pair, robust to the PM10 skew.

        FGE = (2/N) sum |(P - O) / (P + O)|     range 0..2, 0 = perfect
        MFB = (2/N) sum  (P - O) / (P + O)      range -2..2, 0 = unbiased

    Unlike RMSE these are not dominated by a handful of extreme cells, and unlike
    relative error they stay finite as the observation goes to zero, so no
    minimum-observation mask is needed.
    """
    p = np.asarray(pred, dtype="float64")
    o = np.asarray(obs, dtype="float64")
    denom = p + o
    valid = denom > 0
    if not np.any(valid):
        return {"FGE": np.nan, "MFB": np.nan}
    frac = (p[valid] - o[valid]) / denom[valid]
    return {"FGE": float(2.0 * np.mean(np.abs(frac))), "MFB": float(2.0 * np.mean(frac))}


def multi_threshold_scores(pred, obs, thresholds=TS_THRESHOLDS):
    """TS / ETS / SEDI / POD / FAR at several thresholds, as SDS-WAS reports them.

    A single 200 ug/m^3 cut hides where a model actually works: a model can look
    competent on moderate dust and be useless on severe events, which are the ones
    an early-warning system exists for.

    Read the SEDI row, not the TS row, when comparing ACROSS thresholds: the base
    rate falls by orders of magnitude from the lowest cut to the highest, and TS
    falls with it whether or not the forecast got worse.  Compare TS only within
    one threshold, against another model or another lead time.
    """
    out = {}
    for thr in thresholds:
        s = threat_score(pred, obs, thr)
        tag = int(thr)
        out[f"TS@{tag}"] = s["TS"]
        out[f"ETS@{tag}"] = s["ETS"]
        out[f"SEDI@{tag}"] = s["SEDI"]
        out[f"POD@{tag}"] = s["POD"]
        out[f"FAR@{tag}"] = s["FAR"]
        out[f"nobs@{tag}"] = s["events_observed"]
        out[f"base@{tag}"] = s["base_rate"]
    return out


def spatial_correlation(pred, obs):
    """Pearson across cells for each timestep -> ``[T]`` (AI-DUST Fig. 4a)."""
    return _pearson(pred, obs, axis=1)


def temporal_correlation(pred, obs):
    """Pearson across time for each cell -> ``[N]`` (AI-DUST Figs 2, 6a-d)."""
    return _pearson(pred, obs, axis=0)


def rmse(pred, obs):
    """Root mean squared error in ug/m^3."""
    d = np.asarray(pred, dtype="float64") - np.asarray(obs, dtype="float64")
    return float(np.sqrt(np.mean(d ** 2)))


def mae(pred, obs):
    return float(np.mean(np.abs(np.asarray(pred, dtype="float64")
                                - np.asarray(obs, dtype="float64"))))


def relative_error(pred, obs, min_obs=RE_MIN_OBS):
    """Relative error % with mean/median/Q1/Q3 -- the envelope in AI-DUST Fig. 2."""
    pred = np.asarray(pred, dtype="float64")
    obs = np.asarray(obs, dtype="float64")
    keep = obs >= min_obs
    if not np.any(keep):
        return {"RE_mean": np.nan, "RE_median": np.nan, "RE_q1": np.nan,
                "RE_q3": np.nan, "RE_coverage": 0.0}
    re = (pred[keep] - obs[keep]) / obs[keep] * 100.0
    return {
        "RE_mean": float(np.mean(re)),
        "RE_median": float(np.median(re)),
        "RE_q1": float(np.percentile(re, 25)),
        "RE_q3": float(np.percentile(re, 75)),
        "RE_coverage": float(keep.mean()),   # fraction of points RE is defined on
    }


def add_skill_scores(rows, reference_label="persistence"):
    """Skill relative to persistence -- the control ETS does *not* provide.

    ETS removes hits that would occur by chance.  At this domain's 8.5% event base
    rate chance hits are only ~9% of actual hits, so ETS lands within ~0.02 of TS
    and corrects almost nothing.  The thing inflating TS here is not luck, it is
    that **dust fields are persistent**: holding the last frame scores TS 0.586.

    The right control is therefore a skill score against persistence,

        SS = (score_model - score_persistence) / (perfect - score_persistence)

    which is 0 when a model merely reproduces persistence and 1 when perfect.
    Applied to TS (perfect = 1) and to RMSE (perfect = 0, so the sign flips).
    """
    ref = next((r for r in rows if r.get("label") == reference_label), None)
    if ref is None:
        return rows
    for r in rows:
        ts_p, rmse_p = ref.get("TS"), ref.get("RMSE_ugm3")
        if isinstance(ts_p, float) and ts_p == ts_p and ts_p < 1:
            r["SS_TS"] = (r["TS"] - ts_p) / (1.0 - ts_p)
        if isinstance(rmse_p, float) and rmse_p and rmse_p > 0:
            r["SS_RMSE"] = (rmse_p - r["RMSE_ugm3"]) / rmse_p
    return rows


def r2_log(pred_log, obs_log):
    """Log-space R^2 -- kept only for continuity with this repo's earlier numbers."""
    p = np.asarray(pred_log, dtype="float64")
    o = np.asarray(obs_log, dtype="float64")
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def evaluate(pred_log, obs_log, threshold=DUST_THRESHOLD, emission_active=None,
             label=""):
    """Full AI-DUST metric set for one model's predictions.

    Parameters
    ----------
    pred_log, obs_log : ``[T, N]`` predictions and truth in log1p(ug/m^3).
    emission_active : optional ``[T, N]`` bool.  When given, the whole metric set
        is additionally reported separately over emission-active and inactive
        points -- AI-DUST's Fig. 1/Fig. 2a-b split, where their skill differs most.
    """
    pred = to_physical(pred_log)
    obs = to_physical(obs_log)

    out = {"label": label, "n_points": int(pred.size)}
    out.update(threat_score(pred, obs, threshold))
    out.update(relative_error(pred, obs))
    out.update(fractional_error(pred, obs))
    out.update(multi_threshold_scores(pred, obs))

    r_s = spatial_correlation(pred, obs)
    r_t = temporal_correlation(pred, obs)
    out.update({
        "R_spatial_mean": float(np.nanmean(r_s)),
        "R_spatial_median": float(np.nanmedian(r_s)),
        "R_temporal_mean": float(np.nanmean(r_t)),
        "R_temporal_median": float(np.nanmedian(r_t)),
        "RMSE_ugm3": rmse(pred, obs),
        "MAE_ugm3": mae(pred, obs),
        "R2_log": r2_log(pred_log, obs_log),
    })

    if emission_active is not None:
        act = np.asarray(emission_active, dtype=bool)
        for name, sel in (("emis", act), ("noemis", ~act)):
            if not np.any(sel):
                continue
            sub = threat_score(pred[sel], obs[sel], threshold)
            sub.update(relative_error(pred[sel], obs[sel]))
            out[f"{name}_TS"] = sub["TS"]
            out[f"{name}_RMSE_ugm3"] = rmse(pred[sel], obs[sel])
            out[f"{name}_RE_mean"] = sub["RE_mean"]
            out[f"{name}_n"] = int(sel.sum())
    return out


def evaluate_rollout(pred_log_by_step, obs_log_by_step, threshold=DUST_THRESHOLD,
                     steps_per_day=8, label=""):
    """Per-lead-step metrics for a free-running rollout (AI-DUST Figs 2, 3, 4).

    Parameters
    ----------
    pred_log_by_step, obs_log_by_step : ``[K, T, N]`` -- lead step, sample, cell.
        ``K`` up to 80 (= 10 days at 3 h), matching AI-DUST's Fig. 2.
    steps_per_day : used to aggregate TS by lead *day*, as in their Fig. 3.

    Returns
    -------
    per_step : list of dicts, one per lead step
    per_day  : list of dicts, TS / RMSE / R aggregated by lead day
    """
    pred_all = np.asarray(pred_log_by_step)
    obs_all = np.asarray(obs_log_by_step)
    n_steps = pred_all.shape[0]

    per_step = []
    for k in range(n_steps):
        m = evaluate(pred_all[k], obs_all[k], threshold, label=f"{label}|step{k+1}")
        m["lead_step"] = k + 1
        m["lead_hours"] = (k + 1) * 24 / steps_per_day
        per_step.append(m)

    per_day = []
    for d0 in range(0, n_steps, steps_per_day):
        block = slice(d0, min(d0 + steps_per_day, n_steps))
        p = pred_all[block].reshape(-1, pred_all.shape[-1])
        o = obs_all[block].reshape(-1, obs_all.shape[-1])
        m = evaluate(p, o, threshold, label=f"{label}|day{d0//steps_per_day+1}")
        m["lead_day"] = d0 // steps_per_day + 1
        per_day.append(m)

    return per_step, per_day


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
TABLE_COLUMNS = [
    ("label", "model", "{:<18}"),
    ("TS", "TS@200", "{:>7.3f}"),
    # ETS/HSS are the equitable versions -- the honest headline, since plain TS
    # credits chance hits and persistence already scores TS 0.586 here.
    ("ETS", "ETS", "{:>7.3f}"),
    ("HSS", "HSS", "{:>7.3f}"),
    # The headline honesty column: 0 = no better than holding the last frame.
    ("SS_TS", "SSvsP", "{:>7.3f}"),
    ("POD", "POD", "{:>6.3f}"),
    ("FAR", "FAR", "{:>6.3f}"),
    ("FGE", "FGE", "{:>6.3f}"),
    ("R_spatial_mean", "R_spat", "{:>7.3f}"),
    ("R_temporal_mean", "R_temp", "{:>7.3f}"),
    ("RMSE_ugm3", "RMSE", "{:>8.1f}"),
    ("RE_mean", "RE%", "{:>8.1f}"),
    ("R2_log", "R2_log", "{:>7.3f}"),
    ("emis_TS", "TS_emis", "{:>8.3f}"),
    ("noemis_TS", "TS_noem", "{:>8.3f}"),
    # Lead-time TS is the only column comparable to AI-DUST's published numbers
    # (their Fig. 3: 0.33 at day 1, 0.22-0.23 at day 10).  The single-step TS
    # above is a 3 h forecast and is NOT comparable -- persistence alone scores
    # ~0.59 there, because only 4.8% of cells change event state in one step.
    ("TS_day1", "TSday1", "{:>7.3f}"),
    ("TS_day10", "TSday10", "{:>8.3f}"),
]


def format_table(rows):
    """Render a list of :func:`evaluate` dicts as the benchmark's results table."""
    head = " ".join(hdr.rjust(len(fmt.format(0).strip()) if "f" in fmt else 18)
                    if key != "label" else hdr.ljust(18)
                    for key, hdr, fmt in TABLE_COLUMNS)
    lines = [head, "-" * len(head)]
    for row in rows:
        cells = []
        for key, _hdr, fmt in TABLE_COLUMNS:
            val = row.get(key, np.nan)
            if key == "label":
                cells.append(fmt.format(str(val)[:18]))
            elif val is None or (isinstance(val, float) and not np.isfinite(val)):
                cells.append(fmt.replace("f", "s").format("--"))
            else:
                cells.append(fmt.format(val))
        lines.append(" ".join(cells))
    return "\n".join(lines)
