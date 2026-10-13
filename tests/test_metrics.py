"""Is SEDI actually base-rate independent, and is TS actually not?

The claim this project now rests its headline on is that TS falls with the base
rate whether or not the forecast got worse, and that SEDI does not.  That claim
is testable without any data: hold the forecast's quality fixed (hit rate H and
false-alarm rate F), vary only how often the event happens, and see which scores
move.

If this file fails, the threshold sweep in `train_station_dust.py` and the
comparability argument in `src/metrics/literature.py` are both unsafe to quote.

Run:  pytest tests/test_metrics.py
"""
import numpy as np
import pytest
import torch

from src.data.station_target import EVENT_THRESHOLDS
from src.metrics.aidust_metrics import contingency_scores, extremal_dependence
from src.models.ordinal import (
    SORDLoss, cumulative_probabilities, grades_from_labels, inverse_frequency_weights,
    sord_targets)


def table_at(base_rate, hit_rate, false_alarm_rate, n=4_000_000):
    """The 2x2 table of a forecast with fixed quality at a given event frequency.

    H and F are properties of the forecast; the base rate is a property of the
    world.  Holding the first two fixed while moving the third is exactly the
    comparison that TS gets wrong.
    """
    events = base_rate * n
    non_events = n - events
    nc = hit_rate * events                 # hits
    no = events - nc                       # misses
    nf = false_alarm_rate * non_events     # false alarms
    cn = non_events - nf                   # correct negatives
    return contingency_scores(round(nc), round(nf), round(no), round(cn))


# =========================================================================== #
# categorical scores under a changing base rate
# =========================================================================== #

# ---- no-skill forecast scores 0 ------------------------------------------- #
# A forecast that fires as often on non-events as on events (H == F) has
# learned nothing. Every equitable score must say 0.
def test_no_skill_forecast_scores_sedi_zero():
    s = table_at(0.05, 0.30, 0.30)
    assert abs(s["SEDI"]) < 1e-9, (
        f"SEDI {s['SEDI']:+.2e}, ETS {s['ETS']:+.4f}, but TS {s['TS']:.4f} > 0")


# ---- the headline claim: TS moves with the base rate, SEDI does not ------- #
# Same forecast quality (H = 0.42, F = 0.03) at our three visibility grades.
H_FIXED, F_FIXED = 0.42, 0.03
GRADES = [("dusty      vis < 5 km", 0.070),
          ("dust_storm vis < 1 km", 0.010),
          ("severe     vis < 0.4 km", 0.002)]


@pytest.fixture(scope="module")
def fixed_quality():
    """Scores of one fixed-quality forecast at each grade's base rate, rarest last."""
    return [table_at(base_rate, H_FIXED, F_FIXED) for _, base_rate in GRADES]


def test_ts_collapses_as_event_gets_rarer_at_constant_quality(fixed_quality):
    ts = [s["TS"] for s in fixed_quality]
    assert ts[0] > 2 * ts[-1], (
        f"TS {ts[0]:.3f} -> {ts[-1]:.3f} ({ts[0]/max(ts[-1], 1e-9):.1f}x) "
        f"with no change in H or F")


def test_sedi_unchanged_across_35x_change_in_base_rate(fixed_quality):
    sedi = [s["SEDI"] for s in fixed_quality]
    assert max(sedi) - min(sedi) < 5e-3, (
        f"SEDI {sedi[0]:.4f} -> {sedi[-1]:.4f}, spread {max(sedi)-min(sedi):.2e}")


# ETS corrects for luck, not for rarity. This is why 'ETS ~ TS' from the CAMS
# benchmark does not carry to the station target.
def test_ets_also_collapses_with_rarity(fixed_quality):
    ets = [s["ETS"] for s in fixed_quality]
    assert ets[0] > 2 * ets[-1], f"ETS {ets[0]:.3f} -> {ets[-1]:.3f}"


# ---- symmetry: swapping events and non-events leaves SEDI alone ------------ #
# Under the swap H' = 1 - F and F' = 1 - H. EDI is not symmetric; SEDI is.
def test_sedi_symmetric_under_exchanging_events_and_non_events():
    h, f = 0.42, 0.03
    base = extremal_dependence(h, f, symmetric=True)
    same = extremal_dependence(1 - f, 1 - h, symmetric=True)
    assert abs(same - base) < 1e-12, (
        f"SEDI(H,F) {base:.6f} != SEDI(1-F,1-H) {same:.6f}")


# EDI is reported alongside SEDI, never instead of it.
def test_edi_not_symmetric():
    h, f = 0.42, 0.03
    edi_base = extremal_dependence(h, f, symmetric=False)
    edi_swapped = extremal_dependence(1 - f, 1 - h, symmetric=False)
    assert abs(edi_swapped - edi_base) > 1e-3, (
        f"EDI(H,F) {edi_base:.4f} vs EDI(1-F,1-H) {edi_swapped:.4f}")


# ---- ordering: a better forecast must score higher ------------------------- #
def test_sedi_increases_with_hit_rate_at_fixed_false_alarm_rate():
    good = table_at(0.01, 0.60, 0.02)["SEDI"]
    poor = table_at(0.01, 0.30, 0.02)["SEDI"]
    assert good > poor, f"H 0.60 -> SEDI {good:.3f}; H 0.30 -> SEDI {poor:.3f}"


def test_sedi_decreases_with_false_alarm_rate_at_fixed_hit_rate():
    clean = table_at(0.01, 0.42, 0.005)["SEDI"]
    noisy = table_at(0.01, 0.42, 0.050)["SEDI"]
    assert clean > noisy, f"F 0.005 -> SEDI {clean:.3f}; F 0.050 -> SEDI {noisy:.3f}"


# ---- degenerate tables return NaN rather than a number --------------------- #
# A forecast that never fires has H = F = 0. TS calls that 0.000, which reads
# like a measurement; it is an empty table. NaN is the honest answer, and it
# is what our `severe` grade at 24 h actually produces.
def test_never_fires_forecast_gives_sedi_nan_not_zero():
    never = contingency_scores(nc=0, nf=0, no=35, cn=17_000)
    assert np.isnan(never["SEDI"]) and never["TS"] == 0.0, (
        f"TS {never['TS']:.3f} (looks measured), SEDI {never['SEDI']} (is not)")


# H = 1 and F = 0 send both logarithms to infinity.
def test_perfect_forecast_gives_sedi_nan_and_ts_one():
    perfect = contingency_scores(nc=100, nf=0, no=0, cn=9900)
    assert np.isnan(perfect["SEDI"]) and perfect["TS"] == 1.0, (
        f"SEDI {perfect['SEDI']}, TS {perfect['TS']:.3f}")


# ---- false-alarm RATE is not false-alarm RATIO ----------------------------- #
# The two are routinely confused and SEDI takes the rate. At a low base rate
# they differ by orders of magnitude, so swapping them silently ruins SEDI.
def test_false_alarm_rate_and_ratio_are_separate_fields():
    s = table_at(0.01, 0.42, 0.02)
    assert abs(s["F_rate"] - s["FAR"]) > 0.5, (
        f"F_rate {s['F_rate']:.4f} (of non-events) vs "
        f"FAR {s['FAR']:.4f} (of forecasts) -- SEDI takes the first")


# =========================================================================== #
# ordinal severity head
# The ordinal head must reproduce the binary events, and must nest them.
# =========================================================================== #
NAMES = list(EVENT_THRESHOLDS)                     # dust_storm, severe, dusty
COL = {n: i for i, n in enumerate(NAMES)}


def test_nested_threshold_labels_map_to_grades_in_severity_order():
    # Rows: clear, dusty only, dusty+storm, all three.
    lab = np.array([[False, False, False],             # vis >= 5 km        -> 0
                    [False, False, True],              # vis in [1, 5)      -> 1
                    [True,  False, True],              # vis in [0.4, 1)    -> 2
                    [True,  True,  True]])             # vis < 0.4 km       -> 3
    g = grades_from_labels(lab, EVENT_THRESHOLDS)
    assert np.array_equal(g, [0, 1, 2, 3]), f"grades {g.tolist()}"


@pytest.fixture(scope="module")
def confident():
    """Event probabilities for a confident grade-3 and a confident grade-1 output."""
    logits = torch.tensor([[0.0, 0.0, 0.0, 12.0], [0.0, 12.0, 0.0, 0.0]])
    return cumulative_probabilities(logits, EVENT_THRESHOLDS, NAMES).numpy()


# A confident grade-3 prediction must give P(dusty) = P(storm) = P(severe) = 1.
def test_confident_severe_prediction_implies_all_three_events(confident):
    p = confident[0]
    assert p.min() > 0.99, (
        f"dusty {p[COL['dusty']]:.3f}, dust_storm {p[COL['dust_storm']]:.3f}, "
        f"severe {p[COL['severe']]:.3f}")


# A confident grade-1 prediction must give P(dusty) = 1 and the other two ~0.
def test_confident_dusty_prediction_implies_dusty_but_not_storm_or_severe(confident):
    p = confident[1]
    assert p[COL["dusty"]] > 0.99 and p[COL["dust_storm"]] < 0.01, (
        f"dusty {p[COL['dusty']]:.3f}, dust_storm {p[COL['dust_storm']]:.3f}, "
        f"severe {p[COL['severe']]:.3f}")


# Monotonicity across 500 random outputs -- the guarantee three independent
# sigmoid heads cannot make, and the reason this head replaces them.
def test_event_probabilities_nested_for_every_random_output():
    rng = np.random.default_rng(0)
    rand = torch.tensor(rng.normal(size=(500, 4)), dtype=torch.float32)
    pr = cumulative_probabilities(rand, EVENT_THRESHOLDS, NAMES).numpy()
    violations = int((pr[:, COL["severe"]] > pr[:, COL["dust_storm"]] + 1e-6).sum()
                     + (pr[:, COL["dust_storm"]] > pr[:, COL["dusty"]] + 1e-6).sum())
    assert violations == 0, (
        f"P(severe) <= P(dust_storm) <= P(dusty) violated {violations} times "
        f"in {len(pr)} random logit vectors")


# SORD's soft target peaks on the truth and decays with ordinal distance.
def test_sord_target_peaks_at_true_grade_and_decays_with_distance():
    t = sord_targets(torch.tensor([3]), 4).numpy()[0]
    assert t.argmax() == 3 and t[3] > t[2] > t[1] > t[0], (
        f"delta(y=3) = {np.round(t, 4).tolist()}")


# Being three grades wrong must cost more than being one grade wrong. Plain
# cross-entropy scores these two identically; that is the point of SORD.
def test_forecasting_clear_for_severe_storm_costs_more_than_forecasting_storm():
    loss = SORDLoss(4)
    near = float(loss(torch.tensor([[0.0, 0.0, 8.0, 0.0]]), torch.tensor([3])))
    far = float(loss(torch.tensor([[8.0, 0.0, 0.0, 0.0]]), torch.tensor([3])))
    assert far > near, (
        f"predicted storm (1 grade out) {near:.3f} vs "
        f"predicted clear (3 grades out) {far:.3f}")


@pytest.fixture(scope="module")
def grade_weights():
    """Grade labels at a realistic imbalance, and their capped class weights."""
    y = np.concatenate([np.zeros(96_900, int), np.ones(2_400, int),
                        np.full(600, 2), np.full(100, 3)])
    return y, inverse_frequency_weights(y, 4, cap=200.0)


# Class weights: rare grades get more weight, capped so one hour cannot
# dominate the gradient.
def test_majority_grade_weighs_one_rarer_grades_more_all_capped(grade_weights):
    y, w = grade_weights
    assert abs(w[0] - 1.0) < 1e-6 and w[3] > w[2] > w[1] > w[0] and w.max() <= 200.0, (
        f"weights {np.round(w, 2).tolist()} for counts "
        f"{np.bincount(y).tolist()} (cap 200)")


# The bug this replaced: mean-normalising after the cap sent the majority
# weight to ~0.004, so the model got almost no gradient for saying "clear".
def test_majority_grade_never_driven_to_near_zero_weight(grade_weights):
    y, w = grade_weights
    counts = np.bincount(y)
    capped = np.minimum(counts.sum() / (4 * counts), 200.0)
    assert w[0] >= 1.0 - 1e-6, (
        f"grade 0 weight {w[0]:.4f}; mean-normalising after the cap gives "
        f"{(capped / capped.mean())[0]:.4f}")
