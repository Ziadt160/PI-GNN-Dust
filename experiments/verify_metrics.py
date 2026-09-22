"""Verification audit: is SEDI actually base-rate independent, and is TS actually not?

The claim this project now rests its headline on is that TS falls with the base
rate whether or not the forecast got worse, and that SEDI does not.  That claim
is testable without any data: hold the forecast's quality fixed (hit rate H and
false-alarm rate F), vary only how often the event happens, and see which scores
move.

If this file fails, the threshold sweep in `train_station_dust.py` and the
comparability argument in `src/metrics/literature.py` are both unsafe to quote.

Run:  python -u experiments/verify_metrics.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.metrics.aidust_metrics import (                      # noqa: E402
    contingency_scores, extremal_dependence)

PASS, FAIL = "PASS", "** FAIL **"
results = []


def report(name, ok, detail):
    results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}\n        {detail}")


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


def main():
    print("=== categorical scores under a changing base rate ===\n")

    # ---- 1. no-skill forecast scores 0 -------------------------------------- #
    # A forecast that fires as often on non-events as on events (H == F) has
    # learned nothing. Every equitable score must say 0.
    s = table_at(0.05, 0.30, 0.30)
    report("no-skill forecast (H == F) scores SEDI 0",
           abs(s["SEDI"]) < 1e-9,
           f"SEDI {s['SEDI']:+.2e}, ETS {s['ETS']:+.4f}, but TS {s['TS']:.4f} > 0")

    # ---- 2. the headline claim: TS moves with the base rate, SEDI does not --- #
    # Same forecast quality (H = 0.42, F = 0.03) at our three visibility grades.
    H_FIXED, F_FIXED = 0.42, 0.03
    grades = [("dusty      vis < 5 km", 0.070),
              ("dust_storm vis < 1 km", 0.010),
              ("severe     vis < 0.4 km", 0.002)]
    rows = [(nm, table_at(br, H_FIXED, F_FIXED)) for nm, br in grades]

    print(f"\n        {'grade':<24}{'base':>7}{'TS':>8}{'ETS':>8}{'SEDI':>8}")
    for nm, s in rows:
        print(f"        {nm:<24}{s['base_rate']:>7.3%}{s['TS']:>8.3f}"
              f"{s['ETS']:>8.3f}{s['SEDI']:>8.3f}")
    print()

    ts = [s["TS"] for _, s in rows]
    sedi = [s["SEDI"] for _, s in rows]
    report("TS collapses as the event gets rarer, at constant forecast quality",
           ts[0] > 2 * ts[-1],
           f"TS {ts[0]:.3f} -> {ts[-1]:.3f} ({ts[0]/max(ts[-1],1e-9):.1f}x) "
           f"with no change in H or F")
    report("SEDI is unchanged across the same 35x change in base rate",
           max(sedi) - min(sedi) < 5e-3,
           f"SEDI {sedi[0]:.4f} -> {sedi[-1]:.4f}, spread {max(sedi)-min(sedi):.2e}")
    report("ETS also collapses -- it corrects for luck, not for rarity",
           rows[0][1]["ETS"] > 2 * rows[-1][1]["ETS"],
           f"ETS {rows[0][1]['ETS']:.3f} -> {rows[-1][1]['ETS']:.3f}. This is why "
           f"'ETS ~ TS' from the CAMS benchmark does not carry to the station target")

    # ---- 3. symmetry: swapping events and non-events leaves SEDI alone ------- #
    # Under the swap H' = 1 - F and F' = 1 - H. EDI is not symmetric; SEDI is.
    h, f = 0.42, 0.03
    same = extremal_dependence(1 - f, 1 - h, symmetric=True)
    base = extremal_dependence(h, f, symmetric=True)
    report("SEDI is symmetric under exchanging events and non-events",
           abs(same - base) < 1e-12,
           f"SEDI(H,F) {base:.6f} == SEDI(1-F,1-H) {same:.6f}")
    edi_swapped = extremal_dependence(1 - f, 1 - h, symmetric=False)
    edi_base = extremal_dependence(h, f, symmetric=False)
    report("EDI is NOT symmetric -- reported alongside, never instead of, SEDI",
           abs(edi_swapped - edi_base) > 1e-3,
           f"EDI(H,F) {edi_base:.4f} vs EDI(1-F,1-H) {edi_swapped:.4f}")

    # ---- 4. ordering: a better forecast must score higher -------------------- #
    good = table_at(0.01, 0.60, 0.02)["SEDI"]
    poor = table_at(0.01, 0.30, 0.02)["SEDI"]
    report("SEDI increases with hit rate at fixed false-alarm rate",
           good > poor, f"H 0.60 -> SEDI {good:.3f}; H 0.30 -> SEDI {poor:.3f}")
    clean = table_at(0.01, 0.42, 0.005)["SEDI"]
    noisy = table_at(0.01, 0.42, 0.050)["SEDI"]
    report("SEDI decreases with false-alarm rate at fixed hit rate",
           clean > noisy, f"F 0.005 -> SEDI {clean:.3f}; F 0.050 -> SEDI {noisy:.3f}")

    # ---- 5. degenerate tables return NaN rather than a number --------------- #
    # A forecast that never fires has H = F = 0. TS calls that 0.000, which reads
    # like a measurement; it is an empty table. NaN is the honest answer, and it
    # is what our `severe` grade at 24 h actually produces.
    never = contingency_scores(nc=0, nf=0, no=35, cn=17_000)
    report("a never-fires forecast gives SEDI NaN, not a spurious 0.000",
           np.isnan(never["SEDI"]) and never["TS"] == 0.0,
           f"TS {never['TS']:.3f} (looks measured), SEDI {never['SEDI']} (is not)")
    perfect = contingency_scores(nc=100, nf=0, no=0, cn=9900)
    report("a perfect forecast gives SEDI NaN at the boundary, TS 1.000",
           np.isnan(perfect["SEDI"]) and perfect["TS"] == 1.0,
           f"H=1 and F=0 send both logarithms to infinity; TS {perfect['TS']:.3f}")

    # ---- 6. false-alarm RATE is not false-alarm RATIO ----------------------- #
    # The two are routinely confused and SEDI takes the rate. At a low base rate
    # they differ by orders of magnitude, so swapping them silently ruins SEDI.
    s = table_at(0.01, 0.42, 0.02)
    report("F_rate and FAR are reported as separate fields and differ sharply",
           abs(s["F_rate"] - s["FAR"]) > 0.5,
           f"F_rate {s['F_rate']:.4f} (of non-events) vs "
           f"FAR {s['FAR']:.4f} (of forecasts) -- SEDI takes the first")

    check_ordinal()

    print(f"\n{sum(results)}/{len(results)} checks passed")
    if not all(results):
        sys.exit("METRIC AUDIT FAILED -- the threshold comparison is not safe to quote")


def check_ordinal():
    """The ordinal head must reproduce the binary events, and must nest them."""
    import torch
    from src.data.station_target import EVENT_THRESHOLDS
    from src.models.ordinal import (
        SORDLoss, cumulative_probabilities, grades_from_labels, sord_targets,
        inverse_frequency_weights)

    print("\n=== ordinal severity head ===\n")
    names = list(EVENT_THRESHOLDS)                     # dust_storm, severe, dusty

    # Grades from nested labels. Rows: clear, dusty only, dusty+storm, all three.
    lab = np.array([[False, False, False],             # vis >= 5 km        -> 0
                    [False, False, True],              # vis in [1, 5)      -> 1
                    [True,  False, True],              # vis in [0.4, 1)    -> 2
                    [True,  True,  True]])             # vis < 0.4 km       -> 3
    g = grades_from_labels(lab, EVENT_THRESHOLDS)
    report("nested threshold labels map to grades 0..3 in severity order",
           np.array_equal(g, [0, 1, 2, 3]), f"grades {g.tolist()}")

    # A confident grade-3 prediction must give P(dusty) = P(storm) = P(severe) = 1;
    # a confident grade-1 prediction must give P(dusty)=1 and the other two ~0.
    logits = torch.tensor([[0.0, 0.0, 0.0, 12.0], [0.0, 12.0, 0.0, 0.0]])
    p = cumulative_probabilities(logits, EVENT_THRESHOLDS, names).numpy()
    col = {n: i for i, n in enumerate(names)}
    report("a confident `severe` prediction implies all three events",
           p[0].min() > 0.99, f"dusty {p[0][col['dusty']]:.3f}, "
           f"dust_storm {p[0][col['dust_storm']]:.3f}, severe {p[0][col['severe']]:.3f}")
    report("a confident `dusty` prediction implies dusty but not storm or severe",
           p[1][col["dusty"]] > 0.99 and p[1][col["dust_storm"]] < 0.01,
           f"dusty {p[1][col['dusty']]:.3f}, dust_storm {p[1][col['dust_storm']]:.3f}, "
           f"severe {p[1][col['severe']]:.3f}")

    # Monotonicity across 500 random outputs -- the guarantee three independent
    # sigmoid heads cannot make, and the reason this head replaces them.
    rng = np.random.default_rng(0)
    rand = torch.tensor(rng.normal(size=(500, 4)), dtype=torch.float32)
    pr = cumulative_probabilities(rand, EVENT_THRESHOLDS, names).numpy()
    nested = (pr[:, col["severe"]] <= pr[:, col["dust_storm"]] + 1e-6).all() and \
             (pr[:, col["dust_storm"]] <= pr[:, col["dusty"]] + 1e-6).all()
    report("P(severe) <= P(dust_storm) <= P(dusty) for every random output",
           nested, f"checked {len(pr)} random logit vectors, 0 violations")

    # SORD's soft target peaks on the truth and decays with ordinal distance.
    t = sord_targets(torch.tensor([3]), 4).numpy()[0]
    report("SORD target peaks at the true grade and decays with distance",
           t.argmax() == 3 and t[3] > t[2] > t[1] > t[0],
           f"delta(y=3) = {np.round(t, 4).tolist()}")

    # Being three grades wrong must cost more than being one grade wrong. Plain
    # cross-entropy scores these two identically; that is the point of SORD.
    loss = SORDLoss(4)
    near = float(loss(torch.tensor([[0.0, 0.0, 8.0, 0.0]]), torch.tensor([3])))
    far = float(loss(torch.tensor([[8.0, 0.0, 0.0, 0.0]]), torch.tensor([3])))
    report("forecasting `clear` for a severe storm costs more than forecasting `storm`",
           far > near, f"predicted storm (1 grade out) {near:.3f} vs "
           f"predicted clear (3 grades out) {far:.3f}")

    # Class weights: rare grades get more weight, capped so one hour cannot
    # dominate the gradient.
    y = np.concatenate([np.zeros(96_900, int), np.ones(2_400, int),
                        np.full(600, 2), np.full(100, 3)])
    w = inverse_frequency_weights(y, 4, cap=200.0)
    report("the majority grade weighs 1, rarer grades weigh more, all capped",
           abs(w[0] - 1.0) < 1e-6 and w[3] > w[2] > w[1] > w[0] and w.max() <= 200.0,
           f"weights {np.round(w, 2).tolist()} for counts "
           f"{np.bincount(y).tolist()} (cap 200)")
    # The bug this replaced: mean-normalising after the cap sent the majority
    # weight to ~0.004, so the model got almost no gradient for saying "clear".
    report("the majority grade is never driven to a near-zero weight",
           w[0] >= 1.0 - 1e-6,
           f"grade 0 weight {w[0]:.4f}; mean-normalising after the cap gives "
           f"{(np.minimum(np.bincount(y).sum() / (4 * np.bincount(y)), 200.0) / np.minimum(np.bincount(y).sum() / (4 * np.bincount(y)), 200.0).mean())[0]:.4f}")


if __name__ == "__main__":
    main()
