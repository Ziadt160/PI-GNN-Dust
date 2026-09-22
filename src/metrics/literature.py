"""Published dust-forecast scores, with the caveats that make them comparable or not.

What changed here
-----------------
This file used to warn that our numbers were not comparable to the literature
because *we scored against CAMS reanalysis and everyone else scored against
stations*.  Since the pivot to observed station events that reason is gone: we
score against station observations too.

The comparability problem that remains is **base rate**, and it is bigger:

    Sarafian 2023  dust event = PM10 > 65.2 ug/m^3   base rate  13%
    Sarafian 2025  dust event = PM10 > 50   ug/m^3   base rate  22%
    Xu 2026        dust event = visibility <= 10 km  base rate  <8%
    ours `dusty`   visibility < 5 km                 base rate  ~7%   (1229 events)
    ours `storm`   visibility < 1 km                 base rate  ~1%   ( 185 events)
    ours `severe`  visibility < 0.4 km               base rate  ~0.2% (  35 events)

Every published "dust event" threshold sits at or below our `dusty` grade.  TS,
POD and FAR are all base-rate dependent, so setting our `dust_storm` TS beside a
published TS understates us by roughly a factor of two, and setting our `severe`
TS beside one is meaningless.

**Rule: compare at `dusty`, or compare with SEDI.**  Our own measured numbers at
24 h, station-verified, daily, interior (``results/station_daily_interior.json``)::

    severe      (vis < 0.4 km)   TS 0.000   POD 0.000   FAR 1.000
    dust_storm  (vis < 1 km)     TS 0.135   POD 0.238   FAR 0.762
    dusty       (vis < 5 km)     TS 0.260   POD 0.413   FAR 0.587   <- the comparable row

SEDI (Ferro & Stephenson 2011, Wea. Forecasting 26, 699-713) is a function of the
hit rate and false-alarm rate only, so it is base-rate independent by
construction and is the right metric for the `severe` grade where TS degenerates.
It is not yet implemented in ``aidust_metrics``; until it is, report the
threshold sweep rather than a single number.

Provenance
----------
Each entry carries ``verified``:

    "primary"   numbers read out of the paper's own text
    "abstract"  numbers read from the paper's abstract only
    "snippet"   numbers seen only in a search result or secondary summary -- a
                lead to chase, not a citable figure
"""


# Recall/precision and POD/FAR describe the same contingency table; several of
# the references below publish one pair while we publish the other.
def ts_from_recall_precision(recall, precision):
    """Threat score implied by a published recall/precision pair.

    TS = hits / (hits + false alarms + misses).  Per unit of observed events,
    hits = recall and misses = 1 - recall; from precision = hits/(hits + false
    alarms) it follows that false alarms = hits * (1 - precision) / precision.
    """
    if not (0.0 < precision <= 1.0) or not (0.0 < recall <= 1.0):
        return float("nan")
    hits = recall
    false_alarms = hits * (1.0 - precision) / precision
    misses = 1.0 - recall
    return hits / (hits + false_alarms + misses)


def far_from_precision(precision):
    """FAR is 1 - precision.  Spelled out because the two are routinely confused."""
    return 1.0 - precision


# Each entry: metric values as published, plus what they were measured against.
# ``base_rate`` is the fraction of samples that are events under that paper's own
# event definition -- the field that decides whether a TS comparison is honest.
REFERENCES = [
    {
        "name": "Sarafian et al. (multi-task)",
        "venue": "npj Clim Atmos Sci 6:23, 2023",
        "region": "Israel (30 stations); input domain 19-43N, 2-51E",
        "task": "local PM10 dust event, 12-72 h lead, ordinal (13 levels, SORD loss)",
        "truth": "in-situ PM10 (TEOM), 2003-2020, 3-hourly",
        "resolution": "station point; 0.5 deg input fields (40 x 49 x 99 tensor)",
        "event_def": "PM10 > 65.2 ug/m^3 (2 sd above summer background)",
        "base_rate": 0.13,
        "metrics": {"recall_24h": 0.76, "precision_24h": 0.67,
                    "recall_24h_DJFMA": 0.83,
                    "TS_24h_implied": 0.553,
                    "xgboost_recall_24h": 0.67, "xgboost_precision_24h": 0.47,
                    "persistence_recall_24h": 0.51, "persistence_precision_24h": 0.51},
        "comparable": True,
        "comparable_to": "our `dusty` grade, not `dust_storm`",
        "verified": "primary",
        "note": "THE closest comparator in the literature and the one to beat. "
                "7,019 events in 52,245 3-hourly samples. Four transferable design "
                "choices, all aimed at problems we measured: (1) an ordinal target "
                "instead of a binary one -- our `severe` head scores TS 0.000 "
                "precisely because a binary head starves at a 0.2% base rate; "
                "(2) a regional auxiliary task (reconstruct the CAMS PM10 field) "
                "trained jointly with the local task at lambda = 0.05 -- dense "
                "supervision for a rare-event problem, and a use for CAMS that our "
                "feature importance says it cannot serve as an input (5%); "
                "(3) ERA5 Z/U/V/W/PV/Q/T at 250/500/700/850/900 hPa -- we are "
                "surface-only; (4) a domain covering the whole Sahara and Arabia, "
                "which is the wide-domain hypothesis already in HANDOVER.md. "
                "Integrated gradients rank lower-tropospheric winds, AOD and SLP "
                "highest. Their misclassified events are the locally-driven ones, "
                "mirroring our finding that 42% of storms are imported.",
    },
    {
        "name": "Sarafian, Nathan, Nissenbaum, Khan & Rudich (CAMS bias correction)",
        "venue": "Remote Sensing 17(2):222, 2025",
        "region": "Eastern Mediterranean; train Turkey (237 stns), test Balkans (38)",
        "task": "correct CAMS PM10 pixel-wise, then forecast dust events 0-72 h",
        "truth": "TOAR in-situ PM10, 275 stations, 2013-2020, 3-hourly",
        "resolution": "CAMS grid 0.75 deg",
        "event_def": "PM10 > 50 ug/m^3 (WHO lowest risk level)",
        "base_rate": 0.22,
        "metrics": {"in_situ_median_ugm3": 36.3, "CAMS_median_ugm3": 18.5,
                    "in_situ_mean_ugm3": 48.0, "CAMS_mean_ugm3": 22.0,
                    "accuracy_gain_ugm3": 12.0},
        "comparable": False,
        "verified": "primary",
        "note": "THE method paper for our biggest finding. They diagnosed what we "
                "diagnosed -- CAMS PM10 runs low against stations, ~2x on their "
                "domain against our 15-49x during Arabian storms -- and then FIXED "
                "it instead of abandoning the field: an XGBoost model of the "
                "pixel-wise CAMS error on meteorological / AOD / NDVI features with "
                "6 h lags, trained on station pixels in one region and applied out "
                "of domain. Corrected fields improved the downstream AI dust-event "
                "forecast at every lead time and every metric. Shallow beat deep "
                "for the correction task. Code: github.com/saginat/CAMS_bias_correction. "
                "We hold 234,512 matched station-hours -- a larger training set for "
                "the error model than they had, over a region where the bias is an "
                "order of magnitude worse. Previously filed here under the wrong "
                "first author, with an accuracy gain of 14.0 and two KL figures "
                "that are not in the paper; all corrected.",
    },
    {
        "name": "Xu, Zhang et al. (physics-guided ML, PGML)",
        "venue": "Sci Rep 16:11824, 2026",
        "region": "Kumtag Desert, NW China (4 AWS)",
        "task": "post-process ECMWF IFS to 5 ordinal visibility grades, 0-72 h",
        "truth": "station visibility, 2020-2022, 3-hourly, 47,122 valid samples",
        "resolution": "station point; 0.25 deg / 3 h IFS predictors (65 of them)",
        "event_def": "ordinal grades; vis <= 10 km is the low-visibility class",
        "base_rate": 0.08,
        "metrics": {"QWK_0_24h": 0.264, "QWK_24_48h": 0.174, "QWK_48_72h": 0.182,
                    "MAE_grades_0_24h": 0.483, "MAE_grades_48_72h": 0.506,
                    "physics_MAE_reduction_min": 0.06,
                    "physics_MAE_reduction_max": 0.10,
                    "dataonly_QWK_loss_24_48h": 0.24},
        "comparable": False,
        "comparable_to": "same phenomenon, ordinal metric -- not a TS",
        "verified": "primary",
        "note": "Our exact problem -- physics plus ML for dust visibility in an "
                "arid region -- and the result that most directly contradicts our "
                "architecture. Physics enters as FEATURES, not as a loss: Shao-Lu "
                "threshold friction velocity, u*, relative saltation flux Q_rel, "
                "gradient Richardson number Ri(850-700), low-level jet height, "
                "10-100 m shear. Measured payoff: MAE -6 to -10%, and removing them "
                "costs 24% of QWK at 24-48 h. Compare our finding #9 (gated physics "
                "loss inert, 0.1-0.3% activation) and the anchor at +0.003 TS "
                "against the bounded head at +0.037 -- the same verdict from both "
                "sides. Their SHAP ranking is a shopping list we cannot currently "
                "buy: t_500, t_700-500 and Ri(850-700) occupy the top 3 at every "
                "horizon and we have NO pressure-level fields at all. Mechanism "
                "shares: vertical mixing >> transport > emission > wet scavenging. "
                "Also: they screen RH >= 70% and precip > 0.1 mm/h out of the "
                "labels so 'low visibility' means dust and not fog -- our fog "
                "control measures the same problem but we do not screen. Imbalance "
                "handled with day-wise SMOTE-Tomek on the training split only, "
                "evaluated on the natural distribution.",
    },
    {
        "name": "WMO SDS-WAS operational multi-model",
        "venue": "WMO SDS-WAS regional centres (NAMEE; GCC node Jeddah since 2023)",
        "region": "Northern Africa / Middle East / Europe",
        "task": "operational dust forecast",
        "truth": "AERONET, MODIS, ground PM10",
        "resolution": "varies by model",
        "event_def": "varies by model and product",
        "base_rate": None,
        "metrics": {"TS_24h": 0.31, "TS_48h": 0.23, "TS_72h": 0.21,
                    "POD_Sahara_Sahel": 0.68, "FAR_Sahara_Sahel": 0.16,
                    "POD_MiddleEast_subregions_low": 0.19,
                    "POD_MiddleEast_subregions_high": 0.48},
        "comparable": False,
        "verified": "snippet",
        "note": "CAUTION -- the TS 0.31 / 0.23 / 0.21 triple has been quoted in "
                "this repo as 'the operational bar' but could not be traced to a "
                "primary source. Chase the AEMET technical report "
                "(dust.aemet.es/resources/model-intercomparison-and-evaluation-of-"
                "dust-forecasts) before using it in a write-up. What IS reported "
                "secondhand is that operational skill over MIDDLE EAST sub-regions "
                "is markedly lower than over the Sahara/Sahel (POD 0.19-0.48 vs "
                ">0.68) -- so the regional bar is probably lower, not higher, than "
                "0.31. GCC node: Jeddah, Saudi NCM, sdsc@ncm.gov.sa.",
    },
    {
        "name": "Karami et al. (9 operational models, Middle East)",
        "venue": "Geosciences 11(11):458, 2021",
        "region": "Middle East (Iraq, Syria, Arabian Peninsula, Iran)",
        "task": "24 h AOD forecast during 12 synoptic dust events",
        "truth": "Terra-MODIS AOD + AERONET (Kuwait Univ, KAUST, Masdar)",
        "resolution": "1 deg comparison grid",
        "event_def": "12 hand-picked events: cyclonic, pre-frontal, post-frontal, Shamal",
        "base_rate": None,
        "metrics": {"R2_DREAM8_MACC": 0.78, "R2_NASA_GEOS": 0.74},
        "comparable": False,
        "verified": "primary",
        "note": "What the physics-based operational models actually achieve on OUR "
                "storm types, verified at KAUST among others. Two things we can "
                "use: (1) every one of the nine models UNDERESTIMATES peak AOD, "
                "worst for cyclonic dust -- the same failure mode we measured in "
                "CAMS, so it is a dust-model problem and not a CAMS-specific one; "
                "(2) the two best models (DREAM8-MACC, NASA-GEOS) are the two that "
                "assimilate MODIS. Assimilation, not resolution, is what moves "
                "these numbers. Column AOD, so not comparable to surface visibility.",
    },
    {
        "name": "Alshammari, Alrwais & Aksoy",
        "venue": "Atmosphere 15(5):520, 2024",
        "region": "Riyadh, Jeddah, Dammam (Saudi Arabia)",
        "task": "MONTHLY dust-storm-DAY COUNT, one month ahead",
        "truth": "Saudi NCM daily weather codes (WMO 06-09, 30-35), 2005-2020",
        "resolution": "city point, monthly",
        "event_def": "count of dust-storm days per month (not an event/no-event label)",
        "base_rate": None,
        "metrics": {"R2_TCN_Riyadh": 0.51, "MAE_TCN_Riyadh": 2.80,
                    "RMSE_TCN_Riyadh": 3.48,
                    "R2_LSTM_Dammam": 0.64, "MAE_LSTM_Dammam": 3.02,
                    "RMSE_LSTM_Dammam": 3.64,
                    "MAE_LSTM_Jeddah": 2.48, "RMSE_LSTM_Jeddah": 2.96},
        "comparable": False,
        "verified": "primary",
        "note": "CORRECTED. Previously filed here as 'Riyadh LSTM/TCN, PM "
                "concentration regression, R2 0.51 ... the realistic bar for "
                "station-level PM prediction'. It is neither PM nor an event "
                "forecast: it regresses the NUMBER OF DUST DAYS IN THE NEXT MONTH "
                "from monthly-averaged predictors, on 192 monthly points per city "
                "with a 70/30 chronological split. MAE 2.80 means 'wrong by about "
                "three dust days per month' and says nothing about which day. It is "
                "the most-cited Saudi ML dust paper, so cite it as the state of "
                "published Saudi work -- and as evidence that the day-resolved "
                "forecast we are attempting is unoccupied ground.",
    },
    {
        "name": "Al-Ahsa dust-storm classifier",
        "venue": "Aerosol Air Qual Res, 2026",
        "region": "Al-Ahsa, Saudi Arabia",
        "task": "daily dust-storm DETECTION (same day, no lead time)",
        "truth": "NOAA ISD meteorology + MODIS MAIAC AOD, 2015-2024",
        "resolution": "station point, daily",
        "event_def": "daily dust-storm label",
        "base_rate": None,
        "metrics": {"accuracy": 0.97, "precision": 0.87, "recall": 0.89,
                    "F1": 0.88, "ROC_AUC": 0.97},
        "comparable": False,
        "verified": "snippet",
        "note": "DO NOT quote these as forecast numbers. Two disqualifiers. "
                "(1) VISIBILITY IS ONE OF THE INPUT FEATURES, alongside temperature "
                "and wind speed -- against a visibility-derived dust label that is "
                "close to circular, and it is why the numbers are so high. "
                "(2) Same-day detection, zero lead time. Accuracy is also the wrong "
                "metric at any dust base rate: at our 1% a never-dust classifier "
                "scores 99% accuracy and TS 0.000. F1 0.88 is the only honest "
                "figure in the row.",
    },
    {
        "name": "AI-DUST (Wang & Hua)",
        "venue": "npj Clean Air, 2025 (s44407-025-00048-z)",
        "region": "East Asia; generalisation tested over the Sahara",
        "task": "dust forecast driven by AI-generated weather forecasts",
        "truth": "CNEMC ground stations + VIIRS AOT; 14 strong spring-2025 events",
        "resolution": "0.25 deg",
        "event_def": "operational SDS concentration thresholds",
        "base_rate": None,
        "metrics": {"TS_day1": 0.33, "TS_day2": 0.28, "TS_day5": 0.26,
                    "TS_day10": 0.23, "R_spatial_day1": 0.57,
                    "RMSE_day1_ugm3": 289.0,
                    "TS_48h_strong_events": 0.42, "gain_vs_operational": 0.27,
                    "corr_1step": 0.99, "corr_80step": 0.61},
        "comparable": False,
        "verified": "snippet",
        "note": "Stacked graph attention plus a physics-based emission scheme, "
                "driven by AIFS rather than by reanalysis. The architectural "
                "lineage of our flux-GNN. Our own audit of it stands (findings "
                "#9-#11: the gated physics loss is inert, the Eq.1 reference is "
                "CFL-unstable at their own resolution, the 15x15 patch is 4.6x "
                "oversized for a radius-3 receptive field) -- but the AIFS-driven "
                "design is the part worth copying, and it is the same point Xu 2026 "
                "makes: post-process a real forecast, do not train on reanalysis "
                "and call the result a forecast.",
    },
    {
        "name": "DustNet",
        "venue": "arXiv:2406.11754, 2024",
        "region": "North Africa (0-31N, 20W-31E)",
        "task": "24 h AOD forecast",
        "truth": "MODIS AOD; baselines CAMS and climatology",
        "resolution": "1 deg",
        "event_def": "continuous AOD, not an event",
        "base_rate": None,
        "metrics": {"r2_annual": 0.91, "r2_annual_CAMS": 0.71,
                    "r2_Bodele": 0.62, "r2_Bodele_CAMS": 0.01,
                    "beats_CAMS_grid_fraction": 0.9526,
                    "MSE_gain_vs_climatology": 0.5368,
                    "MSE_gain_vs_climatology_CAMS": 0.4236},
        "comparable": False,
        "verified": "abstract",
        "note": "Evidence that a small neural net beats the operational physical "
                "model on its own turf -- 1.3M parameters, 7 min training on a "
                "laptop, 2.1 s inference. Useful as an argument, not as a score: it "
                "is column AOD over the Sahara, and CAMS assimilates AOD, so the "
                "comparison flatters the net less than it first looks.",
    },
    {
        "name": "DustCast (SEVIRI diffusion nowcasting)",
        "venue": "2025 (ScienceDirect PII S2212094725000866)",
        "region": "Sahara / Middle East, Meteosat IODC",
        "task": "nowcast the SEVIRI desert-dust RGB composite, 0-6 h",
        "truth": "SEVIRI imagery",
        "resolution": "SEVIRI native, hourly",
        "event_def": "image-space dust plume",
        "base_rate": None,
        "metrics": {"skill_gt_half_lead_h": 5.0, "convective_skill_lead_h": 4.0},
        "comparable": False,
        "verified": "snippet",
        "note": "The honest answer to 'where is there usable warning skill'. Our "
                "operating-point analysis says 24 h is not deployable at any "
                "threshold (POD 0.5 costs FAR 0.90 and 46 alerts/station/yr). "
                "Nowcasting 0-6 h is a different regime with real skill, and the "
                "data exists for our domain: Meteosat 8 has sat at 41.5E since "
                "2017, and DustSCAN (Sci Data, 2024) publishes 9,950 hourly plumes "
                "over 2018-2022 including the Arabian Desert.",
    },
    {
        "name": "AtmoStack (Buraidah)",
        "venue": "air-quality ML literature",
        "region": "Buraidah, Saudi Arabia",
        "task": "PM10 regression",
        "truth": "station",
        "resolution": "station point",
        "event_def": "continuous PM10",
        "base_rate": None,
        "metrics": {"R2_daily": 0.73, "R2_monthly": 0.96},
        "comparable": False,
        "verified": "snippet",
        "note": "Monthly R2 0.96 is aggregation, not forecast skill.",
    },
    {
        "name": "DeepCAMS (Xiao et al.)",
        "venue": "Sci Total Environ, 2022",
        "region": "global",
        "task": "CAMS spatiotemporal downscaling 0.75 -> 0.25 deg",
        "truth": "OpenAQ stations",
        "resolution": "0.25 deg",
        "event_def": "continuous PM10",
        "base_rate": None,
        "metrics": {"R_CAMS_vs_stations": 0.593, "R_DeepCAMS_vs_stations": 0.597},
        "comparable": False,
        "verified": "snippet",
        "note": "Why HANDOVER.md says do not downscale yet: tripling the resolution "
                "moved correlation against stations by 0.004. Sharpening a field "
                "that is 15-49x too low during storms sharpens the error. Fix the "
                "bias first -- see the Sarafian 2025 entry, which is the same data "
                "problem with a working solution.",
    },
]


def references_for(topic):
    """Entries whose name, region, task or note mention ``topic`` (case-insensitive)."""
    t = topic.lower()
    return [r for r in REFERENCES
            if t in " ".join(str(r.get(k, "")) for k in
                             ("name", "region", "task", "note")).lower()]


def comparable_references():
    """Only the entries whose event definition sits near one of our thresholds."""
    return [r for r in REFERENCES if r.get("comparable")]


def format_comparison(our_rows, our_label_filter=None, our_base_rate=None):
    """Render our results beside the published references, caveats attached.

    ``our_base_rate`` is the event fraction of the threshold ``our_rows`` were
    scored at.  Pass it: it is the number that decides whether any of the rows
    below may be set against ours.
    """
    def num(value, width, dec):
        if isinstance(value, (int, float)) and value == value:
            return f"{value:>{width}.{dec}f}"
        return f"{'--':>{width}}"

    lines = []
    header = "OUR RESULTS  (truth = station observations)"
    if our_base_rate is not None:
        header += f"   base rate {our_base_rate:.1%}"
    lines.append(header)
    lines.append("-" * 96)
    lines.append(f"{'model':<22}{'TS':>8}{'ETS':>8}{'HSS':>8}{'POD':>7}"
                 f"{'FAR':>7}{'FGE':>7}{'RMSE':>9}{'TS_d1':>8}")
    for r in our_rows:
        if our_label_filter and r["label"] not in our_label_filter:
            continue
        lines.append(
            f"{str(r['label'])[:22]:<22}"
            f"{num(r.get('TS'), 8, 3)}{num(r.get('ETS'), 8, 3)}"
            f"{num(r.get('HSS'), 8, 3)}{num(r.get('POD'), 7, 3)}"
            f"{num(r.get('FAR'), 7, 3)}{num(r.get('FGE'), 7, 3)}"
            f"{num(r.get('RMSE_ugm3'), 9, 1)}{num(r.get('TS_day1'), 8, 3)}")

    lines.append("")
    lines.append("PUBLISHED REFERENCES")
    lines.append("-" * 96)
    for ref in REFERENCES:
        br = (f"{ref['base_rate']:.0%}" if isinstance(ref.get("base_rate"), float)
              else "unstated")
        flag = "COMPARABLE" if ref.get("comparable") else "not comparable"
        lines.append(f"{ref['name']}  [{ref['venue']}]   <{flag}, {ref['verified']}>")
        lines.append(f"    region={ref['region']} | truth={ref['truth']} | res={ref['resolution']}")
        lines.append(f"    event={ref['event_def']} | base rate={br}")
        lines.append("    " + "  ".join(f"{k}={v}" for k, v in ref["metrics"].items()))
        lines.append(f"    NOTE: {ref['note']}")
        lines.append("")

    lines.append("COMPARABILITY")
    lines.append("-" * 96)
    lines.append("  TS, POD and FAR all depend on the base rate, so a TS comparison is")
    lines.append("  honest only between rows whose event definitions have similar base")
    lines.append("  rates. Every published 'dust event' threshold above sits at or below")
    lines.append("  our `dusty` grade (vis < 5 km, ~7%):")
    lines.append("")
    lines.append("    ours  dusty      TS 0.260  POD 0.413  FAR 0.587   <- compare here")
    lines.append("    ours  dust_storm TS 0.135  POD 0.238  FAR 0.762")
    lines.append("    ours  severe     TS 0.000  POD 0.000  FAR 1.000   <- TS degenerate")
    lines.append("")
    lines.append("  Sarafian 2023 is the one genuinely comparable entry: station-verified,")
    lines.append("  24 h lead, implied TS 0.553 at a 13% base rate. We are behind it, and")
    lines.append("  the gap is explainable -- they use five pressure levels, AOD, a")
    lines.append("  Sahara-wide domain and a dense auxiliary task; we use surface fields")
    lines.append("  over Saudi Arabia alone.")
    lines.append("")
    lines.append("  For the `severe` grade use SEDI, not TS: at a 0.2% base rate TS")
    lines.append("  degenerates to zero for any forecast that is not near-perfect.")
    return "\n".join(lines)
