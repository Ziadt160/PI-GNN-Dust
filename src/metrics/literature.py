"""Published dust-forecast scores, with the caveats that make them comparable or not.

Our benchmark scores against **CAMS reanalysis**; almost every published number
scores against **ground stations**.  Putting the two in one table without saying
so produces a flattering and wrong conclusion, so each entry here carries an
explicit ``truth`` field and a ``comparable`` flag, and the renderer prints them.

Two structural reasons our numbers run high against these references:

1. **Target smoothness.**  CAMS PM10 at 0.75 deg over this domain has p50 79,
   p99 388 and only 129 of 2.93M test cell-times above 1000 ug/m^3.  Station PM10
   during a shamal reaches thousands.  We are scored on a damped field.
2. **CAMS is not truth.**  DeepCAMS validated CAMS against OpenAQ and found
   R ~ 0.59, so a model reproducing CAMS perfectly would still correlate ~0.6 with
   reality.  That is a ceiling no architecture in the roster can pass.

Where a reference reports TS, remember that plain TS is not equitable: on our data
the event base rate is 8.5% and persistence alone scores TS 0.586.  ETS/HSS are
the fair comparison, but most published work does not report them -- which is a
limitation of the comparison, not of our numbers.
"""

# Each entry: metric values as published, plus what they were measured against.
REFERENCES = [
    {
        "name": "AI-DUST (Wang & Hua)",
        "venue": "npj Clean Air, 2026",
        "region": "East Asia",
        "task": "3-hourly dust concentration, 10-day forecast",
        "truth": "CNEMC ground stations + VIIRS AOT",
        "resolution": "0.25 deg",
        "metrics": {"TS_day1": 0.33, "TS_day2": 0.28, "TS_day5": 0.26,
                    "TS_day10": 0.23, "R_spatial_day1": 0.57, "RMSE_day1_ugm3": 289.0},
        "comparable": False,
        "note": "Scored at stations, not grid cells. Their 48-h TS of 0.42 on 14 "
                "strong events beat the KMA operational model's 0.33 by 27%.",
    },
    {
        "name": "WMO SDS-WAS operational multi-model",
        "venue": "WMO SDS-WAS regional centre",
        "region": "Northern Africa / Middle East / Europe",
        "task": "operational dust forecast",
        "truth": "AERONET, MODIS, ground PM10",
        "resolution": "varies by model",
        "metrics": {"TS_24h": 0.31, "TS_48h": 0.23, "TS_72h": 0.21},
        "comparable": False,
        "note": "The operational bar. Metric set (RE, RMSE, R, TS + POD/FAR/CSI) is "
                "the protocol this repo's aidust_metrics mirrors.",
    },
    {
        "name": "Sarafian et al.",
        "venue": "npj Clim Atmos Sci, 2023",
        "region": "Middle East (Israel)",
        "task": "dust-event early warning, multi-task PM10",
        "truth": "in-situ PM10 + satellite regional PM10",
        "resolution": "station point",
        "metrics": {"POD_24h": 0.76},
        "comparable": False,
        "note": "Detects 76% of events 24 h ahead. Point forecasts, not a field.",
    },
    {
        "name": "CAMS bias correction (Nissenbaum et al.)",
        "venue": "Remote Sensing, 2025",
        "region": "Middle East",
        "task": "correct CAMS PM10, then forecast dust events",
        "truth": "in-situ stations",
        "resolution": "CAMS grid",
        "metrics": {"KL_corrected": 0.41, "KL_raw_CAMS": 0.56,
                    "accuracy_gain_ugm3": 14.0},
        "comparable": False,
        "note": "Directly addresses our CAMS-is-not-truth problem. Code at "
                "github.com/saginat/CAMS_bias_correction.",
    },
    {
        "name": "Al-Ahsa dust-storm classifier",
        "venue": "Aerosol Air Qual Res, 2026",
        "region": "Al-Ahsa, Saudi Arabia",
        "task": "binary dust-storm classification",
        "truth": "NOAA met + MODIS AOD",
        "resolution": "station point",
        "metrics": {"accuracy": 0.97, "precision": 0.87, "recall": 0.89,
                    "F1": 0.88, "ROC_AUC": 0.97},
        "comparable": False,
        "note": "ACCURACY IS MISLEADING HERE. At our 8.5% base rate a model that "
                "never forecasts dust scores 91.5% accuracy and TS 0.000. Their F1 "
                "of 0.88 is the honest figure. Classification task, not a field "
                "forecast -- not comparable to TS on gridded concentration.",
    },
    {
        "name": "Riyadh LSTM / TCN",
        "venue": "air-quality ML literature",
        "region": "Riyadh",
        "task": "PM concentration regression",
        "truth": "station",
        "resolution": "station point",
        "metrics": {"R2": 0.51, "MAE": 2.80, "RMSE": 3.48},
        "comparable": False,
        "note": "Point time-series regression. R2 0.51 at a single station is the "
                "realistic bar for station-level PM prediction in this region.",
    },
    {
        "name": "AtmoStack (Buraidah)",
        "venue": "air-quality ML literature",
        "region": "Buraidah, Saudi Arabia",
        "task": "PM10 regression",
        "truth": "station",
        "resolution": "station point",
        "metrics": {"R2_daily": 0.73, "R2_monthly": 0.96},
        "comparable": False,
        "note": "Monthly R2 0.96 is aggregation, not forecast skill.",
    },
    {
        "name": "DeepCAMS (Xiao et al.)",
        "venue": "Sci Total Environ, 2022",
        "region": "global",
        "task": "CAMS spatiotemporal downscaling",
        "truth": "OpenAQ stations",
        "resolution": "0.75 -> 0.25 deg",
        "metrics": {"R_CAMS_vs_stations": 0.593, "R_DeepCAMS_vs_stations": 0.597},
        "comparable": False,
        "note": "The source of our ceiling estimate: CAMS itself correlates ~0.59 "
                "with in-situ PM. Tripling resolution moved R by 0.004.",
    },
]


def format_comparison(our_rows, our_label_filter=None):
    """Render our results beside the published references, caveats attached."""
    lines = []
    lines.append("OUR RESULTS  (truth = CAMS reanalysis, 0.75 deg, 251 cells)")
    lines.append("-" * 96)
    lines.append(f"{'model':<22}{'TS@200':>8}{'ETS':>8}{'HSS':>8}{'POD':>7}"
                 f"{'FAR':>7}{'FGE':>7}{'RMSE':>9}{'TS_d1':>8}")
    for r in our_rows:
        if our_label_filter and r["label"] not in our_label_filter:
            continue
        g = lambda k, f="{:>8.3f}": (f.format(r[k]) if isinstance(r.get(k), (int, float))
                                     and r[k] == r[k] else f.replace("f", "s").format("--"))
        lines.append(f"{str(r['label'])[:22]:<22}{g('TS')}{g('ETS')}{g('HSS')}"
                     f"{g('POD','{:>7.3f}')}{g('FAR','{:>7.3f}')}{g('FGE','{:>7.3f}')}"
                     f"{g('RMSE_ugm3','{:>9.1f}')}{g('TS_day1')}")

    lines.append("")
    lines.append("PUBLISHED REFERENCES")
    lines.append("-" * 96)
    for ref in REFERENCES:
        ms = "  ".join(f"{k}={v}" for k, v in ref["metrics"].items())
        lines.append(f"{ref['name']}  [{ref['venue']}]")
        lines.append(f"    region={ref['region']} | truth={ref['truth']} | res={ref['resolution']}")
        lines.append(f"    {ms}")
        lines.append(f"    NOTE: {ref['note']}")
        lines.append("")

    lines.append("COMPARABILITY")
    lines.append("-" * 96)
    lines.append("  NONE of the references above are directly comparable to our numbers.")
    lines.append("  Every one scores against ground stations or point observations;")
    lines.append("  we score against CAMS reanalysis, which DeepCAMS measured at R~0.59")
    lines.append("  versus in-situ PM. Our target is also damped: CAMS p99 = 388 ug/m^3")
    lines.append("  over the test set, while real dust storms reach thousands.")
    lines.append("  Report our results as 'reproduces CAMS reanalysis', not 'forecasts PM10'.")
    return "\n".join(lines)
