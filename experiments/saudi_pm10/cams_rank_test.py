"""Does CAMS PM10 get the ORDER of dust storms right, even though the scale is wrong?

We know CAMS under-reports storm dust by 15-50x. That kills it as a training
target. It does NOT automatically kill it as a predictor: a field that is
uniformly too low but correctly ORDERED can be rescaled, and Sarafian et al. 2025
did exactly that to good effect.

The question that decides it is whether CAMS has SATURATED. If it reports ~300
whether reality is 1,000 or 10,000, the ordering is gone and no correction
recovers it. If it still ranks a bad storm above a mild one, the information is
there and merely mis-scaled.

Rank correlation answers this. It ignores magnitude entirely and asks only
whether the two series rise and fall together.

Memory-light on purpose: selects the ~26 station cells lazily before loading
anything, so it can run beside a trainer holding several GB.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import paths  # noqa: E402

MERGED = paths.SAUDI_MERGED
METAR = paths.METAR_DIR
MILES_TO_KM = 1.609344
KOSCHMIEDER = 3.912
DUST = r"DS|SS|BLDU|BLSA|DU|SA|PO"

stations = json.load(open(os.path.join(METAR, "stations.json")))
ds = xr.open_dataset(MERGED)
pm_name = "pm10" if "pm10" in ds else [v for v in ds.data_vars if "pm10" in v.lower()][0]
tname = "valid_time" if "valid_time" in ds.coords else list(ds[pm_name].dims)[0]
times = pd.to_datetime(ds[tname].values, utc=True)
print(f"CAMS field '{pm_name}', {len(times):,} timesteps\n")

rows = []
for s in stations:
    path = os.path.join(METAR, f"{s['id']}.csv")
    if not os.path.exists(path):
        continue
    # One grid cell, loaded on its own -- never the whole field.
    cams = ds[pm_name].isel({ds[pm_name].dims[1]: s["row"],
                             ds[pm_name].dims[2]: s["col"]}).values * 1e9  # -> ug/m3
    cams = pd.Series(cams, index=times)

    d = pd.read_csv(path, na_values=["M", "T"], usecols=["valid", "vsby", "wxcodes"])
    t = pd.to_datetime(d["valid"], utc=True, errors="coerce")
    vis = pd.to_numeric(d["vsby"], errors="coerce") * MILES_TO_KM
    code = d["wxcodes"].fillna("").astype(str).str.contains(DUST, regex=True)
    obs = pd.DataFrame({"vis": vis.values, "dust": code.values}, index=t)
    obs = obs[~obs.index.isna()].sort_index()
    obs = obs[~obs.index.duplicated(keep="first")]

    # Extinction is what is ~linear in dust mass; visibility is its reciprocal.
    obs["beta"] = np.where(obs["vis"] > 0, KOSCHMIEDER / obs["vis"], np.nan)
    m = obs.reindex(cams.index, method="nearest", tolerance=pd.Timedelta("90min"))
    ok = m["beta"].notna().to_numpy() & np.isfinite(cams.to_numpy())
    if ok.sum() < 500:
        continue
    rows.append(pd.DataFrame({"station": s["id"], "cams": cams.to_numpy()[ok],
                              "beta": m["beta"].to_numpy()[ok],
                              "vis": m["vis"].to_numpy()[ok],
                              "dust": m["dust"].to_numpy()[ok].astype(bool)}))

a = pd.concat(rows, ignore_index=True)
print(f"matched station-hours: {len(a):,} across {a.station.nunique()} stations\n")


def report(name, sub):
    if len(sub) < 100:
        print(f"{name:<34} (too few: {len(sub)})")
        return
    rho = stats.spearmanr(sub["cams"], sub["beta"]).statistic
    r = stats.pearsonr(np.log1p(sub["cams"]), np.log(sub["beta"])).statistic
    print(f"{name:<34}{len(sub):>9,}{rho:>10.3f}{r:>10.3f}"
          f"{sub['cams'].median():>12.0f}{sub['cams'].quantile(0.99):>10.0f}")


print(f"{'subset':<34}{'n':>9}{'Spearman':>10}{'Pearson':>10}"
      f"{'CAMS med':>12}{'CAMS p99':>10}")
print("-" * 85)
report("all matched hours", a)
report("dust-coded hours", a[a.dust])
report("dust-coded AND vis < 5 km", a[a.dust & (a.vis < 5)])
report("dust-coded AND vis < 1 km", a[a.dust & (a.vis < 1)])

# Saturation test: within storms only, does CAMS still separate bad from worse?
storm = a[a.dust & (a.vis < 5)]
if len(storm) > 200:
    q = pd.qcut(storm["beta"], 4, labels=["mildest", "mild", "bad", "worst"])
    print("\nWithin dust storms, does CAMS rise with severity?")
    print(f"{'observed severity quartile':<34}{'n':>9}{'CAMS median':>14}")
    for lab, g in storm.groupby(q, observed=True):
        print(f"  {str(lab):<32}{len(g):>9,}{g['cams'].median():>14.0f}")
    lo = storm[q == "mildest"]["cams"].median()
    hi = storm[q == "worst"]["cams"].median()
    print(f"\n  worst / mildest ratio in CAMS: {hi/max(lo,1e-9):.2f}x")
    print(f"  worst / mildest ratio in reality: "
          f"{storm[q=='worst']['beta'].median()/storm[q=='mildest']['beta'].median():.2f}x")
ds.close()
