"""Download MODIS MAIAC AOD (MCD19A2 v061) -- the only independent observation we can get.

Why this matters
----------------
Every number in `benchmark_aidust.py` is scored against CAMS reanalysis, which is a
model output, not an observation.  DeepCAMS measured CAMS against ground monitors
and found R ~ 0.59, so our models are being graded against a field that is itself
~40% decorrelated from reality.  OpenAQ turned out to have only 2 usable PM10
stations inside this domain (0.8% of cells), so station validation is not
available here.

MAIAC AOD is the remaining independent check: a satellite retrieval that knows
nothing about CAMS, gridded at 1 km with near-complete coverage over a desert
domain.  It is **column aerosol optical depth, not surface PM10**, so it cannot
give an absolute ug/m^3 validation without a PM-AOD calibration (which would
itself need the monitors we lack).  What it does give is spatial-pattern and
event-detection agreement -- exactly the validation AI-DUST used for their
Arabian Peninsula generalisation test (their Figs. 7 and 11).

Tiles
-----
The 10 sinusoidal tiles covering 33.8-56.3E, 15.18-33.18N, verified against CMR:
h20v05 h20v06 h21v05 h21v06 h21v07 h22v05 h22v06 h22v07 h23v06 h23v07

Note the Riyadh 1 km box (24.3-25.1N) falls in **h22v06**, not the h22v05/h23v05
that an earlier note in riyadh_1km/DATA.md claimed -- v05 spans ~30-40N.

Usage
-----
  python experiments/download_maiac.py validation      # every 12th day, ~12 GB
  python experiments/download_maiac.py riyadh          # tile h22v06 daily, ~18 GB
  python experiments/download_maiac.py events          # CAMS-flagged dust days
  python experiments/download_maiac.py validation --stride 24 --dry-run

Credentials: a bearer token in ``~/.edl_token`` (Earthdata Login -> Generate Token).
Downloads are resumable -- existing files are skipped, partials are re-fetched.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta

OUT_DIR = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/maiac"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
SHORT_NAME, VERSION = "MCD19A2", "061"

DOMAIN_BBOX = "33.8,15.18,56.3,33.18"      # W,S,E,N -- the saudi_merged.nc footprint
RIYADH_BBOX = "46.3,24.3,47.1,25.1"

RETRIES, BACKOFF, TIMEOUT = 4, 5, 300


def token():
    path = os.path.expanduser("~/.edl_token")
    if not os.path.exists(path):
        sys.exit("No ~/.edl_token. Create one at urs.earthdata.nasa.gov -> Generate Token.")
    return open(path).read().strip()


def cmr_day(day, bbox):
    """Granules intersecting `bbox` on one UTC day."""
    params = {
        "short_name": SHORT_NAME, "version": VERSION, "bounding_box": bbox,
        "temporal": f"{day}T00:00:00Z,{day}T23:59:59Z", "page_size": 200,
    }
    req = urllib.request.Request(CMR + "?" + urllib.parse.urlencode(params),
                                 headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.load(r)["feed"]["entry"]


def data_url(granule):
    for link in granule.get("links", []):
        if link.get("rel", "").endswith("/data#") and link["href"].startswith("http"):
            return link["href"]
    return None


def fetch(url, dest, tok):
    """Resumable single-granule download. Returns bytes written, or 0 if skipped."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return 0
    tmp = dest + ".part"
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + tok})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            size = os.path.getsize(tmp)
            if size < 1024:                       # an error page, not an HDF
                raise IOError(f"suspiciously small response ({size} B)")
            os.replace(tmp, dest)
            return size
        except Exception as exc:
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == RETRIES - 1:
                print(f"    FAILED {os.path.basename(dest)}: {type(exc).__name__}: {exc}",
                      flush=True)
                return -1
            time.sleep(BACKOFF * (attempt + 1))
    return -1


def select_days(mode, stride, start, end):
    """Which UTC days to pull."""
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)

    if mode == "events":
        return [d.isoformat() for d in cams_event_days(days)]
    return [d.isoformat() for d in days[::stride]]


def cams_event_days(days):
    """Days on which CAMS PM10 exceeds the 200 ug/m^3 dust threshold anywhere in domain.

    Concentrates the download on the cases the threat score actually scores, rather
    than the clean-air majority.
    """
    import numpy as np
    import xarray as xr
    merged = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/saudi_merged.nc"
    ds = xr.open_dataset(merged)
    mask = ds.valid_mask.values.astype(bool)
    pm = ds.pm10.values[:, mask] * 1e9
    times = ds.valid_time.values.astype("datetime64[D]")
    ds.close()
    hit = pm.max(axis=1) >= 200.0
    event_days = set(np.unique(times[hit]).astype(str))
    return [d for d in days if d.isoformat() in event_days]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["validation", "riyadh", "events"])
    ap.add_argument("--stride", type=int, default=12,
                    help="keep every Nth day (validation mode); 12 ~= 12 GB over 4 years")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bbox = RIYADH_BBOX if args.mode == "riyadh" else DOMAIN_BBOX
    stride = 1 if args.mode in ("riyadh", "events") else args.stride
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    days = select_days(args.mode, stride, start, end)

    out_dir = os.path.join(args.out, args.mode)
    os.makedirs(out_dir, exist_ok=True)
    tok = token()
    print(f"mode={args.mode} bbox={bbox} days={len(days)} -> {out_dir}", flush=True)

    total_bytes, got, skipped, failed, seen = 0, 0, 0, 0, 0
    t0 = time.time()
    for i, day in enumerate(days, 1):
        try:
            granules = cmr_day(day, bbox)
        except Exception as exc:
            print(f"  [{i}/{len(days)}] {day} CMR query failed: {exc}", flush=True)
            continue
        seen += len(granules)
        if args.dry_run:
            if i <= 3 or i % 20 == 0:
                mb = sum(float(g.get("granule_size", 0)) for g in granules)
                print(f"  [{i}/{len(days)}] {day}: {len(granules)} granules, {mb:.0f} MB",
                      flush=True)
            continue

        for g in granules:
            url = data_url(g)
            if not url:
                continue
            dest = os.path.join(out_dir, url.split("/")[-1])
            n = fetch(url, dest, tok)
            if n > 0:
                total_bytes += n
                got += 1
            elif n == 0:
                skipped += 1
            else:
                failed += 1

        if i % 5 == 0 or i == len(days):
            el = time.time() - t0
            rate = total_bytes / max(el, 1) / 1e6
            print(f"  [{i}/{len(days)}] {day} | new {got} skip {skipped} fail {failed} "
                  f"| {total_bytes/1e9:.2f} GB | {rate:.1f} MB/s | {el/60:.0f} min",
                  flush=True)

    if args.dry_run:
        print(f"\nDRY RUN: {len(days)} days, {seen} granules matched")
    else:
        print(f"\ndone: {got} new, {skipped} already present, {failed} failed, "
              f"{total_bytes/1e9:.2f} GB in {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main()
