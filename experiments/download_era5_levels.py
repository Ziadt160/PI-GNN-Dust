"""Download the ERA5 PRESSURE-LEVEL fields -- the half of the physics we never had.

Why this exists
---------------
Every feature this project has ever fed a model is a surface field.  The two
studies closest to our problem both use five pressure levels, and in the one that
publishes a feature ranking (Xu et al. 2026, Sci Rep 16:11824, dust visibility
over the Kumtag Desert) the top three predictors at *every* forecast horizon are
pressure-level quantities we do not hold:

    t_500          500 hPa temperature
    t_700-500      mid-tropospheric temperature difference
    Ri(850-700)    gradient Richardson number

Their mechanism grouping is blunter still: vertical mixing dominates transport
and emission at all leads.  That is a claim about what sets surface visibility --
once dust is airborne, boundary-layer stability decides whether it reaches eye
level or is mixed aloft -- and it is invisible in a surface-only feature set.

It may also explain our own finding #5, that graph transport carries only 3% of
feature importance.  A transport model with no stability information is being
asked to explain visibility with the wrong half of the physics.

Level choice follows Sarafian et al. 2023 (npj Clim Atmos Sci 6:23) exactly:
250 / 500 / 700 / 850 / 900 hPa.  Not a free parameter -- 900 and 850 straddle the
dust-carrying boundary layer, 700 caps it, 500 carries the synoptic wave, and 250
is the jet.

Cost, and why the chunking looks odd
------------------------------------
The pressure-level dataset enforces a far tighter per-request cost limit than
`reanalysis-era5-single-levels`, which is what `download_era5_emission.py` pulls.
Measured against the live API rather than guessed:

    1 variable,  1 month   (1,240 fields)    OK
    1 variable,  3 months  (3,720 fields)    OK      <- what this script requests
    5 variables, 1 month   (6,200 fields)    REJECTED "request is too large"
    1 variable,  1 year   (14,600 fields)    REJECTED

So the unit is **one variable, one quarter**: 5 variables x 4 quarters x 22 years
= 440 requests, about 5.8 MB each, so ~2.6 GB in total.  Fully resumable at part
level -- each part is skipped if already on disk, and a year's 20 parts are merged
into one `era5_lev_<year>.nc` only once all of them are present, so killing the
script mid-year loses nothing.

**Concurrency is capped by the service, not by us, and the cap is low.**  One
pressure-level request takes about eight minutes end to end, nearly all of it
queue wait rather than transfer, so serial would be 440 x 8 minutes -- close to
60 hours.  The obvious fix is a thread pool, and it does not work the way you
would hope: at 6 workers CDS rejected 15 of 20 parts outright with

    Number queued requests for this dataset is temporarily limited

The limit is on requests QUEUED per dataset.  Going wider converts successes into
rejections rather than finishing sooner, so WORKERS is 2 and every part retries
with exponential backoff instead of being dropped.  Budget roughly **24-30 hours**
for the full 22 years.  If that is too long, run the test and validation years
first (``2019 2024``) -- but note that training on years without levels and
testing on years with them is the distribution shift this file's next paragraph
warns about, so a partial archive is for experimenting, not for a reported result.

**Download all 22 years, not a recent subset.**  The training split is 2003-2018.
Holding levels for 2015-2024 only would mean training on proxies and testing on
measurements, which is the train/test distribution shift `min_driver_coverage`
exists to prevent.

Usage:
  python experiments/download_era5_levels.py test         # one month, to check the shape
  python experiments/download_era5_levels.py 2003 2024    # the full record, resumable
  python experiments/download_era5_levels.py 2019 2024    # test+val years first
"""
import os
import sys
import glob
import shutil
import zipfile
import tempfile
import time
import concurrent.futures as cf

import xarray as xr
import cdsapi

OUT_DIR = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/levels"

# Exactly the saudi_merged.nc footprint, so no regridding downstream.
AREA = [33.18, 33.80, 15.18, 56.30]   # [North, West, South, East]
GRID = [0.75, 0.75]

# Sarafian et al. 2023's set. See the module docstring on why these five.
LEVELS = ["250", "500", "700", "850", "900"]

LEVEL_VARS = {
    # CDS name                  short name in the netCDF
    "geopotential": "z",         # -> layer thickness, 500 hPa height for the synoptic pattern
    "temperature": "t",          # -> potential temperature, t_500, t_700-500
    "specific_humidity": "q",    # -> moisture gate on dust, and virtual temperature
    "u_component_of_wind": "u",  # -> vertical shear, low-level jet
    "v_component_of_wind": "v",
}

TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
# One variable per quarter is the largest request the cost limit accepts.
QUARTERS = [MONTHS[0:3], MONTHS[3:6], MONTHS[6:9], MONTHS[9:12]]

# Concurrent CDS requests. MEASURED, not guessed: at 6 the service rejected 15 of
# 20 parts outright with "Number queued requests for this dataset is temporarily
# limited". The cap is on requests QUEUED per dataset, not on bandwidth, so going
# wider does not go faster -- it converts successes into rejections. 2 is what the
# service actually tolerates here.
WORKERS = int(os.environ.get("ERA5_LEVEL_WORKERS", "2"))

# The queue cap is transient, so a rejection means "come back later", not "give
# up". Without a retry the part is simply lost until someone re-runs the script.
MAX_ATTEMPTS = 8
BACKOFF_S = 90          # doubles each attempt, capped
BACKOFF_MAX_S = 900
THROTTLE_MARKERS = ("temporarily limited", "too many", "rate limit", "429")


def _extract(zip_or_nc, out_nc):
    """CDS may return a bare .nc or a ZIP bundling several streams."""
    if not zipfile.is_zipfile(zip_or_nc):
        os.replace(zip_or_nc, out_nc)
    else:
        tmp = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(zip_or_nc) as z:
                z.extractall(tmp)
            parts = [xr.open_dataset(f) for f in glob.glob(os.path.join(tmp, "*.nc"))]
            ds = xr.merge(parts, compat="override").load()
            for p in parts:
                p.close()
            ds.to_netcdf(out_nc, encoding={v: {"zlib": True, "complevel": 4}
                                           for v in ds.data_vars})
            ds.close()
        finally:
            os.remove(zip_or_nc)
            shutil.rmtree(tmp, ignore_errors=True)   # Windows: open handles -> ignore
    print(f"  -> {out_nc} ({os.path.getsize(out_nc)/1e6:.1f} MB)", flush=True)


def _retrieve(request, out_nc):
    """Fetch one part, waiting out the queue cap rather than dropping the part."""
    if os.path.exists(out_nc):
        print(f"  skip (exists): {os.path.basename(out_nc)}", flush=True)
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = out_nc + ".download"
    name = os.path.basename(out_nc)

    delay = BACKOFF_S
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            cdsapi.Client(quiet=True).retrieve(
                "reanalysis-era5-pressure-levels", request, raw)
            _extract(raw, out_nc)
            return
        except Exception as exc:
            msg = str(exc)
            throttled = any(m in msg.lower() for m in THROTTLE_MARKERS)
            if os.path.exists(raw):
                os.remove(raw)
            if attempt == MAX_ATTEMPTS or not throttled:
                raise
            print(f"  queue full, retrying {name} in {delay}s "
                  f"(attempt {attempt}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, BACKOFF_MAX_S)


def _base():
    return {"product_type": "reanalysis", "format": "netcdf",
            "pressure_level": LEVELS, "area": AREA, "grid": GRID}


def _part_path(year, short, q):
    return os.path.join(OUT_DIR, f"_part_{year}_{short}_q{q + 1}.nc")


def _part_jobs(year):
    """Every (request, path) this year needs, in a flat list the pool can chew on."""
    jobs = []
    for cds_name, short in LEVEL_VARS.items():
        for q, months in enumerate(QUARTERS):
            jobs.append((dict(_base(), variable=[cds_name], year=[str(year)],
                              month=months, day=DAYS, time=TIMES),
                         _part_path(year, short, q)))
    return jobs


def download_year(year, workers=WORKERS):
    """20 parts (5 variables x 4 quarters) in parallel, merged into one file."""
    out_nc = os.path.join(OUT_DIR, f"era5_lev_{year}.nc")
    if os.path.exists(out_nc):
        print(f"  skip (exists): {os.path.basename(out_nc)}", flush=True)
        return

    jobs = _part_jobs(year)
    todo = [(r, p) for r, p in jobs if not os.path.exists(p)]
    print(f"Levels {year}: {len(todo)}/{len(jobs)} parts to fetch "
          f"({workers} at a time) ...", flush=True)

    if todo:
        os.makedirs(OUT_DIR, exist_ok=True)
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_retrieve, r, p): p for r, p in todo}
            for fut in cf.as_completed(futures):
                path = futures[fut]
                try:
                    fut.result()
                except Exception as exc:               # one bad part must not
                    print(f"  FAILED {os.path.basename(path)}: "   # kill the rest
                          f"{type(exc).__name__}: {str(exc).splitlines()[-1][:110]}",
                          flush=True)

    parts = [p for _, p in jobs]
    # Merge only once every part is present, so an interrupted year leaves its
    # parts on disk to be resumed rather than a half-written year file that the
    # skip-if-exists check would then treat as complete.
    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        print(f"  {year} incomplete ({len(missing)} part(s) missing) -- "
              f"parts kept, re-run to resume", flush=True)
        return

    print(f"  merging {len(parts)} parts -> {os.path.basename(out_nc)}", flush=True)
    opened = [xr.open_dataset(p) for p in parts]
    try:
        per_var = {}
        for ds in opened:
            for name in ds.data_vars:
                per_var.setdefault(name, []).append(ds[name])
        merged = xr.Dataset({
            name: xr.concat(chunks, dim="valid_time").sortby("valid_time")
            for name, chunks in per_var.items()}).load()
        merged.to_netcdf(out_nc, encoding={v: {"zlib": True, "complevel": 4}
                                           for v in merged.data_vars})
        merged.close()
    finally:
        for ds in opened:
            ds.close()
    for p in parts:
        os.remove(p)
    print(f"  -> {out_nc} ({os.path.getsize(out_nc)/1e6:.1f} MB)", flush=True)


def main(argv):
    if argv and argv[0] == "test":
        # One part only: checks credentials, the area/grid, and the returned shape.
        _retrieve(dict(_base(), variable=["temperature"], year=["2020"],
                       month=QUARTERS[1], day=DAYS, time=TIMES),
                  _part_path(2020, "t", 1))
        return
    y0 = int(argv[0]) if argv else 2003
    y1 = int(argv[1]) if len(argv) > 1 else 2024
    workers = int(argv[2]) if len(argv) > 2 else WORKERS
    for year in range(y0, y1 + 1):
        download_year(year, workers)
    print("done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
