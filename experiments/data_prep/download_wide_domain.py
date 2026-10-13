"""Download CAMS + ERA5 over a domain that actually contains the dust sources.

Why the domain has to grow
--------------------------
Measured on 3,369 observed dust-storm hours with real ERA5 friction velocity:

    58% had local u* above the saltation threshold  -> could be emitted locally
    42% did NOT                                     -> the dust was IMPORTED
    68% of storms arrive on a northerly wind        -> the classic shamal

The Mesopotamian floodplain and Kuwait -- the dominant shamal sources -- sit
inside the current grid but outside its land mask, and the Syrian desert is off
the grid entirely.  The northernmost modelled cell is 30.93 N.  So for roughly
four in ten dust storms the model is asked to predict an arrival whose origin has
been deleted, and the graph's northern edge is a wall with zero concentration
behind it.  That is consistent with the permutation test, where the transport
term scored just 3% of total feature importance.

New domain: 12-38 N, 30-60 E at 0.75 deg = 35 x 41 = 1,435 cells (5.7x the
current 251 land cells), covering Mesopotamia, the Syrian desert, Kuwait, the
whole Arabian Peninsula and the Red Sea approaches.

Two data stores
---------------
CAMS reanalysis (EAC4) lives on the Atmosphere Data Store, ERA5 on the Climate
Data Store.  Both accept the same ECMWF token (verified), but they are different
endpoints, so the client is pointed at each explicitly rather than relying on
~/.cdsapirc.

Downloads are per-year and resumable; an existing non-empty file is skipped.

Usage
-----
  python experiments/data_prep/download_wide_domain.py test            # one month, both stores
  python experiments/data_prep/download_wide_domain.py cams 2003 2024
  python experiments/data_prep/download_wide_domain.py era5 2003 2024
  python experiments/data_prep/download_wide_domain.py all  2003 2024
"""
import glob
import os
import shutil
import sys
import tempfile
import time
import zipfile

import cdsapi
import xarray as xr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import paths  # noqa: E402

OUT = paths.WIDE_DIR

# [North, West, South, East] -- includes Mesopotamia, Syria, Kuwait, the Peninsula.
AREA = [38.0, 30.0, 12.0, 60.0]
GRID = [0.75, 0.75]
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]

CDS_URL = "https://cds.climate.copernicus.eu/api"
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"

CAMS_VARS = ["particulate_matter_10um", "particulate_matter_2.5um"]
ERA5_VARS = ["10m_u_component_of_wind", "10m_v_component_of_wind",
             "2m_temperature", "2m_dewpoint_temperature", "total_precipitation",
             "friction_velocity", "volumetric_soil_water_layer_1",
             "boundary_layer_height", "forecast_surface_roughness"]


def token():
    txt = open(os.path.expanduser("~/.cdsapirc")).read()
    return txt.split("key:")[1].strip().splitlines()[0]


def client(url):
    return cdsapi.Client(url=url, key=token())


def extract(raw, dest):
    """CDS/ADS may return a bare .nc or a ZIP of instantaneous/accumulated streams."""
    if not zipfile.is_zipfile(raw):
        os.replace(raw, dest)
    else:
        tmp = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(raw) as z:
                z.extractall(tmp)
            parts = [xr.open_dataset(f) for f in glob.glob(os.path.join(tmp, "*.nc"))]
            ds = xr.merge(parts, compat="override").load()
            for p in parts:
                p.close()
            ds.to_netcdf(dest, encoding={v: {"zlib": True, "complevel": 4}
                                         for v in ds.data_vars})
            ds.close()
        finally:
            if os.path.exists(raw):
                os.remove(raw)
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"    -> {os.path.basename(dest)} ({os.path.getsize(dest)/1e6:.0f} MB)",
          flush=True)


def fetch(cli, dataset, request, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print(f"    skip (exists): {os.path.basename(dest)}", flush=True)
        return True
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    raw = dest + ".part"
    try:
        cli.retrieve(dataset, request, raw)
        extract(raw, dest)
        return True
    except Exception as exc:
        if os.path.exists(raw):
            os.remove(raw)
        print(f"    FAILED {os.path.basename(dest)}: {type(exc).__name__}: {exc}",
              flush=True)
        return False


def cams_year(year, months=None):
    cli = client(ADS_URL)
    dest = os.path.join(OUT, "cams", f"cams_pm_{year}.nc")
    req = {"variable": CAMS_VARS, "date": f"{year}-01-01/{year}-12-31",
           "time": TIMES, "area": AREA, "grid": GRID,
           "data_format": "netcdf"}
    if months:
        req["date"] = f"{year}-{months[0]}-01/{year}-{months[-1]}-28"
    return fetch(cli, "cams-global-reanalysis-eac4", req, dest)


def _era5_request(cli, year, months, dest):
    req = {"product_type": ["reanalysis"], "variable": ERA5_VARS,
           "year": [str(year)], "month": months, "day": DAYS,
           "time": TIMES, "area": AREA, "grid": GRID, "data_format": "netcdf"}
    return fetch(cli, "reanalysis-era5-single-levels", req, dest)


def era5_year(year, months=None):
    """One year of ERA5, chunked to stay under the CDS per-request cost limit.

    A whole year of 9 variables over this 35 x 41 domain is 26,784 fields and is
    rejected with "cost limits exceeded"; the wider domain is what pushed it over,
    since the same request shape worked on the smaller Saudi box.  Quarters are
    tried first (fewer requests, less queueing) and months are the fallback.
    """
    cli = client(CDS_URL)
    if months:                                   # explicit subset (test mode)
        return _era5_request(cli, year, months,
                             os.path.join(OUT, "era5", f"era5_{year}.nc"))

    quarters = [MONTHS[i:i + 3] for i in range(0, 12, 3)]
    ok_all = True
    for q, qm in enumerate(quarters, 1):
        dest = os.path.join(OUT, "era5", f"era5_{year}_q{q}.nc")
        if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
            print(f"    skip (exists): {os.path.basename(dest)}", flush=True)
            continue
        if _era5_request(cli, year, qm, dest):
            continue
        print(f"    quarter {q} rejected; falling back to monthly", flush=True)
        for m in qm:
            ok_all &= _era5_request(cli, year, [m],
                                    os.path.join(OUT, "era5", f"era5_{year}_{m}.nc"))
    return ok_all


def main(argv):
    mode = argv[0] if argv else "test"
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    if mode == "test":
        print(f"TEST: one month over {AREA} at {GRID[0]} deg", flush=True)
        ok_c = cams_year(2020, months=["06"])
        ok_e = era5_year(2020, months=["06"])
        print(f"\nCAMS {'OK' if ok_c else 'FAILED'} | ERA5 {'OK' if ok_e else 'FAILED'}")
        return

    y0, y1 = int(argv[1]), int(argv[2])
    todo = []
    if mode in ("cams", "all"):
        todo += [("cams", y) for y in range(y0, y1 + 1)]
    if mode in ("era5", "all"):
        todo += [("era5", y) for y in range(y0, y1 + 1)]

    done = 0
    for what, year in todo:
        print(f"[{done+1}/{len(todo)}] {what} {year}", flush=True)
        ok = cams_year(year) if what == "cams" else era5_year(year)
        done += 1
        print(f"    {(time.time()-t0)/60:.0f} min elapsed", flush=True)
    print(f"\nfinished {done} requests in {(time.time()-t0)/60:.0f} min")


if __name__ == "__main__":
    main(sys.argv[1:])
