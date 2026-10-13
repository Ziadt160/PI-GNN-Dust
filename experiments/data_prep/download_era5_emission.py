"""Download the ERA5 fields needed for a real dust-emission scheme over the Saudi domain.

`saudi_merged.nc` has pm10/pm2p5/u10/v10/t2m/rh/tp but none of the surface
properties a Shao-type saltation scheme needs.  AI-DUST drives its emission from
WRF friction velocity + SoilGrids texture + MODIS NDVI + a high-resolution
roughness density; the closest freely-available equivalents are these ERA5
single-level fields.

Download is split by how fast each field actually varies, which cuts the CDS
cost ~2.5x versus pulling everything 3-hourly:

  DYNAMIC   (3-hourly, 2003-2024)  zust, swvl1, blh
      zust   friction velocity        -> saltation driver u*
      swvl1  soil water layer 1       -> moisture correction on u*t (Fecan)
      blh    boundary layer height    -> replaces AI-DUST's fixed H = 3000 m

  CLIMATOLOGY (12 monthly means)   fsr, lai_hv, lai_lv
      fsr    surface roughness        -> roughness partition of drag
      lai_*  leaf area index          -> vegetation gate on emission
      (seasonal cycle matters, interannual variation does not, for a drag/veg gate)

  STATIC    (single timestep)      z, lsm
      z      geopotential            -> orography, and AI-DUST's relative elevation
      lsm    land-sea mask           -> emission only over land

Everything is requested straight onto the 0.75 deg PM grid (25 x 31) so no
regridding is needed downstream.

Usage:
  python experiments/data_prep/download_era5_emission.py static      # z, lsm, fsr, lai  (minutes)
  python experiments/data_prep/download_era5_emission.py test        # one month of dynamic fields
  python experiments/data_prep/download_era5_emission.py 2003 2024   # full dynamic range, per-year
"""
import os
import sys
import glob
import shutil
import zipfile
import tempfile

import xarray as xr
import cdsapi

OUT_DIR = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/emission"

# Exactly the saudi_merged.nc footprint: lat 15.18..33.18, lon 33.80..56.30 @ 0.75 deg.
AREA = [33.18, 33.80, 15.18, 56.30]   # [North, West, South, East]
GRID = [0.75, 0.75]

DYNAMIC_VARS = [
    "friction_velocity",
    "volumetric_soil_water_layer_1",
    "boundary_layer_height",
]
CLIM_VARS = [
    "forecast_surface_roughness",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
]
STATIC_VARS = ["geopotential", "land_sea_mask"]

TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def _extract(zip_or_nc, out_nc):
    """CDS may return a bare .nc or a ZIP bundling instantaneous/accumulated streams."""
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
            ds.to_netcdf(out_nc, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
            ds.close()
        finally:
            os.remove(zip_or_nc)
            shutil.rmtree(tmp, ignore_errors=True)   # Windows: open handles -> ignore
    print(f"  -> {out_nc} ({os.path.getsize(out_nc)/1e6:.2f} MB)", flush=True)


def _retrieve(dataset, request, out_nc):
    if os.path.exists(out_nc):
        print(f"  skip (exists): {os.path.basename(out_nc)}", flush=True)
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    raw = out_nc + ".download"
    cdsapi.Client().retrieve(dataset, request, raw)
    _extract(raw, out_nc)


def _base(variables):
    return {"product_type": "reanalysis", "format": "netcdf",
            "variable": variables, "area": AREA, "grid": GRID}


def download_static():
    """Orography + land mask (one timestep) and the monthly roughness/LAI climatology."""
    print("Static fields (z, lsm) ...", flush=True)
    _retrieve("reanalysis-era5-single-levels",
              dict(_base(STATIC_VARS), year=["2010"], month=["01"], day=["01"], time=["00:00"]),
              os.path.join(OUT_DIR, "era5_static.nc"))

    print("Monthly climatology (fsr, lai_hv, lai_lv) ...", flush=True)
    _retrieve("reanalysis-era5-single-levels-monthly-means",
              dict(_base(CLIM_VARS), product_type="monthly_averaged_reanalysis",
                   year=[str(y) for y in range(2003, 2025)], month=MONTHS, time=["00:00"]),
              os.path.join(OUT_DIR, "era5_surface_monthly.nc"))


def download_dynamic(year, month=None):
    """One request per year (falls back to per-month if CDS rejects the cost)."""
    if month is not None:
        _retrieve("reanalysis-era5-single-levels",
                  dict(_base(DYNAMIC_VARS), year=[str(year)], month=[month], day=DAYS, time=TIMES),
                  os.path.join(OUT_DIR, f"era5_emis_{year}_{month}.nc"))
        return

    out_nc = os.path.join(OUT_DIR, f"era5_emis_{year}.nc")
    if os.path.exists(out_nc):
        print(f"  skip (exists): {os.path.basename(out_nc)}", flush=True)
        return
    print(f"Requesting dynamic {year} (full year) ...", flush=True)
    try:
        _retrieve("reanalysis-era5-single-levels",
                  dict(_base(DYNAMIC_VARS), year=[str(year)], month=MONTHS, day=DAYS, time=TIMES),
                  out_nc)
    except Exception as exc:                       # CDS cost limit -> chunk by month
        print(f"  year request failed ({type(exc).__name__}: {exc}); falling back to monthly",
              flush=True)
        for m in MONTHS:
            download_dynamic(year, m)


def main(argv):
    if not argv or argv[0] == "static":
        download_static()
    elif argv[0] == "test":
        download_dynamic(2020, "06")
    else:
        y0 = int(argv[0])
        y1 = int(argv[1]) if len(argv) > 1 else y0
        download_static()
        for year in range(y0, y1 + 1):
            download_dynamic(year)
    print("done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
