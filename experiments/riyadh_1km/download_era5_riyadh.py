"""Download ERA5 single-level forcing for the Riyadh 1 km box (coarse met covariates).

ERA5 is the only piece downloadable here (CDS credentials are configured in
~/.cdsapirc). MAIAC AOD (Earthdata) and ground monitors (OpenAQ) need accounts/keys
that must be set up separately -- see DATA.md.

The new CDS returns NetCDF as a ZIP bundling instantaneous (u10,v10,t2m,d2m,blh)
and accumulated (tp) streams; this script auto-extracts, merges, computes relative
humidity from t2m+dewpoint, and writes one clean `era5_riyadh_<year>.nc` per year
(chunked by year to stay within CDS request limits).

Usage:
  python experiments/riyadh_1km/download_era5_riyadh.py test       # 1 month, validates creds
  python experiments/riyadh_1km/download_era5_riyadh.py 2015 2024  # full range, per-year files
"""
import os
import sys
import glob
import shutil
import zipfile
import tempfile

import numpy as np
import xarray as xr
import cdsapi

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import paths  # noqa: E402

OUT_DIR = paths.RIYADH_DIR
AREA = [25.1, 46.3, 24.3, 47.1]          # [North, West, South, East]
VARIABLES = ["10m_u_component_of_wind", "10m_v_component_of_wind",
             "2m_temperature", "2m_dewpoint_temperature",
             "total_precipitation", "boundary_layer_height"]
TIMES = [f"{h:02d}:00" for h in range(0, 24, 3)]
DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]


def _rh_from_dewpoint(t2m_K, d2m_K):
    """Relative humidity (%) from 2 m temperature and dewpoint (Magnus formula)."""
    def es(t_c):
        return 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
    rh = 100.0 * es(d2m_K - 273.15) / es(t2m_K - 273.15)
    return np.clip(rh, 0, 100).astype("float32")


def _postprocess(zip_path, out_nc):
    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        ncs = glob.glob(os.path.join(tmp, "*.nc"))
        parts = [xr.open_dataset(f) for f in ncs]
        ds = xr.merge(parts, compat="override").load()  # load fully so handles can close
        for p in parts:
            p.close()
        ds["rh"] = (ds["t2m"].dims, _rh_from_dewpoint(ds["t2m"].values, ds["d2m"].values))
        ds = ds.drop_vars("d2m")
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(out_nc, encoding=enc)
        ds.close()
    finally:
        os.remove(zip_path)
        shutil.rmtree(tmp, ignore_errors=True)  # Windows: open handles -> ignore
    print(f"  -> {out_nc} ({os.path.getsize(out_nc)/1e6:.2f} MB)", flush=True)


def retrieve(year, month, out_name):
    """One CDS request per (year, month) -- a full year exceeds the CDS cost limit."""
    os.makedirs(OUT_DIR, exist_ok=True)
    out_nc = os.path.join(OUT_DIR, out_name)
    if os.path.exists(out_nc):
        print(f"  skip (exists): {out_nc}", flush=True); return
    zip_path = out_nc + ".zip"
    print(f"Requesting ERA5 {year}-{month} ...", flush=True)
    cdsapi.Client().retrieve(
        "reanalysis-era5-single-levels",
        {"product_type": "reanalysis", "format": "netcdf", "variable": VARIABLES,
         "year": [str(year)], "month": [month], "day": DAYS, "time": TIMES,
         "area": AREA, "grid": [0.25, 0.25]},
        zip_path,
    )
    _postprocess(zip_path, out_nc)


def main():
    if len(sys.argv) <= 1 or sys.argv[1] == "test":
        retrieve(2024, "06", "era5_riyadh_2024_06.nc")
    else:
        y0, y1 = int(sys.argv[1]), int(sys.argv[2])
        for y in range(y0, y1 + 1):
            for m in ALL_MONTHS:
                retrieve(y, m, f"era5_riyadh_{y}_{m}.nc")


if __name__ == "__main__":
    main()
