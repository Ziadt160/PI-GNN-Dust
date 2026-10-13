"""Download ERA5 surface fields for the original Gulf PM10 domain, one file per year.

Writes ``data/era5_<year>.nc`` (u10, v10, t2m; 3-hourly; 0.75 deg; 24-28.5 N,
47-55.25 E) and skips years already on disk, so an interrupted run resumes.
Needs a CDS API key in ``~/.cdsapirc``.

Run from the repository root:  python experiments/data_prep/download_chunked.py
"""
import os

import cdsapi

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(REPO_ROOT, "data")
YEARS = range(2003, 2025)


def main():
    c = cdsapi.Client()
    os.makedirs(OUT_DIR, exist_ok=True)
    for year in YEARS:
        fname = os.path.join(OUT_DIR, f"era5_{year}.nc")
        if os.path.exists(fname):
            print(f"Skipping {fname}, already exists.")
            continue

        print(f"Downloading {fname}...")
        try:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                    ],
                    'year': str(year),
                    'month': [f"{m:02d}" for m in range(1, 13)],
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': ['00:00', '03:00', '06:00', '09:00',
                             '12:00', '15:00', '18:00', '21:00'],
                    'area': [28.5, 47.0, 24.0, 55.25],
                    'grid': [0.75, 0.75],
                },
                fname)
            print(f"Downloaded {fname}")
        except Exception as e:
            print(f"Failed to download {year}: {e}")


if __name__ == "__main__":
    main()
