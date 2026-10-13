"""Build a clean, aligned, merged dataset from the raw 'ALL DATA SAUDI' files.

The raw files share a 3-hourly 2003-2024 time axis but live on three different,
misaligned grids and store wind as speed+direction. This script:
  1. regrids meteorology (wind, t2m, RH; 21x28 0.75 deg offset grid) and
     precipitation (63x84 0.25 deg) onto the PM target grid (25x31, 0.75 deg),
  2. converts wind speed+direction to u10/v10 components,
  3. computes a STATIC valid-cell mask = cells where every variable is present,
  4. writes a consolidated, compressed NetCDF the model pipeline can consume.

Output: data/ALL DATA SAUDI/saudi_merged.nc with variables
  [pm10, pm2p5, u10, v10, t2m, rh, tp] on (valid_time, latitude, longitude)
  plus a 2-D `valid_mask`.

Run from the repository root:  python experiments/data_prep/build_saudi_dataset.py
"""
import os
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

D = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI"
OUT = os.path.join(D, "saudi_merged.nc")


def open_var(fname, var, rename_time=True):
    ds = xr.open_dataset(os.path.join(D, fname))
    if rename_time and "time" in ds.coords and "valid_time" not in ds.coords:
        ds = ds.rename({"time": "valid_time"})
    return ds[var]


def main():
    print("Loading PM target grid ...")
    pm10 = open_var("pm10_masked.nc", "pm10", rename_time=False).astype("float32")
    pm2p5 = open_var("pm2.5_masked.nc", "pm2p5", rename_time=False).astype("float32")
    pm_lat, pm_lon = pm10.latitude, pm10.longitude
    t_axis = pm10.valid_time
    print(f"  target grid: {pm10.sizes['latitude']}x{pm10.sizes['longitude']}, "
          f"time n={t_axis.size}")

    def regrid(da):
        # drop non-dimension coords (e.g. spatial_ref) that confuse interp
        da = da.reset_coords(drop=True)
        out = da.interp(latitude=pm_lat, longitude=pm_lon, method="linear")
        return out.astype("float32")

    print("Regridding meteorology onto PM grid ...")
    ws = regrid(open_var("10m_wind_speed_clipped.nc", "wind_speed"))
    wd = regrid(open_var("10m_wind_direction_clipped.nc", "wind_direction"))
    t2m = regrid(open_var("2m_temperature_clipped.nc", "t2m"))
    rh = regrid(open_var("relative_humidity_clipped.nc", "relative_humidity"))
    print("Regridding precipitation (0.25 -> 0.75) ...")
    tp = regrid(open_var("era5_tp_2003_2024_clipped.nc", "total_precipitation"))

    # Wind speed + meteorological direction (deg FROM) -> u (east), v (north)
    theta = np.deg2rad(wd)
    u10 = (-ws * np.sin(theta)).astype("float32")
    v10 = (-ws * np.cos(theta)).astype("float32")

    # Align everything on the PM time axis (identical stamps, just re-label).
    for da in (u10, v10, t2m, rh, tp):
        da["valid_time"] = t_axis

    print("Computing static valid mask (all variables present) ...")
    sample = {"pm10": pm10, "pm2p5": pm2p5, "u10": u10, "v10": v10,
              "t2m": t2m, "rh": rh, "tp": tp}
    mask = np.ones((pm10.sizes["latitude"], pm10.sizes["longitude"]), bool)
    for name, da in sample.items():
        mask &= ~np.isnan(da.isel(valid_time=0).values)
    # confirm mask is static using a late timestep
    for name, da in sample.items():
        late = ~np.isnan(da.isel(valid_time=-1).values)
        assert np.array_equal(mask & late, mask), f"{name} mask not static!"
    print(f"  valid cells: {mask.sum()} / {mask.size}")

    ds = xr.Dataset(
        {**sample, "valid_mask": (("latitude", "longitude"), mask)},
        coords={"valid_time": t_axis, "latitude": pm_lat, "longitude": pm_lon},
    )

    enc = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in sample}
    print(f"Writing {OUT} ...")
    ds.to_netcdf(OUT, encoding=enc)

    # Validation report
    print("\n=== VALIDATION ===")
    for v in ["pm10", "pm2p5", "u10", "v10", "t2m", "rh", "tp"]:
        a = ds[v].values[:, mask]  # valid cells only
        print(f"  {v:<6} valid-cell mean={np.nanmean(a):.4e} "
              f"min={np.nanmin(a):.4e} max={np.nanmax(a):.4e} nan%={100*np.isnan(a).mean():.2f}")
    print(f"  saved size: {os.path.getsize(OUT)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
