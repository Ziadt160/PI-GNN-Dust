"""Build a 1 km merged dataset for the Riyadh box (mirrors build_saudi_dataset.py).

The GENERIC machinery is implemented and ready: the 1 km target grid, bilinear
regridding, wind speed/direction -> u/v conversion, static valid mask, and the
compressed NetCDF writer. Only the four source-specific LOADERS are left as stubs
(`load_*`) because their exact file formats depend on what you download. Fill those
in (each just needs to return an xarray DataArray on its native grid with a
`valid_time` axis), then run this to produce `data/riyadh_merged.nc`.

See DATA.md for exactly which products to download.

Run (after filling the loaders):  python -u experiments/riyadh_1km/build_riyadh_dataset.py
"""
import os
import numpy as np
import xarray as xr
import warnings
warnings.filterwarnings("ignore")

# ---------------- Config: Riyadh 1 km box ----------------
LAT_MIN, LAT_MAX = 24.3, 25.1
LON_MIN, LON_MAX = 46.3, 47.1
RES_KM = 1.0
# Big datasets live in the main project data dir (alongside ALL DATA SAUDI).
RAW = r"D:/Quantum Projects/DustProject/data/ALL DATA RIYADH"
OUT = os.path.join(RAW, "riyadh_merged.nc")


def target_grid():
    """1 km lat/lon arrays for the box (latitude descending: row 0 = north)."""
    dlat = RES_KM / 111.0
    mean_lat = 0.5 * (LAT_MIN + LAT_MAX)
    dlon = RES_KM / (111.0 * np.cos(np.deg2rad(mean_lat)))
    lat = np.arange(LAT_MAX, LAT_MIN - 1e-9, -dlat)
    lon = np.arange(LON_MIN, LON_MAX + 1e-9, dlon)
    return lat.astype("float32"), lon.astype("float32")


def regrid(da, lat, lon):
    """Bilinear-interpolate a DataArray onto the 1 km target grid."""
    da = da.reset_coords(drop=True)
    out = da.interp(latitude=lat, longitude=lon, method="linear")
    return out.astype("float32")


def winddir_to_uv(speed, direction_deg):
    """Meteorological speed + 'from' direction -> (u east, v north). Validated convention."""
    theta = np.deg2rad(direction_deg)
    return (-speed * np.sin(theta)).astype("float32"), (-speed * np.cos(theta)).astype("float32")


# ---------------- Source loaders: FILL THESE IN ----------------
# Each must return an xarray DataArray with dims (valid_time, latitude, longitude)
# on the product's NATIVE grid. Regridding to 1 km happens afterwards.

def load_maiac_aod():
    """MODIS MAIAC `MCD19A2` 1 km AOD (tiles h22v05/h23v05), mosaicked + reprojected
    to lat/lon, stacked over time. Returns DataArray 'aod'. See DATA.md."""
    raise NotImplementedError("Load MAIAC AOD HDFs, mosaic tiles, reproject to lat/lon, "
                              "stack over time -> DataArray(valid_time, latitude, longitude).")


def load_pm_target():
    """Surface PM10/PM2.5 truth. Either (a) gridded analysis from AOD->PM calibration,
    or (b) point monitors rasterised/kriged to the grid. Returns Dataset[pm10, pm2p5]."""
    raise NotImplementedError("Load OpenAQ / PME monitor PM, calibrate vs MAIAC AOD, "
                              "produce gridded pm10/pm2p5 targets.")


def load_era5():
    """ERA5/IFS forcing on its native (~0.1-0.25 deg) grid: u10, v10, tp, blh, t2m, rh.
    Returns Dataset. If winds are speed/dir, convert with winddir_to_uv after regrid."""
    raise NotImplementedError("Load ERA5 single-levels for the box -> Dataset of forcing vars.")


def load_static():
    """Time-invariant 1 km covariates: elevation, land cover, NDVI, soil/erodibility,
    distance-to-source. Returns Dataset already near 1 km (regrid to match)."""
    raise NotImplementedError("Load static rasters -> Dataset of 2-D covariates.")


# ---------------- Assemble ----------------
def main():
    lat, lon = target_grid()
    print(f"Riyadh 1 km grid: {len(lat)} x {len(lon)} = {len(lat)*len(lon)} cells "
          f"(lat {lat.min():.2f}-{lat.max():.2f}, lon {lon.min():.2f}-{lon.max():.2f})", flush=True)

    pm = load_pm_target()                       # Dataset[pm10, pm2p5]
    aod = regrid(load_maiac_aod(), lat, lon)    # DataArray
    era = load_era5()
    static = load_static()

    t_axis = pm["valid_time"]
    vars_ = {
        "pm10": regrid(pm["pm10"], lat, lon),
        "pm2p5": regrid(pm["pm2p5"], lat, lon),
        "aod": aod,
        "u10": regrid(era["u10"], lat, lon),
        "v10": regrid(era["v10"], lat, lon),
        "tp": regrid(era["tp"], lat, lon),
        "t2m": regrid(era["t2m"], lat, lon),
        "rh": regrid(era["rh"], lat, lon),
    }
    for k in ["elevation", "landcover", "ndvi", "erodibility"]:
        if k in static:
            v = regrid(static[k], lat, lon)
            vars_[k] = v.broadcast_like(vars_["pm10"]) if "valid_time" not in v.dims else v

    for da in vars_.values():
        if "valid_time" in da.dims:
            da["valid_time"] = t_axis

    mask = np.ones((len(lat), len(lon)), bool)
    for da in vars_.values():
        sl = da.isel(valid_time=0).values if "valid_time" in da.dims else da.values
        mask &= ~np.isnan(sl)
    print(f"  valid cells (all vars present): {mask.sum()} / {mask.size}", flush=True)

    ds = xr.Dataset({**vars_, "valid_mask": (("latitude", "longitude"), mask)},
                    coords={"valid_time": t_axis, "latitude": lat, "longitude": lon})
    enc = {v: {"zlib": True, "complevel": 4} for v in vars_}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ds.to_netcdf(OUT, encoding=enc)
    print(f"  wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
