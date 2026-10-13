"""Derive a dust-source map from satellite observations, since SoilGrids cannot.

Why not SoilGrids
-----------------
SoilGrids was the obvious source and was tested: across 20 random Saudi cells it
returned a median 49% sand, a total spread of 22 percentage points, and **not one
cell above 80% sand** -- in a country containing the world's largest continuous
sand sea.  It is trained on agricultural soil profiles, and hyper-arid dune
fields are effectively absent from that training set, so it regresses to a global
mean.  Using it would replace an honest constant with a field that merely looks
spatial.

The observational definition
----------------------------
A dust source is a place where **wind reliably produces airborne dust**.  That is
directly measurable without any soil property:

    erodibility(cell) = P( high aerosol optical depth | wind above threshold )

Cells whose AOD reliably jumps when the wind blows are sources; cells where the
same wind produces nothing are not, whatever their soil map says.  MAIAC AOD is
an independent satellite retrieval (it knows nothing about CAMS), gridded at 1 km
and aggregated here to the model grid.  ERA5 friction velocity supplies the wind
condition.

The conditional form matters: a raw "mean AOD" map would mostly show where it is
*windy*, not where the surface can supply dust.  Conditioning on wind separates
the surface's contribution from the meteorology's.

Caveats recorded in the output
------------------------------
* MAIAC is daytime-only with cloud gaps; the sample is 122 days over 2021-2024,
  thin for a climatology, so per-cell counts are written alongside the values and
  cells below `--min-samples` are left as NaN rather than guessed.
* AOD is a column property; a high-altitude plume passing overhead inflates it
  without local emission.  Conditioning on *local* wind mitigates this but does
  not remove it.
* MODIS overpass is late morning / early afternoon, which is not when shamal dust
  peaks, so the map is biased toward daytime-emitting sources.

Run:  python -u experiments/data_prep/build_erodibility_map.py
"""
import argparse
import glob
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src import paths  # noqa: E402

MAIAC_DIR = os.path.join(paths.MAIAC_DIR, "validation")
MERGED = paths.SAUDI_MERGED
EMIS_DIR = paths.EMISSION_DIR
OUT = paths.ERODIBILITY

# MODIS sinusoidal projection constants.
R_EARTH = 6371007.181
TILE_M = 1111950.5196666666
ORIGIN_X, ORIGIN_Y = -20015109.354, 10007554.677



def tile_latlon(h, v, n=1200):
    """Centre latitude/longitude of every pixel in one MODIS sinusoidal tile."""
    px = TILE_M / n
    j, i = np.meshgrid(np.arange(n), np.arange(n))
    x = ORIGIN_X + h * TILE_M + (j + 0.5) * px
    y = ORIGIN_Y - v * TILE_M - (i + 0.5) * px
    lat = np.degrees(y / R_EARTH)
    lon = np.degrees(x / (R_EARTH * np.cos(y / R_EARTH)))
    return lat, lon


def read_granule(path):
    """Orbit-composited AOD 0.55 um, plus the tile's lat/lon.

    netCDF4 applies ``scale_factor`` and the fill mask itself, so the values come
    back already in AOD units.  Multiplying by the scale factor again -- which an
    earlier version of this script did -- makes every AOD 1000x too small and
    silently produces an all-zero erodibility map.
    """
    import netCDF4
    m = re.search(r"\.h(\d\d)v(\d\d)\.", os.path.basename(path))
    if not m:
        return None
    h, v = int(m.group(1)), int(m.group(2))
    try:
        with netCDF4.Dataset(path) as d:
            if "Optical_Depth_055" not in d.variables:
                return None
            a = d.variables["Optical_Depth_055"][:]      # already scaled & masked
    except Exception:
        return None
    a = np.ma.masked_invalid(np.ma.masked_less(a, 0.0)).astype("float32")
    if a.ndim == 3:
        a = np.ma.mean(a, axis=0)          # composite the day's orbits
    lat, lon = tile_latlon(h, v, a.shape[0])
    return lat, lon, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aod-hi", type=float, default=0.35,
                    help="cell-mean AOD above this counts as a dust-laden column. 0.35 sits well above the regional background (median cell AOD ~0.13) while staying common enough to estimate per cell")
    ap.add_argument("--min-samples", type=int, default=12,
                    help="cells with fewer windy observations stay NaN")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    ds = xr.open_dataset(MERGED)
    mask = ds.valid_mask.values.astype(bool)
    lat_g, lon_g = ds.latitude.values, ds.longitude.values
    tv = pd.to_datetime(ds.valid_time.values, utc=True)
    ds.close()
    nlat, nlon = len(lat_g), len(lon_g)
    dlat = float(lat_g[1] - lat_g[0])          # negative: descending
    dlon = float(lon_g[1] - lon_g[0])

    # ERA5 friction velocity and its moisture-corrected threshold.
    from src.physics import emission as E
    parts = [xr.open_dataset(f) for f in sorted(glob.glob(os.path.join(EMIS_DIR, "era5_emis_*.nc")))]
    ed = xr.concat(parts, dim="valid_time").sortby("valid_time")
    n = min(len(ed.valid_time), len(tv))
    zust = np.nan_to_num(ed.zust.values[:n])
    ustar_t = E.threshold_friction_velocity(swvl1=np.nan_to_num(ed.swvl1.values[:n]))
    for p in parts:
        p.close()
    windy_all = zust > ustar_t
    print(f"ERA5 friction velocity: {n:,} steps | windy fraction "
          f"{windy_all.mean()*100:.1f}%", flush=True)

    files = sorted(glob.glob(os.path.join(MAIAC_DIR, "*.hdf")))
    print(f"MAIAC granules: {len(files):,}", flush=True)

    # Accumulators on the model grid.
    n_windy = np.zeros((nlat, nlon), "int32")     # windy observations seen
    n_dusty = np.zeros((nlat, nlon), "int32")     # ...of which AOD was high
    n_obs = np.zeros((nlat, nlon), "int32")       # all valid observations
    aod_sum = np.zeros((nlat, nlon), "float64")

    t0 = time.time()
    for k, path in enumerate(files, 1):
        got = read_granule(path)
        if got is None:
            continue
        lat_p, lon_p, aod = got
        valid = ~np.ma.getmaskarray(aod)
        if not valid.any():
            continue

        # Bin 1 km pixels onto the 0.75 deg grid.
        ii = np.floor((lat_p[valid] - lat_g[0]) / dlat).astype(int)
        jj = np.floor((lon_p[valid] - lon_g[0]) / dlon).astype(int)
        vals = np.asarray(aod[valid])
        inside = (ii >= 0) & (ii < nlat) & (jj >= 0) & (jj < nlon)
        if not inside.any():
            continue
        ii, jj, vals = ii[inside], jj[inside], vals[inside]

        # Nearest model timestep to this granule's day (MODIS overpass ~10:30 local).
        m = re.search(r"\.A(\d{4})(\d{3})\.", os.path.basename(path))
        day = pd.Timestamp(f"{m.group(1)}-01-01", tz="UTC") + pd.Timedelta(days=int(m.group(2)) - 1)
        t_idx = int(np.argmin(np.abs(tv[:n] - (day + pd.Timedelta(hours=9)))))

        # Reduce this granule's 1 km pixels to ONE value per model cell first.
        # Accumulating per pixel counts ~740,000 "observations" per cell and turns
        # the conditional probability into a pixel-level exceedance fraction,
        # which is not the quantity we want.
        flat = ii * nlon + jj
        cell_sum = np.bincount(flat, weights=vals, minlength=nlat * nlon)
        cell_cnt = np.bincount(flat, minlength=nlat * nlon)
        seen = cell_cnt > 0
        cell_aod = np.full(nlat * nlon, np.nan)
        cell_aod[seen] = cell_sum[seen] / cell_cnt[seen]
        cell_aod = cell_aod.reshape(nlat, nlon)
        seen = seen.reshape(nlat, nlon)

        n_obs[seen] += 1
        aod_sum[seen] += cell_aod[seen]
        w = seen & windy_all[t_idx]
        n_windy[w] += 1
        n_dusty[w & (cell_aod > args.aod_hi)] += 1

        if k % 200 == 0 or k == len(files):
            print(f"  [{k}/{len(files)}] {(time.time()-t0)/60:.1f} min", flush=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        erod = np.where(n_windy >= args.min_samples, n_dusty / np.maximum(n_windy, 1), np.nan)
        mean_aod = np.where(n_obs > 0, aod_sum / np.maximum(n_obs, 1), np.nan)

    out = xr.Dataset(
        {"erodibility": (("latitude", "longitude"), erod.astype("float32")),
         "mean_aod": (("latitude", "longitude"), mean_aod.astype("float32")),
         "n_windy": (("latitude", "longitude"), n_windy),
         "n_obs": (("latitude", "longitude"), n_obs)},
        coords={"latitude": lat_g, "longitude": lon_g},
        attrs={
            "definition": "P(MAIAC AOD 550nm > %.2f | ERA5 u* > moisture-corrected "
                          "threshold), per grid cell" % args.aod_hi,
            "why": "SoilGrids has no discriminating signal over Arabian sand seas "
                   "(0/20 sampled cells above 80%% sand); this is the observational "
                   "substitute",
            "caveats": "MAIAC is daytime-only with cloud gaps; 122 sampled days "
                       "2021-2024; AOD is a column property so overhead plumes can "
                       "inflate it; MODIS overpass is not when shamal dust peaks",
            "min_samples": args.min_samples,
            "built": time.strftime("%Y-%m-%d"),
        })
    for v in out.data_vars:
        out[v].encoding.update(zlib=True, complevel=4)
    out.to_netcdf(args.out)

    land = mask & np.isfinite(erod)
    print(f"\nwrote {args.out}")
    print(f"  cells with a usable estimate: {land.sum()} / {mask.sum()} valid land cells")
    if land.any():
        e = erod[land]
        print(f"  erodibility  min {e.min():.3f}  median {np.median(e):.3f}  max {e.max():.3f}")
        print(f"  spread (max-min) {e.max()-e.min():.3f}  "
              f"-- compare SoilGrids sand, which spanned only 0.22 across the country")
        flat = np.argsort(-np.where(land, erod, -1), axis=None)[:5]
        print("  strongest sources:")
        for f in flat:
            i, j = np.unravel_index(f, erod.shape)
            print(f"    {lat_g[i]:5.2f} N {lon_g[j]:5.2f} E   erodibility {erod[i,j]:.3f} "
                  f"(n_windy {n_windy[i,j]})")


if __name__ == "__main__":
    main()
