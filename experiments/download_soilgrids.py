"""Fetch SoilGrids soil texture -- the erodibility map the emission scheme lacks.

Why
---
The Shao emission scheme currently uses a **single global constant** for clay
content (10%) and one threshold friction velocity for the whole country.  Dust
lifts from specific surfaces: a sand sea emits, a clay pan does not, and a gravel
desert barely does.  Without texture the scheme cannot say *where* a storm starts,
which is the first thing a dust forecast has to get right.

AI-DUST drives its threshold friction velocity from SoilGrids texture (Hengl et
al.) plus grain-size distributions; this is the same source, sampled at our grid.

How the values are used (see src/physics/emission.py)
-----------------------------------------------------
* **clay %** sets the Fecan residual soil moisture ``w' = 0.0014 clay^2 + 0.17 clay``
  below which moisture does not resist saltation -- currently a constant.
* **sand %** scales an erodibility factor: sandy surfaces supply saltators, clay-
  and silt-dominated surfaces are cohesive and supply few.

Sampling
--------
SoilGrids is a 250 m product; we need a 0.75 deg cell value, so each cell is
sampled at its centre via the ISRIC point API rather than downloading rasters.
That is one small request per cell (a few hundred), well inside fair use, and it
avoids pulling gigabytes to then average it away.  A cell centre is a point
estimate of a ~6,900 km^2 cell -- crude, but far better than one constant for the
whole country, and the limitation is recorded in the output file.

Run:  python -u experiments/download_soilgrids.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import xarray as xr

MERGED = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/saudi_merged.nc"
OUT = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/soilgrids_texture.nc"
API = "https://rest.isric.org/soilgrids/v2.0/properties/query"

PROPS = ("clay", "sand", "silt")
DEPTH = "0-5cm"
PAUSE = 0.35          # ISRIC fair use: stay well under a few requests/second
RETRIES = 3


def query_point(lat, lon):
    """Mean clay/sand/silt (%) in the top 5 cm at one point, or None."""
    q = [("lon", f"{lon:.4f}"), ("lat", f"{lat:.4f}"),
         ("depth", DEPTH), ("value", "mean")]
    q += [("property", p) for p in PROPS]
    url = f"{API}?{urllib.parse.urlencode(q)}"
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json",
                                                       "User-Agent": "DustProject/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                js = json.load(r)
            out = {}
            for layer in js.get("properties", {}).get("layers", []):
                name = layer["name"]
                if name not in PROPS:
                    continue
                # SoilGrids returns integers scaled by d_factor; dividing by it
                # yields the layer's target unit, which for texture is percent
                # (mapped units g/kg, target %, d_factor 10 -> raw 250 = 25%).
                d = layer.get("unit_measure", {}).get("d_factor", 10) or 10
                vals = [dp["values"]["mean"] for dp in layer["depths"]
                        if dp["values"].get("mean") is not None]
                if vals:
                    out[name] = float(vals[0]) / d
            return out or None
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < RETRIES - 1:
                time.sleep(3 * (attempt + 1))
                continue
            return None
        except Exception:
            if attempt < RETRIES - 1:
                time.sleep(2 * (attempt + 1))
                continue
            return None
    return None


def main():
    ds = xr.open_dataset(MERGED)
    mask = ds.valid_mask.values.astype(bool)
    lat, lon = ds.latitude.values, ds.longitude.values
    ds.close()

    shape = mask.shape
    clay = np.full(shape, np.nan, "float32")
    sand = np.full(shape, np.nan, "float32")
    silt = np.full(shape, np.nan, "float32")

    cells = [(i, j) for i in range(shape[0]) for j in range(shape[1]) if mask[i, j]]
    print(f"sampling SoilGrids at {len(cells)} valid cell centres "
          f"({DEPTH}, mean clay/sand/silt)", flush=True)

    got, miss, t0 = 0, 0, time.time()
    for n, (i, j) in enumerate(cells, 1):
        res = query_point(float(lat[i]), float(lon[j]))
        if res:
            clay[i, j] = res.get("clay", np.nan)
            sand[i, j] = res.get("sand", np.nan)
            silt[i, j] = res.get("silt", np.nan)
            got += 1
        else:
            miss += 1
        if n % 25 == 0 or n == len(cells):
            print(f"  [{n}/{len(cells)}] ok {got} missing {miss} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        time.sleep(PAUSE)

    out = xr.Dataset(
        {"clay": (("latitude", "longitude"), clay),
         "sand": (("latitude", "longitude"), sand),
         "silt": (("latitude", "longitude"), silt)},
        coords={"latitude": lat, "longitude": lon},
        attrs={
            "source": "ISRIC SoilGrids v2.0 point API, depth 0-5 cm, mean",
            "units": "percent by mass",
            "sampling": "one point query at each 0.75 deg cell centre -- a point "
                        "estimate standing for a ~6,900 km2 cell",
            "use": "clay drives the Fecan residual soil moisture; sand scales the "
                   "erodibility factor in src/physics/emission.py",
            "retrieved": time.strftime("%Y-%m-%d"),
        })
    for v in out.data_vars:
        out[v].encoding.update(zlib=True, complevel=4)
    out.to_netcdf(OUT)
    valid_frac = np.isfinite(clay[mask]).mean()
    print(f"\nwrote {OUT}")
    print(f"  coverage {valid_frac*100:.1f}% of valid cells")
    if np.isfinite(clay[mask]).any():
        print(f"  clay  {np.nanmin(clay[mask]):5.1f} - {np.nanmax(clay[mask]):5.1f} %"
              f"  (mean {np.nanmean(clay[mask]):.1f})")
        print(f"  sand  {np.nanmin(sand[mask]):5.1f} - {np.nanmax(sand[mask]):5.1f} %"
              f"  (mean {np.nanmean(sand[mask]):.1f})")


if __name__ == "__main__":
    main()
