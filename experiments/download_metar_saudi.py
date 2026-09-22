"""Download Saudi METAR observations -- the first real ground truth in this project.

Why METAR and not PM monitors
-----------------------------
Everything scored so far used CAMS reanalysis as truth, which is a model output:
DeepCAMS measured CAMS at R ~ 0.59 against in-situ PM, so our numbers describe how
well we reproduce CAMS, not how well we forecast dust.  OpenAQ looked like the fix
but has only 2 usable PM10 stations inside this domain (0.8% of cells, ~2 months).

Airport METAR is the dataset operational dust forecasting actually verifies
against, and for this domain it is far richer:

    OpenAQ PM10   2 stations    0.8% of cells   ~2 months     API key
    METAR        23 stations    7.2% of cells   full 2021-24  no auth

Each report carries horizontal visibility and present-weather codes, hourly, many
stations back to the 1970s.  A dust storm is defined by visibility, not by PM:

    WMO duststorm        visibility < 1 km
    NWS warning level    visibility <= 0.4 km

Present-weather codes (DS, SS, BLDU, BLSA, DU, SA) are downloaded too but treated
as secondary: a published audit of dust records found ~50% of coded DS events had
visibility above 1 km, so the visibility number is the reliable signal and the
code is corroboration.

Source: Iowa Environmental Mesonet ASOS archive (free, unauthenticated).

Usage
-----
  python experiments/download_metar_saudi.py --list
  python experiments/download_metar_saudi.py --start 2021-01-01 --end 2024-12-31
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

OUT_DIR = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/metar"
MERGED = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/saudi_merged.nc"

# The wide domain (12-38 N, 30-60 E) spans twelve national networks. Restricting
# to Saudi left the model with 26 stations and, worse, no observations at all in
# the Mesopotamian source region that supplies 42% of its dust storms.
IEM_NETWORKS = ["SA__ASOS", "IQ__ASOS", "KW__ASOS", "QA__ASOS", "BH__ASOS",
                "AE__ASOS", "OM__ASOS", "JO__ASOS", "SY__ASOS", "YE__ASOS",
                "IR__ASOS", "EG__ASOS"]
GEOJSON = "https://mesonet.agron.iastate.edu/geojson/network/{net}.geojson"
ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"

# Visibility thresholds (km) for a dust event.
WMO_DUSTSTORM_KM = 1.0
NWS_WARNING_KM = 0.4
DUST_CODES = ("DS", "SS", "BLDU", "BLSA", "DU", "SA", "PO")

RETRIES, BACKOFF, TIMEOUT = 4, 5, 300
UA = {"User-Agent": "DustProject-research/1.0"}


def _get(url, timeout=TIMEOUT):
    for attempt in range(RETRIES):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            if attempt == RETRIES - 1:
                raise
            time.sleep(BACKOFF * (attempt + 1))


def stations_in_domain(start, end, bbox=None):
    """Stations active over the period, inside the mask (or inside `bbox` if given).

    `bbox` is (south, north, west, east); pass it for the wide domain, where there
    is no land mask yet.
    """
    import xarray as xr
    mask = lat = lon = None
    if bbox is None:
        ds = xr.open_dataset(MERGED)
        mask = ds.valid_mask.values.astype(bool)
        lat, lon = ds.latitude.values, ds.longitude.values
        ds.close()

    feats = []
    for net in IEM_NETWORKS:
        try:
            feats += json.loads(_get(GEOJSON.format(net=net)))["features"]
        except Exception as exc:
            print(f"  network {net} unavailable: {type(exc).__name__}", flush=True)
    keep = []
    for f in feats:
        lon_s, lat_s = f["geometry"]["coordinates"]
        p = f["properties"]
        if bbox is not None:
            s_, n_, w_, e_ = bbox
            if not (s_ <= lat_s <= n_ and w_ <= lon_s <= e_):
                continue
            i = j = -1                     # grid indices assigned when the grid exists
        else:
            i = int(np.argmin(np.abs(lat - lat_s)))
            j = int(np.argmin(np.abs(lon - lon_s)))
            if not mask[i, j]:
                continue                   # outside the modelled land mask
        begin = str(p.get("archive_begin") or "9999")
        finish = p.get("archive_end")
        if begin > end or (finish and str(finish) < start):
            continue                       # no overlap with the evaluation period
        keep.append({"id": f["id"], "name": p.get("sname", ""), "lat": lat_s,
                     "lon": lon_s, "row": i, "col": j,
                     "begin": begin, "end": str(finish) if finish else "active"})
    return keep


def fetch_station(sid, start, end, out_dir):
    """One station's hourly visibility + present weather over the period."""
    dest = os.path.join(out_dir, f"{sid}.csv")
    if os.path.exists(dest) and os.path.getsize(dest) > 200:
        return 0
    y1, m1, d1 = start.split("-")
    y2, m2, d2 = end.split("-")
    q = urllib.parse.urlencode({
        "station": sid, "data": ["vsby", "wxcodes", "drct", "sknt"],
        "year1": y1, "month1": int(m1), "day1": int(d1),
        "year2": y2, "month2": int(m2), "day2": int(d2),
        "tz": "UTC", "format": "onlycomma", "missing": "M", "trace": "T",
        "latlon": "no", "report_type": [3, 4],
    }, doseq=True)
    txt = _get(f"{ASOS}?{q}")
    n = txt.count("\n") - 1
    if n < 1:
        return -1
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write(txt)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--list", action="store_true", help="list stations and exit")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("S", "N", "W", "E"),
                    help="use a bounding box instead of the Saudi land mask "
                         "(the wide domain is 12 38 30 60)")
    args = ap.parse_args()

    sts = stations_in_domain(args.start, args.end, tuple(args.bbox) if args.bbox else None)
    print(f"{len(sts)} METAR stations inside the valid mask, {args.start}..{args.end}",
          flush=True)
    for s in sts:
        print(f"  {s['id']:<6} {s['name'][:28]:<28} {s['lat']:6.2f},{s['lon']:6.2f} "
              f"cell({s['row']:2d},{s['col']:2d})  {s['begin']} -> {s['end']}", flush=True)
    if args.list:
        return

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "stations.json"), "w") as fh:
        json.dump(sts, fh, indent=2)

    total, failed = 0, []
    t0 = time.time()
    for k, s in enumerate(sts, 1):
        try:
            n = fetch_station(s["id"], args.start, args.end, args.out)
        except Exception as exc:
            print(f"  [{k}/{len(sts)}] {s['id']} FAILED: {type(exc).__name__}: {exc}",
                  flush=True)
            failed.append(s["id"])
            continue
        if n < 0:
            failed.append(s["id"])
            print(f"  [{k}/{len(sts)}] {s['id']}: no rows", flush=True)
        else:
            total += n
            print(f"  [{k}/{len(sts)}] {s['id']}: {n:,} obs "
                  f"({'cached' if n == 0 else 'downloaded'})", flush=True)

    print(f"\ndone: {total:,} observations, {len(failed)} failed "
          f"({', '.join(failed) if failed else 'none'}) in {(time.time()-t0)/60:.1f} min")
    print(f"stations -> {os.path.join(args.out, 'stations.json')}")


if __name__ == "__main__":
    main()
