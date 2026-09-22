"""Observed dust intensity from station visibility -- a target that actually exists.

The problem this replaces
------------------------
Training against CAMS PM10 means training against a field that, measured over
234,512 matched station-hours, under-represents real dust storms by **17-56x** and
never once exceeded 1,000 ug/m^3 across 499 WMO-duststorm hours.  A model cannot
learn to predict a phenomenon its target does not contain.

Airport visibility does contain it, it is observed rather than modelled, and there
is a great deal of it: 22 Saudi stations, hourly, back to the 1970s.

From visibility to dust intensity
---------------------------------
Visibility is set by light extinction, and for a dust-dominated atmosphere the
extinction coefficient is close to linear in coarse-mode dust mass.  The
Koschmieder relation inverts it::

    beta_ext = 3.912 / V          V in metres, beta in 1/m

``beta_ext`` -- not visibility -- is therefore the physically meaningful
continuous target: it is (approximately) proportional to dust concentration,
whereas visibility is proportional to its reciprocal and saturates.

Censoring
---------
METAR visibility is reported up to a ceiling (commonly 9999 m / 10 km) and in
coarse increments at the top of the range, so clear-air values are **right-censored
in visibility**, i.e. left-censored in extinction.  Treating the ceiling as a real
measurement would teach the model that every calm hour has exactly the same dust
load.  `DustTarget` therefore exposes:

  * ``event``      binary labels at several thresholds -- unaffected by the ceiling,
                   because every threshold of interest sits well below it;
  * ``log_beta``   the continuous target, with a ``censored`` mask so a training
                   loss can skip or down-weight the saturated values.

Binary labels are the primary target: dust storms are a **rare-event detection**
problem (0.27% base rate at vis < 1 km), not a regression problem.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

METAR_DIR = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/metar"
MILES_TO_KM = 1.609344
KOSCHMIEDER = 3.912

# Reported-visibility ceiling: at or above this the observation is censored, not a
# measurement of clear air.
VIS_CEILING_KM = 9.5

# Event thresholds, in km of visibility.
EVENT_THRESHOLDS = {"dust_storm": 1.0, "severe": 0.4, "dusty": 5.0}

DUST_CODE_RE = "DS|SS|BLDU|BLSA|DU|SA|PO"

# Water-based obscurations: fog, mist, drizzle, rain, thunderstorms, hail, snow.
# Visibility below a threshold for one of THESE reasons is not a dust event, and
# measured over all 3.55M observations the contamination is severe at the tail:
#
#     label                hours    dust-coded   water-coded   haze   uncoded
#     dusty  (vis<5 km)   112,550     64.9%         10.7%      5.8%    18.9%
#     storm  (vis<1 km)    14,677     71.0%         17.4%      1.9%    10.4%
#     severe (vis<0.4 km)   2,768     51.7%       * 36.0% *    0.6%    12.6%
#
# More than a third of "severe dust storms" were fog and rain -- and `severe` is
# the grade that scores TS 0.000. Part of that is rarity; part of it is being
# asked to predict a mixture of two unrelated phenomena.
#
# The reverse error is as large: only 68.1% of dust-coded hours drop visibility
# below 5 km, so roughly a third of observer-confirmed dust is currently labelled
# "clear" and trains the model against itself.
WET_CODE_RE = "FG|BR|RA|DZ|TS|SH|GR|GS|SN|PL|IC"


def beta_ext_from_visibility(vis_km):
    """Extinction coefficient (1/km) from visibility (km) -- linear in dust mass.

    ``np.divide(..., where=)`` rather than ``np.where``: the latter evaluates both
    branches, so zero/missing visibilities raise a divide warning for every row
    even though the result is discarded.
    """
    v = np.asarray(vis_km, dtype="float64")
    out = np.full(v.shape, np.nan)
    np.divide(KOSCHMIEDER, v, out=out, where=(v > 0))
    return out


class DustTarget:
    """Station dust observations aligned onto the model's time axis."""

    def __init__(self, metar_dir=METAR_DIR):
        self.dir = metar_dir
        with open(os.path.join(metar_dir, "stations.json")) as fh:
            self.stations = json.load(fh)
        self._cache = {}

    # ------------------------------------------------------------------ #
    def raw(self, station_id):
        """Hourly visibility (km), extinction, dust-code flag, censoring mask."""
        if station_id in self._cache:
            return self._cache[station_id]
        path = os.path.join(self.dir, f"{station_id}.csv")
        if not os.path.exists(path):
            self._cache[station_id] = None
            return None

        d = pd.read_csv(path, na_values=["M", "T"])
        t = pd.to_datetime(d["valid"], utc=True, errors="coerce")
        vis = pd.to_numeric(d["vsby"], errors="coerce") * MILES_TO_KM
        wx = d["wxcodes"].fillna("").astype(str)
        code = wx.str.contains(DUST_CODE_RE, regex=True)
        # Water obscuration only counts against the observation when dust is NOT
        # also reported: a haboob ahead of a thunderstorm outflow carries both
        # codes and is a real dust event, so `TS` alone must not discard it.
        wet = wx.str.contains(WET_CODE_RE, regex=True) & ~code

        frame = pd.DataFrame({
            "vis_km": vis.values,
            "beta": beta_ext_from_visibility(vis.values),
            "dust_code": code.values,
            "wet_code": wet.values,
            "censored": (vis.values >= VIS_CEILING_KM),
        }, index=t)
        frame = frame[~frame.index.isna()].sort_index()
        frame = frame[~frame.index.duplicated(keep="first")]
        self._cache[station_id] = frame
        return frame

    # ------------------------------------------------------------------ #
    def align(self, times, tolerance="90min", dust_gate=False):
        """Observations on the model's time axis.

        Returns ``(labels, valid, extra)``:
          labels ``[T, S, n_thresholds]`` bool  -- event at each threshold
          valid  ``[T, S]`` bool                -- an observation was matched
          extra  dict with ``log_beta``, ``censored``, ``dust_code``, ``station_ids``
        """
        times = pd.DatetimeIndex(times)
        names = list(EVENT_THRESHOLDS)
        sids, labs, vals, lbeta, cens, codes, wets = [], [], [], [], [], [], []

        for s in self.stations:
            frame = self.raw(s["id"])
            if frame is None or frame.empty:
                continue
            a = frame.reindex(times, method="nearest",
                              tolerance=pd.Timedelta(tolerance))
            ok = a["vis_km"].notna().to_numpy()
            wet = a["wet_code"].astype("boolean").fillna(False).to_numpy(dtype=bool)
            if dust_gate:
                # Screen rather than relabel. A fog hour is not evidence that no
                # dust was present, so calling it a negative would be as wrong as
                # calling it an event; it carries no information either way.
                # Xu et al. 2026 do the same, discarding samples with RH >= 70%
                # or precipitation above 0.1 mm/h before training.
                ok = ok & ~wet
            sids.append(s["id"])
            vals.append(ok)
            wets.append(wet)
            labs.append(np.stack([(a["vis_km"] < EVENT_THRESHOLDS[n]).to_numpy()
                                  for n in names], axis=-1))
            with np.errstate(divide="ignore", invalid="ignore"):
                lbeta.append(np.log(a["beta"].to_numpy(dtype="float64")))
            cens.append(a["censored"].astype("boolean").fillna(True).to_numpy(dtype=bool))
            codes.append(a["dust_code"].astype("boolean").fillna(False).to_numpy(dtype=bool))

        if not sids:
            raise RuntimeError(f"no station observations found in {self.dir}")

        return (np.stack(labs, axis=1), np.stack(vals, axis=1),
                {"log_beta": np.stack(lbeta, axis=1).astype("float32"),
                 "censored": np.stack(cens, axis=1),
                 "dust_code": np.stack(codes, axis=1),
                 "wet_code": np.stack(wets, axis=1),
                 "dust_gate": bool(dust_gate),
                 "station_ids": sids,
                 "threshold_names": names})

    # ------------------------------------------------------------------ #
    def node_indices(self, mask):
        """Model node index for each station, and the stations that map into one."""
        n_cols = mask.shape[1]
        node_flat = np.flatnonzero(mask.ravel())
        lookup = {(int(f // n_cols), int(f % n_cols)): i for i, f in enumerate(node_flat)}
        out, kept = [], []
        for s in self.stations:
            key = (s["row"], s["col"])
            if key in lookup and self.raw(s["id"]) is not None:
                out.append(lookup[key])
                kept.append(s["id"])
        return np.asarray(out, dtype=int), kept

    # ------------------------------------------------------------------ #
    @staticmethod
    def summarise(labels, valid, names):
        """Base rates -- the numbers that dictate the loss weighting."""
        lines = []
        for k, n in enumerate(names):
            lab = labels[..., k][valid]
            lines.append(f"  {n:<12} {lab.sum():>8,} events / {lab.size:>10,} obs "
                         f"= {lab.mean()*100:6.3f}%   pos_weight {(1-lab.mean())/max(lab.mean(),1e-9):8.1f}")
        return "\n".join(lines)
