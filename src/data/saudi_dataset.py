"""Shared data layer for the AI-DUST benchmark -- one preprocessing pass, 26 models.

`SaudiFields` loads `saudi_merged.nc` (plus the ERA5 emission drivers when they
have been downloaded), runs the emission scheme and the transport terms once, and
caches everything at **node level** (251 valid land cells) rather than grid level.
That matters: at the native 3 h cadence the domain is 64,288 timesteps, so a
gridded float32 field is ~200 MB while the node-level equivalent is ~64 MB.  The
grid is only materialised inside a chunked loop, for the spatial derivatives that
genuinely need it.

Two topologies are served from the same cached fields:

``full``   your layout -- one graph per timestep over all 251 cells, every cell
           predicted in a single pass.
``patch``  AI-DUST's layout -- a P x P patch per (timestep, cell), only the centre
           node predicted, zero-padded at the domain edge.

The patch gather map is precomputed once as an index array, so patch extraction
is a fancy-index into the node array rather than a per-sample grid crop.

Feature switches map one-to-one onto the model roster (``+courant``, ``+relief``,
``+emission``, ``+futuremet``, ``+divergence``, ``+history``), so a model variant
is a config change, not a new data path.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import xarray as xr
from torch_geometric.data import Data

from src.physics import emission as emis
from src.physics import stability as stab
from src.physics import transport as trans
from src import paths
from src import paths

MERGED_DEFAULT = paths.SAUDI_MERGED
EMISSION_DIR_DEFAULT = paths.EMISSION_DIR

CADENCE_FACTOR = {"3h": 1, "6h": 2, "12h": 4}
SECONDS_PER_STEP = {"3h": 10800.0, "6h": 21600.0, "12h": 43200.0}

# Dust emission is coarse-mode; only a small fine fraction reaches PM2.5.
# Exposed because it is an assumption, not a measurement.
PM25_EMISSION_FRACTION = 0.12

CHUNK_STEPS = 2000          # time-chunk for the gridded derivative pass

# e-folding time of the soil-moisture memory, days. Desert soil dries in a
# few days; this is the horizon over which past rain still suppresses dust.
WETNESS_TAU_DAYS = 5.0


@dataclass
class DataConfig:
    """Everything that distinguishes one model's data view from another's."""
    merged_path: str = MERGED_DEFAULT
    emission_dir: str = EMISSION_DIR_DEFAULT
    cadence: str = "3h"
    window: int = 3                  # temporal history frames (1 = none)
    future_met: bool = True          # met at t+dt (AI-DUST's AIFS forcing role)
    courant: bool = False            # u/dx, v/dy normalisation
    relief: bool = False             # relative elevation
    emission_features: bool = False  # u*, E as input channels
    transport_feature: bool = False  # expose the transport tendency as a channel
    divergence: bool = True          # div(VC) vs u.grad C in the transport term
    reference_mode: str = "euler"    # "euler" (AI-DUST, faithful) | "semilag" (CFL-stable)
    drag_partition: bool = False     # see src.physics.emission.compute_emission
    erodibility_path: str = paths.ERODIBILITY
    # Boundary-layer stability from ERA5 pressure levels (see src.physics.stability).
    # Off by default so every result predating the levels download stays reproducible;
    # `experiments/station_dust/ablation_levels.py` is what turns it on and measures the difference.
    pressure_levels: bool = False
    levels_dir: str = paths.LEVELS_DIR
    # Minimum share of the record the dynamic ERA5 drivers must cover before they
    # are used at all. A partial download is ignored rather than edge-padded,
    # which would silently present one year's values as two decades of drivers.
    min_driver_coverage: float = 0.98
    # Extra physical channels: hour of day, mixing depth, threshold excess,
    # soil-moisture memory. Off by default so every run predating them stays
    # reproducible and a comparison in flight cannot silently change features.
    physics_features: bool = False
    multitask: bool = True           # predict PM10 + PM2.5 (vs PM10 only)
    topology: str = "full"           # "full" | "patch"
    patch: int = 7
    neighbors: int = 8               # 8 = yours (diagonals), 4 = AI-DUST
    train_years: tuple = (2003, 2018)
    val_years: tuple = (2019, 2020)
    test_years: tuple = (2021, 2024)


class SaudiFields:
    """Loads, derives and caches every field the benchmark needs, once."""

    def __init__(self, cfg: DataConfig, verbose=True):
        self.cfg = cfg
        self.verbose = verbose
        self._log(f"loading {os.path.basename(cfg.merged_path)} (cadence {cfg.cadence})")

        ds = xr.open_dataset(cfg.merged_path)
        self.mask = ds.valid_mask.values.astype(bool)
        self.lats = ds.latitude.values
        self.lons = ds.longitude.values
        self.grid_shape = self.mask.shape
        self.node_flat = np.flatnonzero(self.mask.ravel())
        self.n_nodes = len(self.node_flat)
        self.dt = SECONDS_PER_STEP[cfg.cadence]
        self.dx, self.dy = trans.grid_spacing(self.lats, self.lons)

        factor = CADENCE_FACTOR[cfg.cadence]
        tv_full = ds.valid_time.values
        n_trim = (len(tv_full) // factor) * factor
        self.times = tv_full[:n_trim:factor]
        self.n_steps = len(self.times)

        raw = {v: self._block_mean(ds[v].values, n_trim, factor) for v in
               ("pm10", "pm2p5", "u10", "v10", "t2m", "rh", "tp")}
        ds.close()

        self._log(f"  grid {self.grid_shape} | nodes {self.n_nodes} | steps {self.n_steps}")

        self.drivers_meta = {}
        self._derive(raw)

    # ------------------------------------------------------------------ #
    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    @staticmethod
    def _block_mean(arr, n_trim, factor):
        """Block-average the time axis. NumPy reshape-mean: xarray.resample hangs here."""
        a = np.nan_to_num(arr[:n_trim]).astype("float32")
        if factor == 1:
            return a
        return a.reshape(-1, factor, *a.shape[1:]).mean(axis=1).astype("float32")

    def _gather(self, grid_field):
        """``[T, H, W]`` -> ``[T, n_nodes]`` over the valid land cells."""
        return grid_field.reshape(grid_field.shape[0], -1)[:, self.node_flat]

    # ------------------------------------------------------------------ #
    def _load_emission_drivers(self, speed_grid, tp_grid):
        """ERA5 drivers when downloaded, documented proxies otherwise."""
        static_p = os.path.join(self.cfg.emission_dir, "era5_static.nc")
        monthly_p = os.path.join(self.cfg.emission_dir, "era5_surface_monthly.nc")
        dyn = sorted(f for f in (os.listdir(self.cfg.emission_dir)
                                 if os.path.isdir(self.cfg.emission_dir) else [])
                     if f.startswith("era5_emis_") and f.endswith(".nc"))

        drivers = emis.EmissionDrivers(speed=speed_grid, tp=tp_grid)
        self.elevation = None
        self.land = None

        if os.path.exists(static_p):
            st = xr.open_dataset(static_p)
            self.elevation = (st["z"].values[0] / emis.GRAVITY).astype("float32")  # geopot -> m
            self.land = st["lsm"].values[0].astype("float32")
            drivers.land = self.land
            st.close()

        if os.path.exists(monthly_p):
            mo = xr.open_dataset(monthly_p)
            months = mo.valid_time.values.astype("datetime64[M]").astype(int) % 12
            step_month = self.times.astype("datetime64[M]").astype(int) % 12
            # Map each timestep onto the matching calendar-month mean.
            fsr = np.stack([mo["fsr"].values[months == m].mean(0) for m in range(12)])
            lai = np.stack([(mo["lai_hv"].values[months == m].mean(0)
                             + mo["lai_lv"].values[months == m].mean(0)) for m in range(12)])
            drivers.z0 = fsr[step_month]
            drivers.lai = lai[step_month]
            mo.close()

        self.driver_coverage = 0.0
        # Observational dust-source map (built by experiments/data_prep/build_erodibility_map.py).
        if self.cfg.erodibility_path and os.path.exists(self.cfg.erodibility_path):
            er = xr.open_dataset(self.cfg.erodibility_path)
            drivers.erodibility = er["erodibility"].values.astype("float32")
            er.close()

        if dyn:
            parts = [xr.open_dataset(os.path.join(self.cfg.emission_dir, f)) for f in dyn]
            ed = xr.concat(parts, dim="valid_time").sortby("valid_time")
            factor = CADENCE_FACTOR[self.cfg.cadence]
            n_trim = ((min(len(ed.valid_time), self.n_steps * factor)) // factor) * factor
            self.driver_coverage = n_trim / factor / self.n_steps

            # A partial download must NOT be padded into the remaining years: edge
            # padding would present one year's values as two decades of "ERA5
            # drivers", which is worse than honestly falling back to the proxy.
            if self.driver_coverage >= self.cfg.min_driver_coverage:
                for name in ("zust", "swvl1", "blh"):
                    if name in ed:
                        arr = self._block_mean(ed[name].values, n_trim, factor)
                        setattr(drivers, name, arr[:self.n_steps])
            else:
                self._log(
                    f"  NOTE: dynamic ERA5 drivers cover only "
                    f"{self.driver_coverage*100:.1f}% of the record "
                    f"({len(dyn)} year file(s)); below the "
                    f"{self.cfg.min_driver_coverage*100:.0f}% threshold, so they are "
                    f"IGNORED and the documented proxies are used instead. "
                    f"Re-run once the download completes.")
            for p in parts:
                p.close()
            ed.close()

        return drivers

    # ------------------------------------------------------------------ #
    def _load_levels(self):
        """Stability diagnostics from ERA5 pressure levels, or nothing.

        Same contract as the emission drivers: a partial download is REFUSED
        rather than padded.  Holding levels for only the recent years would mean
        training on one feature set and testing on another, which is exactly the
        distribution shift `min_driver_coverage` exists to prevent -- and here it
        would be worse, because the affected channels are the ones we are adding
        the whole feature block to measure.

        Sets ``self.levels`` to ``[T, C, n_nodes]`` or leaves it ``None``.
        """
        self.levels = None
        self.level_channels = ()
        self.level_coverage = 0.0
        if not self.cfg.pressure_levels:
            return
        if not os.path.isdir(self.cfg.levels_dir):
            self._log(f"  NOTE: pressure levels requested but {self.cfg.levels_dir} "
                      f"does not exist -- run experiments/data_prep/download_era5_levels.py. "
                      f"Continuing WITHOUT the stability block.")
            return

        files = sorted(f for f in os.listdir(self.cfg.levels_dir)
                       if f.startswith("era5_lev_") and f.endswith(".nc"))
        if not files:
            self._log("  NOTE: no era5_lev_*.nc found (parts may still be "
                      "downloading) -- continuing WITHOUT the stability block.")
            return

        parts = [xr.open_dataset(os.path.join(self.cfg.levels_dir, f)) for f in files]
        try:
            lev = xr.concat(parts, dim="valid_time").sortby("valid_time")
            factor = CADENCE_FACTOR[self.cfg.cadence]
            n_trim = ((min(len(lev.valid_time), self.n_steps * factor)) // factor) * factor
            self.level_coverage = n_trim / factor / self.n_steps
            if self.level_coverage < self.cfg.min_driver_coverage:
                self._log(
                    f"  NOTE: pressure levels cover only "
                    f"{self.level_coverage*100:.1f}% of the record "
                    f"({len(files)} year file(s)); below the "
                    f"{self.cfg.min_driver_coverage*100:.0f}% threshold, so the "
                    f"stability block is IGNORED. Re-run once the download "
                    f"completes -- partial levels would train and test on "
                    f"different feature sets.")
                return
            stacked = stab.derive(lev.isel(valid_time=slice(0, n_trim)))
            # [T, C, H, W] -> block-mean the time axis -> gather to nodes per channel.
            chans = [self._gather(self._block_mean(stacked[:, c], n_trim, factor)
                                  )[:self.n_steps]
                     for c in range(stacked.shape[1])]
            self.levels = np.stack(chans, axis=1).astype("float32")
            self.level_channels = stab.CHANNEL_NAMES
            self._log(f"  pressure levels: {len(stab.CHANNEL_NAMES)} stability "
                      f"channels from {len(files)} year file(s), coverage "
                      f"{self.level_coverage*100:.1f}%")
        finally:
            for p in parts:
                p.close()

    # ------------------------------------------------------------------ #
    def _derive(self, raw):
        cfg = self.cfg
        pm10_phys = raw["pm10"] * 1e9                     # kg/m^3 -> ug/m^3
        pm25_phys = raw["pm2p5"] * 1e9
        u_grid, v_grid = raw["u10"], raw["v10"]
        speed_grid = np.sqrt(u_grid ** 2 + v_grid ** 2).astype("float32")

        drivers = self._load_emission_drivers(speed_grid, raw["tp"])
        em = emis.compute_emission(drivers, use_drag_partition=cfg.drag_partition)
        self.drivers_meta = drivers.provenance(use_drag_partition=cfg.drag_partition)
        self.emission_active_fraction = em["active_fraction"]
        self._log("  emission drivers: " + "; ".join(f"{k}={v}" for k, v in
                                                     self.drivers_meta.items()))
        self._log(f"  emission-active fraction: {em['active_fraction']*100:.1f}% "
                  f"(drag_partition={em['drag_partition']}, "
                  f"dynamic-driver coverage {self.driver_coverage*100:.1f}%)")
        if not 0.001 < em["active_fraction"] < 0.6:
            self._log("  WARNING: emission scheme is effectively dead or saturated -- "
                      "the emission feature and the regime split will carry no signal")

        # Courant diagnostic: the Euler reference is only valid below 1.
        cfl = trans.courant_number(u_grid, v_grid, self.dx, self.dy, self.dt)
        self.cfl_exceed_fraction = float(np.mean(cfl >= 1.0))
        self._log(f"  CFL: mean {float(cfl.mean()):.2f}, "
                  f"{self.cfl_exceed_fraction*100:.1f}% of cell-times >= 1 "
                  f"| reference_mode={cfg.reference_mode}")

        # --- chunked gridded pass: transport + reference concentration --------- #
        ref10 = np.empty((self.n_steps, self.n_nodes), "float32")
        ref25 = np.empty((self.n_steps, self.n_nodes), "float32")
        transport = np.empty((self.n_steps, self.n_nodes), "float32")

        for s in range(0, self.n_steps, CHUNK_STEPS):
            e = min(s + CHUNK_STEPS, self.n_steps)
            tend10 = em["tendency"][s:e]
            tend25 = tend10 * PM25_EMISSION_FRACTION
            transport[s:e] = self._gather(
                trans.transport_tendency(pm10_phys[s:e], u_grid[s:e], v_grid[s:e],
                                         self.dx, self.dy, divergence=cfg.divergence))
            ref10[s:e] = self._gather(trans.build_reference(
                cfg.reference_mode, pm10_phys[s:e], u_grid[s:e], v_grid[s:e],
                self.dx, self.dy, self.dt, emission_tendency=tend10,
                divergence=cfg.divergence, mask=self.mask))
            ref25[s:e] = self._gather(trans.build_reference(
                cfg.reference_mode, pm25_phys[s:e], u_grid[s:e], v_grid[s:e],
                self.dx, self.dy, self.dt, emission_tendency=tend25,
                divergence=cfg.divergence, mask=self.mask))

        # --- node-level cache --------------------------------------------------- #
        self.pm10 = trans.to_log(self._gather(pm10_phys))
        self.pm25 = trans.to_log(self._gather(pm25_phys))
        self.c_ref10 = trans.to_log(ref10)
        self.c_ref25 = trans.to_log(ref25)
        self.transport = transport
        self.u = self._gather(u_grid)
        self.v = self._gather(v_grid)
        self.speed = self._gather(speed_grid)
        self.t2m = self._gather(raw["t2m"])
        self.rh = self._gather(raw["rh"])
        self.tp = np.log1p(np.clip(self._gather(raw["tp"]), 0, None) * 1e3).astype("float32")

        self.ustar = self._gather(np.broadcast_to(em["ustar_eff"], pm10_phys.shape))
        self.emission = np.log1p(self._gather(
            np.broadcast_to(em["E"], pm10_phys.shape)) * 1e9).astype("float32")
        self.emis_tendency = self._gather(em["tendency"])
        self.emis_active = self._gather(
            np.broadcast_to(em["active"], pm10_phys.shape).astype("float32")) > 0.5

        # Courant-normalised winds: u/dx, v/dy -- AI-DUST's scale-invariant form.
        dx_nodes = np.broadcast_to(self.dx, self.grid_shape).ravel()[self.node_flat]
        dy_nodes = np.full(self.n_nodes, abs(self.dy), "float32")
        self.u_courant = (self.u / dx_nodes * self.dt).astype("float32")
        self.v_courant = (self.v / dy_nodes * self.dt).astype("float32")

        # Relative elevation: absolute height minus the domain mean (full-domain
        # topology) or minus the patch centre (patch topology, applied at assembly).
        if self.elevation is not None:
            elev_nodes = self.elevation.ravel()[self.node_flat]
            self.elev = (elev_nodes - elev_nodes.mean()).astype("float32")
        else:
            self.elev = np.zeros(self.n_nodes, "float32")

        doy = (self.times.astype("datetime64[D]") - self.times.astype("datetime64[Y]")
               ).astype(int) + 1
        self.dsin = np.sin(2 * np.pi * doy / 365.25).astype("float32")
        self.dcos = np.cos(2 * np.pi * doy / 365.25).astype("float32")

        # Computed only when asked for. These allocate several node-level
        # arrays plus a gridded gather for the precipitation memory, and this
        # loader already peaks near the machine's limit -- an unconditional
        # version cost a run its remaining headroom and crashed it.
        self.hsin = self.hcos = self.blh = None
        self.ustar_excess = self.wetness = None
        if cfg.physics_features:
            # --- hour of day -------------------------------------------------- #
            # The model had NO time-of-day channel at all, only day-of-year, so at a
            # 3-hourly cadence it saw eight samples a day and could not tell 03:00
            # from 15:00. Dust has a sharp diurnal cycle: a nocturnal low-level jet
            # decouples from the surface overnight, convective mixing drags it down
            # after sunrise as a morning emission burst, mixing peaks in the
            # afternoon, and a stable nocturnal layer concentrates whatever is left
            # at eye level. None of that was representable.
            hour = (self.times.astype("datetime64[h]").astype(int) % 24).astype("float32")
            self.hsin = np.sin(2 * np.pi * hour / 24.0).astype("float32")
            self.hcos = np.cos(2 * np.pi * hour / 24.0).astype("float32")

            # --- mixing depth, exposed for transport --------------------------- #
            # `em["H"]` is the ERA5 boundary-layer height (clipped), already computed
            # for the emission scheme but never visible outside it. Deposition needs
            # it: the same removal rate thins a deep mixed layer far more slowly than
            # a shallow one.
            self.blh = self._gather(
                np.broadcast_to(em["H"], pm10_phys.shape)).astype("float32")

            # --- how far above the saltation threshold ------------------------- #
            # Emission is ~ u*(u*^2 - u*t^2) above threshold and EXACTLY zero below.
            # The product was supplied but the switch itself was not, leaving the
            # network to infer a hard nonlinearity from a smooth input.
            self.ustar_excess = np.clip(
                self._gather(np.broadcast_to(em["ustar_eff"] - em["ustar_t"],
                                             pm10_phys.shape)), 0.0, None).astype("float32")

            # --- soil moisture MEMORY ------------------------------------------ #
            # Fecan corrects the threshold using instantaneous moisture, but soil
            # stays damp for days after rain and stays dust-suppressed that whole
            # time. An exponentially weighted sum of antecedent precipitation,
            # acc_t = a*acc_{t-1} + p_t, gives the model a drying clock.
            #
            # Done in place, in float32. `scipy.signal.lfilter` is the obvious way
            # to write this recursion and it upcasts to float64, so it wanted a
            # 129 MB temporary on top of the input and the log1p output -- ~260 MB
            # of transients at the exact moment this loader already peaks. That
            # was enough to abort a run on this machine. The explicit loop is
            # 64k iterations over a 251-element row: about a second, no
            # allocation beyond the array itself.
            hours_per_step = SECONDS_PER_STEP[cfg.cadence] / 3600.0
            a = float(np.exp(-hours_per_step / (WETNESS_TAU_DAYS * 24.0)))
            wet = self._gather(raw["tp"]).astype("float32", copy=True)
            np.multiply(wet, 1e3, out=wet)          # m -> mm
            np.clip(wet, 0.0, None, out=wet)
            for t in range(1, wet.shape[0]):
                wet[t] += a * wet[t - 1]
            np.log1p(wet, out=wet)
            self.wetness = wet
        self.years = self.times.astype("datetime64[Y]").astype(int) + 1970

        # Last, because it gathers to nodes and so needs node_flat and n_steps.
        self._load_levels()

        self._log(f"  cached {self.n_steps} steps x {self.n_nodes} nodes")

    # ------------------------------------------------------------------ #
    def split_indices(self, y0, y1):
        """Valid sample indices for a year range (needs history and a t+1 target)."""
        cfg = self.cfg
        sel = np.flatnonzero((self.years >= y0) & (self.years <= y1))
        lo, hi = int(sel[0]), int(sel[-1])
        start = max(lo, cfg.window - 1)
        return np.arange(start, min(hi, self.n_steps - 2) + 1, dtype=int)
