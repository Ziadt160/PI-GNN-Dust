"""Shao-type dust emission scheme -- the source term your models currently lack.

AI-DUST attributes much of its skill to an explicit, physically-based emission
scheme fed in as an *input feature* (not as a loss penalty): saltation driven by
friction velocity against a soil/vegetation-dependent threshold, with the
resulting flux entered as ``E / H``.  Their Fig. 9 ablation shows the surface
roughness input alone is worth ~+18% TS at every lead day.

`saudi_merged.nc` contains none of the required surface properties, so every
driver here has two paths:

  * **measured**  -- the ERA5 field, when `experiments/download_era5_emission.py`
    has been run (`zust`, `swvl1`, `fsr`, `lai_*`, `blh`);
  * **proxy**     -- a documented approximation from what the merged file already
    has (`u10`, `v10`, `tp`), used so the pipeline runs before/without those
    downloads.

`EmissionDrivers.provenance()` reports which path each driver took, and the
benchmark prints it, so a proxy-driven number is never mistaken for a
measured-driver one.

Physics
-------
Friction velocity (log law, neutral):      u*  = k U(z) / ln(z/z0)
Drag partition (Marticorena & Bergametti,   f_eff -- fraction of u* felt by the
1995), the "surface roughness" lever:       erodible surface between roughness
                                            elements
Soil-moisture threshold correction          f_w = sqrt(1 + 1.21 (w - w')^0.68)
(Fecan et al., 1999):
Saltation flux (Shao 2004 simplification    Q   = c (rho_a/g) u*^3 (1 - u*t^2/u*^2)
of Owen/White -- drops the (1 + u*t/u*)
factor, which is the "simplification" the
AI-DUST paper cites):
Vertical dust flux (sandblasting):          E   = alpha Q (1 - veg) land

Known gap vs AI-DUST: they derive u*t from SoilGrids soil texture and Wei/Gong
grain-size distributions.  We have no soil texture for this domain, so texture is
folded into the tuned constants `ALPHA` and `UST_T_DRY` and held uniform.  That is
the single largest fidelity gap in this module and is reported as such.

All functions are NumPy, shape-agnostic, and broadcast over ``[T, N]`` or ``[N]``.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --- physical constants ----------------------------------------------------- #
KARMAN = 0.4            # von Karman constant
RHO_AIR = 1.225         # kg m^-3, near-surface air density
GRAVITY = 9.81          # m s^-2
REF_HEIGHT = 10.0       # m, ERA5 wind reference height

# --- tunable scheme constants (texture/grain-size surrogates) ---------------- #
Z0_BARE = 1.0e-2        # m, roughness length of bare desert used by the log-law proxy
Z0_SMOOTH = 1.0e-5      # m, smooth-surface roughness in the drag partition
X_MB95 = 0.1            # m, Marticorena & Bergametti reference length
UST_T_DRY = 0.30        # m s^-1, dry threshold friction velocity (fine desert sand)
CLAY_PCT = 10.0         # %, assumed clay content (Fecan residual moisture)
SOIL_BULK_DENSITY = 1300.0   # kg m^-3, for volumetric -> gravimetric soil moisture
C_SALTATION = 2.61      # Owen/White saltation constant
ALPHA_SANDBLAST = 1.0e-5     # m^-1, vertical/horizontal flux ratio
DEFAULT_MIXING_HEIGHT = 3000.0   # m -- AI-DUST's fixed H, used when `blh` is absent

# Wetness proxy: precipitation memory used when ERA5 soil moisture is unavailable.
TP_DECAY_STEPS = 16     # ~2 days at 3 h cadence
TP_TO_SOILWATER = 20.0  # m^-1, maps decayed precip (m) onto volumetric soil water


@dataclass
class EmissionDrivers:
    """Per-timestep driver fields, each either the ERA5 product or a proxy.

    Shapes broadcast: any of ``[T, N]`` (time-varying) or ``[N]`` (static /
    climatology broadcast over time).
    """
    speed: np.ndarray                          # 10 m wind speed (m/s) -- always required
    zust: Optional[np.ndarray] = None          # ERA5 friction velocity (m/s)
    swvl1: Optional[np.ndarray] = None         # volumetric soil water, layer 1 (m3/m3)
    tp: Optional[np.ndarray] = None            # total precipitation (m) -- proxy path
    z0: Optional[np.ndarray] = None            # surface roughness length (m), ERA5 fsr
    lai: Optional[np.ndarray] = None           # leaf area index (m2/m2), lai_hv + lai_lv
    blh: Optional[np.ndarray] = None           # boundary layer height (m)
    land: Optional[np.ndarray] = None          # land fraction in [0, 1]
    # Observationally-derived dust-source strength, [H, W] in [0, 1]: the fraction
    # of windy satellite observations at which that cell actually turned dusty.
    # Replaces the single uniform erodibility assumption. SoilGrids cannot supply
    # this over Arabia -- it returns a near-constant global mean (0/20 sampled
    # cells above 80% sand in a country of sand seas), so the map is built from
    # MAIAC AOD conditioned on ERA5 friction velocity instead.
    erodibility: Optional[np.ndarray] = None
    _prov: dict = field(default_factory=dict, repr=False)

    def provenance(self, use_drag_partition=False) -> dict:
        """Which path each driver resolved to -- printed in the benchmark header.

        Reports what was *used*, not what was downloaded, so a proxy-driven result
        can never be read as a measured-driver one.
        """
        if self.z0 is None:
            roughness = "none available (f_eff=1)"
        elif use_drag_partition:
            roughness = "ERA5 fsr, MB95 drag partition ON"
        else:
            roughness = "ERA5 fsr present but UNUSED (drag partition off: fsr is "
            roughness += "aerodynamic, not aeolian -- would zero 99% of cells)"
        return {
            "u*": "ERA5 zust" if self.zust is not None else f"log-law proxy (z0={Z0_BARE} m)",
            "soil moisture": "ERA5 swvl1" if self.swvl1 is not None else (
                "precip-decay proxy" if self.tp is not None else "assumed dry"),
            "roughness": roughness,
            "vegetation": "ERA5 LAI" if self.lai is not None else "none (bare)",
            "mixing height": "ERA5 blh" if self.blh is not None else f"fixed {DEFAULT_MIXING_HEIGHT} m",
            "soil texture": f"NOT AVAILABLE -- uniform clay={CLAY_PCT}%, u*t_dry={UST_T_DRY} m/s",
            "source strength": ("MAIAC-derived erodibility map"
                                if self.erodibility is not None else
                                "uniform (every cell equally erodible)"),
        }


# Range of aeolian roughness lengths a drag-partition scheme is valid over.
# ERA5 `fsr` on this domain averages ~1.6e-2 m -- an *aerodynamic* roughness that
# folds in subgrid orography and land cover, one to two orders above the aeolian
# roughness MB95 assumes.  Values outside this window fall back to Z0_BARE.
Z0_AEOLIAN_RANGE = (1.0e-5, 5.0e-3)


def friction_velocity(speed, zust=None, z0=None):
    """u* (m/s): ERA5 `zust` when present, else the neutral log-law from 10 m wind.

    When a roughness field is supplied it is used in the log law, but only where
    it is physically plausible as an aeolian roughness; elsewhere the bare-desert
    constant is used.  ERA5's own `fsr` is almost entirely outside that window on
    this domain, so in practice the proxy runs at ``Z0_BARE``.
    """
    if zust is not None:
        return np.asarray(zust, dtype="float32")
    if z0 is None:
        z0_eff = np.float32(Z0_BARE)
    else:
        z0 = np.asarray(z0, dtype="float32")
        ok = (z0 >= Z0_AEOLIAN_RANGE[0]) & (z0 <= Z0_AEOLIAN_RANGE[1])
        z0_eff = np.where(ok, z0, Z0_BARE).astype("float32")
    return (KARMAN * np.asarray(speed, dtype="float32")
            / np.log(REF_HEIGHT / z0_eff)).astype("float32")


def drag_partition(z0=None):
    """Fraction of u* acting on the erodible surface (Marticorena & Bergametti 1995).

    Roughness elements absorb momentum that would otherwise drive saltation, so the
    effective friction velocity is ``f_eff * u*`` with ``f_eff <= 1``.  This is the
    ERA5 stand-in for AI-DUST's high-resolution roughness density -- their single
    most valuable surface input.  Returns 1.0 when no roughness field is supplied.
    """
    if z0 is None:
        return np.float32(1.0)
    z0 = np.clip(np.asarray(z0, dtype="float32"), Z0_SMOOTH, 1.0)
    num = np.log(z0 / Z0_SMOOTH)
    den = np.log(0.7 * (X_MB95 / Z0_SMOOTH) ** 0.8)
    return np.clip(1.0 - num / den, 0.0, 1.0).astype("float32")


def _wetness_proxy(tp, decay_steps=TP_DECAY_STEPS):
    """Volumetric-soil-water surrogate from an exponentially decayed precip memory.

    Used only when ERA5 `swvl1` is unavailable.  `tp` is ERA5 total precipitation in
    metres accumulated per step, with time on axis 0 and any trailing spatial shape
    (``[T, N]`` node-wise or ``[T, H, W]`` gridded); recent rain suppresses emission
    for a couple of days.
    """
    tp = np.clip(np.nan_to_num(np.asarray(tp, dtype="float32")), 0.0, None)
    if tp.ndim == 1:                      # static field, no time axis
        return (tp * TP_TO_SOILWATER).astype("float32")
    decay = np.float32(np.exp(-1.0 / decay_steps))
    out = np.empty_like(tp)
    acc = np.zeros(tp.shape[1:], dtype="float32")
    for t in range(tp.shape[0]):
        acc = acc * decay + tp[t]
        out[t] = acc
    return (out * TP_TO_SOILWATER).astype("float32")


def soil_moisture_correction(swvl1=None, tp=None, clay_pct=CLAY_PCT):
    """Fecan et al. (1999) multiplier on the dry threshold friction velocity.

    Converts volumetric soil water to gravimetric %, subtracts the clay-bound
    residual w' below which moisture does not resist saltation, and returns
    ``f_w >= 1``.
    """
    if swvl1 is not None:
        vol = np.clip(np.nan_to_num(np.asarray(swvl1, dtype="float32")), 0.0, 1.0)
    elif tp is not None:
        vol = np.clip(_wetness_proxy(tp), 0.0, 1.0)
    else:
        return np.float32(1.0)

    w = vol * (1000.0 / SOIL_BULK_DENSITY) * 100.0          # volumetric -> gravimetric %
    w_res = 0.0014 * clay_pct ** 2 + 0.17 * clay_pct        # Fecan residual moisture
    excess = np.clip(w - w_res, 0.0, None)
    return np.sqrt(1.0 + 1.21 * excess ** 0.68).astype("float32")


def threshold_friction_velocity(swvl1=None, tp=None, ust_t_dry=UST_T_DRY, clay_pct=CLAY_PCT):
    """u*t (m/s) -- dry threshold scaled up by the soil-moisture correction."""
    return (ust_t_dry * soil_moisture_correction(swvl1, tp, clay_pct)).astype("float32")


def vegetation_gate(lai=None):
    """Erodible (unvegetated) area fraction from LAI via a Beer-Lambert cover model."""
    if lai is None:
        return np.float32(1.0)
    lai = np.clip(np.nan_to_num(np.asarray(lai, dtype="float32")), 0.0, None)
    return np.exp(-0.5 * lai).astype("float32")           # = 1 - cover


def saltation_flux(ustar_eff, ustar_t):
    """Horizontal saltation flux Q (kg m^-1 s^-1), Shao (2004) simplified form.

    ``Q = c (rho_a/g) u*^3 (1 - u*t^2/u*^2)`` for ``u* > u*t``, else 0.
    """
    ustar_eff = np.asarray(ustar_eff, dtype="float32")
    ustar_t = np.asarray(ustar_t, dtype="float32")
    active = ustar_eff > ustar_t
    safe = np.where(active, ustar_eff, 1.0).astype("float32")     # avoid 0-division
    q = C_SALTATION * (RHO_AIR / GRAVITY) * safe ** 3 * (1.0 - (ustar_t / safe) ** 2)
    return np.where(active, q, 0.0).astype("float32")


def compute_emission(drivers: EmissionDrivers, alpha=ALPHA_SANDBLAST,
                     use_drag_partition=False):
    """Full scheme -> dict of fields used as model inputs and in ``C_ref``.

    Parameters
    ----------
    use_drag_partition : **off by default, deliberately.**  Applying MB95 to
        ERA5 `fsr` on this domain drives ``f_eff`` to exactly 0 for 99.2% of
        cells (measured), which zeroes emission everywhere and turns the
        emission feature into a dead channel and the regime split into a no-op.
        The cause is a units-of-meaning mismatch, not a coding error: `fsr` is an
        *aerodynamic* roughness including subgrid orography and land cover
        (~1.6e-2 m here), whereas MB95's partition expects the *aeolian*
        roughness of the erodible surface (~1e-4 to 1e-3 m).  With the partition
        off, ``u* > u*t`` fires ~17% of the time, a plausible emission-active
        frequency for the Arabian Peninsula.  Turn it on only with a genuine
        aeolian roughness product.

    Returns
    -------
    ustar, ustar_eff, ustar_t : friction velocities (m/s)
    E          : vertical dust flux (kg m^-2 s^-1)
    H          : mixing height (m)
    tendency   : ``E/H`` as a concentration tendency (ug m^-3 s^-1) -- the term
                 that enters AI-DUST Eq. 1
    active     : bool, u*_eff > u*t -- AI-DUST's emission/non-emission split key
    active_fraction : share of cell-times with emission; a sanity check that the
                 scheme is neither dead nor saturated
    """
    ustar = friction_velocity(drivers.speed, drivers.zust, drivers.z0)
    f_eff = drag_partition(drivers.z0) if use_drag_partition else np.float32(1.0)
    ustar_eff = (ustar * f_eff).astype("float32")

    ustar_t = threshold_friction_velocity(drivers.swvl1, drivers.tp)
    q = saltation_flux(ustar_eff, ustar_t)

    erodible = vegetation_gate(drivers.lai)
    land = np.float32(1.0) if drivers.land is None else np.clip(
        np.nan_to_num(np.asarray(drivers.land, dtype="float32")), 0.0, 1.0)

    # Observational source strength. Without it every cell is assumed equally
    # erodible, so the scheme knows when the wind is strong but not whether the
    # surface can supply dust -- which is why replacing the log-law u* with
    # measured ERA5 zust changed the forecast scores by nothing. Normalised to a
    # mean of 1 so the tuned ALPHA keeps its meaning and only the *relative*
    # pattern is imposed.
    if drivers.erodibility is None:
        source = np.float32(1.0)
    else:
        e = np.asarray(drivers.erodibility, dtype="float32")
        finite = np.isfinite(e)
        mean = float(e[finite].mean()) if finite.any() else 0.0
        source = (np.where(finite, e, mean) / mean).astype("float32") if mean > 0 \
            else np.float32(1.0)

    emission = (alpha * q * erodible * land * source).astype("float32")  # kg m^-2 s^-1
    height = (DEFAULT_MIXING_HEIGHT if drivers.blh is None
              else np.clip(np.nan_to_num(np.asarray(drivers.blh, dtype="float32")), 50.0, 6000.0))

    tendency = (emission / height * 1.0e9).astype("float32")       # -> ug m^-3 s^-1
    active = ustar_eff > ustar_t

    return {
        "ustar": ustar,
        "ustar_eff": ustar_eff,
        "ustar_t": np.broadcast_to(ustar_t, ustar_eff.shape).astype("float32"),
        "E": emission,
        "H": height,
        "tendency": tendency,
        "active": active,
        "active_fraction": float(np.mean(active)),
        "drag_partition": bool(use_drag_partition),
    }
