"""Boundary-layer stability diagnostics from ERA5 pressure levels.

The missing half of the physics
-------------------------------
Emission decides how much dust leaves the ground and transport decides where it
goes, but neither decides how much of it is at eye level.  That is set by how
deeply the boundary layer mixes: the same dust load gives a 20 km visibility when
stirred through 3 km of atmosphere and a 500 m visibility when trapped under an
inversion.

Every feature in this repo before now was a surface field, so this control was
absent from the model entirely.  Xu et al. 2026 (Sci Rep 16:11824), post-
processing ECMWF over the Kumtag Desert, measure vertical mixing as the dominant
mechanism at every forecast horizon -- ahead of both transport and emission -- and
their top three SHAP predictors are ``t_500``, ``t_700-500`` and ``Ri(850-700)``.
This module computes those and their neighbours.

Definitions
-----------
Potential temperature, the quantity that is conserved under dry adiabatic
ascent and so the one whose vertical gradient measures stability::

    theta = T (p0/p)^(R_d/c_p),   p0 = 1000 hPa,  R_d/c_p = 0.2857

Bulk gradient Richardson number between a lower level ``a`` and an upper level
``b`` -- the ratio of buoyant suppression to shear production of turbulence::

    Ri = g (theta_b - theta_a)(z_b - z_a) / [ theta_mean ((du)^2 + (dv)^2) ]

``Ri`` below ~0.25 means shear wins and the layer mixes; large positive ``Ri``
means a stable layer that caps mixing and concentrates dust near the surface.
It is the physically meaningful form of "is this air stirred or stratified".

Numerical care
--------------
``Ri`` has a pole: as the vertical wind difference goes to zero the denominator
does too, and the value runs to +-inf.  Clipping the *output* would pile a
physically meaningless spike at the clip value.  Instead the denominator is
floored at a small shear (``MIN_SHEAR_MS``, 0.5 m/s across the layer, below ERA5's
own useful resolution) and the result is then squashed with ``arctan`` into a
bounded, monotone, derivative-friendly range.  Both steps are reversible and
neither invents an ordering that is not in the data.

Heights come from geopotential (``z = Phi / g``), not from a standard atmosphere,
so layer thicknesses respond to the actual thermal structure.
"""
import numpy as np

GRAVITY = 9.80665           # m/s^2
R_OVER_CP = 0.2857          # R_d / c_p for dry air
P0_HPA = 1000.0

# Floor on the vector wind difference across a layer, in m/s. ERA5 wind at 0.75
# degrees does not resolve shear below this, so a smaller value is noise, and it
# is exactly where Ri's denominator explodes.
MIN_SHEAR_MS = 0.5

# The five levels this module expects, in hPa. Listed top-down for reading; the
# ORDER HERE IS NOT THE FILE'S ORDER -- CDS returns pressure_level ascending by
# height, i.e. [900, 850, 700, 500, 250]. Every access goes through `_idx`, which
# looks a level up by value, so neither order can silently swap two levels. Do
# not replace those lookups with positional indexing.
LEVELS_HPA = (250, 500, 700, 850, 900)

# Feature names, in the order `derive` stacks them. Kept as a module constant so
# the feature assembler and the ablation can name channels without recomputing.
CHANNEL_NAMES = (
    "t_500",             # Xu's single most important predictor
    "t_700_500",         # mid-tropospheric lapse: subsidence vs destabilisation
    "ri_850_700",        # Xu's top physics-derived feature, capping layer
    "ri_900_850",        # near-surface stability: does dust stay at eye level
    "theta_900",         # absolute low-level stability state
    "dtheta_900_700",    # bulk stratification across the dust-carrying layer
    "shear_900_700",     # shear production of turbulence, 1/s
    "llj_speed",         # strongest low-level wind (900/850/700)
    "llj_level",         # which of those three carries it, as 0/0.5/1
    "q_900",             # low-level moisture: wet ground and hygroscopic growth
    "z_500",             # 500 hPa height: the synoptic wave carrying the shamal
)


def _idx(level_hpa, levels):
    """Position of `level_hpa` in the file's own level coordinate."""
    levels = [int(round(float(v))) for v in levels]
    return levels.index(int(level_hpa))


def potential_temperature(t, p_hpa):
    """Potential temperature [K] from temperature [K] at pressure `p_hpa`."""
    return t * (P0_HPA / float(p_hpa)) ** R_OVER_CP


def bulk_richardson(theta_lo, theta_hi, z_lo, z_hi, u_lo, u_hi, v_lo, v_hi):
    """Bulk gradient Richardson number across a layer, squashed to (-pi/2, pi/2).

    Positive = stable (buoyancy suppresses mixing, dust stays low).
    Near zero or negative = well mixed or convective.

    Returns ``arctan(Ri)`` rather than ``Ri``: see the module docstring. The
    transform is monotone, so any threshold a model learns in the squashed space
    corresponds to a real threshold in ``Ri``.
    """
    du = u_hi - u_lo
    dv = v_hi - v_lo
    shear_sq = np.maximum(du * du + dv * dv, MIN_SHEAR_MS ** 2)
    dz = z_hi - z_lo
    theta_mean = 0.5 * (theta_lo + theta_hi)
    ri = GRAVITY * (theta_hi - theta_lo) * dz / (theta_mean * shear_sq)
    return np.arctan(ri).astype("float32")


def derive(ds, levels_name="pressure_level"):
    """Stack the stability diagnostics from an open ERA5 pressure-level dataset.

    Parameters
    ----------
    ds : xarray.Dataset with ``z``, ``t``, ``q``, ``u``, ``v`` on
        ``(valid_time, pressure_level, latitude, longitude)``.

    Returns
    -------
    ``[T, len(CHANNEL_NAMES), H, W]`` float32, channels in ``CHANNEL_NAMES`` order.
    """
    levels = ds[levels_name].values
    i250, i500, i700, i850, i900 = (_idx(p, levels) for p in LEVELS_HPA)

    z = ds["z"].values / GRAVITY            # geopotential -> geometric height, m
    t = ds["t"].values
    q = ds["q"].values
    u = ds["u"].values
    v = ds["v"].values

    th = {p: potential_temperature(t[:, i], p)
          for p, i in ((500, i500), (700, i700), (850, i850), (900, i900))}

    # Low-level jet: which of the three lowest levels carries the strongest wind,
    # and how strong. A nocturnal jet at 900-850 hPa is a classic shamal dust
    # driver -- it mixes down after sunrise and lifts dust in a burst.
    spd = np.stack([np.hypot(u[:, i], v[:, i]) for i in (i900, i850, i700)], axis=1)
    llj_speed = spd.max(axis=1)
    llj_level = (spd.argmax(axis=1) / 2.0)          # 0 = 900 hPa, 1 = 700 hPa

    du = u[:, i700] - u[:, i900]
    dv = v[:, i700] - v[:, i900]
    dz = np.maximum(z[:, i700] - z[:, i900], 1.0)
    shear = np.hypot(du, dv) / dz                    # 1/s

    out = np.stack([
        t[:, i500],
        t[:, i700] - t[:, i500],
        bulk_richardson(th[850], th[700], z[:, i850], z[:, i700],
                        u[:, i850], u[:, i700], v[:, i850], v[:, i700]),
        bulk_richardson(th[900], th[850], z[:, i900], z[:, i850],
                        u[:, i900], u[:, i850], v[:, i900], v[:, i850]),
        th[900],
        th[700] - th[900],
        shear,
        llj_speed,
        llj_level,
        q[:, i900],
        z[:, i500],
    ], axis=1).astype("float32")

    # 250 hPa is downloaded and deliberately unused here: it carries the jet
    # stream, which matters for synoptic development but not for whether dust is
    # at eye level. It is kept in the archive so a later experiment can add it
    # without a second 22-year download.
    _ = i250
    return out
