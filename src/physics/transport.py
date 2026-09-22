"""Transport terms and the physical reference concentration ``C_ref``.

Two jobs:

1. **Flux divergence** ``div(VC) = d(uC)/dx + d(vC)/dy`` -- AI-DUST's transport
   term.  It differs from the ``u . grad C`` used in `src/physics/loss.py` by the
   extra ``C div(V)`` term, i.e. wind **convergence**, a real dust-accumulation
   mechanism that an advection-only formulation cannot represent.  Both forms are
   provided so the `+divergence` ablation can switch between them.

2. **The reference concentration** that AI-DUST's output is anchored to
   (their Eq. 1)::

       C_ref = C_t + (-div(VC) + E/H) * dt

   The paper writes the bracket without a ``dt``, which is dimensionally
   incomplete -- a divergence is a concentration *per second*.  We read it as a
   one-step explicit-Euler update and insert ``dt`` explicitly.  We also evaluate
   it in **physical** space (ug/m^3) and convert to log afterwards, because a
   divergence of a log-concentration has no physical meaning, while the paper's
   Fig. 1 makes clear the *learning* target is log-transformed.

Sign convention is derived from the coordinate vectors rather than assumed: the
signed grid spacing carries the sign of the latitude/longitude step, so a
north-to-south ordered grid (the ERA5 default, and the source of an earlier sign
bug in this repo) is handled without a special case.

Arrays are ``[T, H, W]`` (or ``[H, W]``) NumPy, with ``H`` the latitude axis.
"""
import numpy as np

EARTH_M_PER_DEG = 111_320.0


def grid_spacing(lats, lons):
    """Signed physical spacing of a regular lat/lon grid.

    Returns
    -------
    dx : ``[H, 1]`` signed east-west spacing in metres (shrinks as ``cos(lat)``)
    dy : signed north-south spacing in metres (negative if latitude descends)
    """
    lats = np.asarray(lats, dtype="float64")
    lons = np.asarray(lons, dtype="float64")
    dlat = float(lats[1] - lats[0])
    dlon = float(lons[1] - lons[0])
    dy = dlat * EARTH_M_PER_DEG
    dx = (dlon * EARTH_M_PER_DEG * np.cos(np.deg2rad(lats)))[:, None]
    return dx.astype("float64"), float(dy)


def _d_dx(field, dx):
    """d/d(east) in m^-1 units -- central differences along the longitude axis."""
    return np.gradient(field, axis=-1) / dx


def _d_dy(field, dy):
    """d/d(north) -- signed dy makes a descending-latitude grid come out correct."""
    return np.gradient(field, axis=-2) / dy


def flux_divergence(conc, u, v, dx, dy):
    """``div(VC) = d(uC)/dx + d(vC)/dy``  (concentration per second)."""
    return (_d_dx(u * conc, dx) + _d_dy(v * conc, dy)).astype("float32")


def advection(conc, u, v, dx, dy):
    """``u . grad C``  -- the advection-only form, for the ablation control."""
    return (u * _d_dx(conc, dx) + v * _d_dy(conc, dy)).astype("float32")


def transport_tendency(conc, u, v, dx, dy, divergence=True):
    """Transport contribution to dC/dt: ``-div(VC)`` (default) or ``-u . grad C``."""
    term = flux_divergence(conc, u, v, dx, dy) if divergence else advection(conc, u, v, dx, dy)
    return (-term).astype("float32")


def reference_concentration(conc_phys, u, v, dx, dy, dt,
                            emission_tendency=None, divergence=True, mask=None):
    """AI-DUST Eq. 1 reference state, in physical units (ug/m^3).

    ``C_ref = clamp( C_t + (-div(VC) + E/H) * dt , 0 )``

    Parameters
    ----------
    conc_phys : ``[T, H, W]`` concentration at t in ug/m^3 (NOT log space).
    emission_tendency : ``E/H`` in ug m^-3 s^-1, or None to omit the source term.
    mask : optional ``[H, W]`` validity mask; invalid cells are zero-filled before
        differencing, which imposes a zero-concentration (open outflow) boundary --
        the same treatment AI-DUST gives its patch edges.
    """
    conc_phys = np.nan_to_num(np.asarray(conc_phys, dtype="float32"))
    if mask is not None:
        conc_phys = conc_phys * mask.astype("float32")

    tend = transport_tendency(conc_phys, u, v, dx, dy, divergence=divergence)
    if emission_tendency is not None:
        tend = tend + np.asarray(emission_tendency, dtype="float32")

    ref = conc_phys + tend * float(dt)
    if mask is not None:
        ref = ref * mask.astype("float32")
    return np.clip(ref, 0.0, None).astype("float32")


def semilagrangian_reference(conc_phys, u, v, dx, dy, dt,
                             emission_tendency=None, mask=None):
    """CFL-stable alternative to :func:`reference_concentration`.

    Why this exists
    ---------------
    AI-DUST's Eq. 1 advances the reference state with an explicit Euler step.
    That is only stable for Courant numbers below 1.  Measured on this domain
    (0.75 deg, 3 h): ``CFL = 1.42`` at 10 m/s, and 4% of all cell-times exceed 1.
    On AI-DUST's *own* 0.25 deg / 3 h grid the same wind gives ``CFL = 3.9`` -- the
    published reference is unconditionally unstable at its published resolution.
    It survives in their pipeline only because the output is clamped to
    ``[0, 2 C_ref]`` and ``C_ref`` is floored at zero, which hides the blow-up
    rather than preventing it.

    The semi-Lagrangian form is unconditionally stable at any Courant number:
    instead of stepping the field forward, back-trace each cell to the departure
    point its air parcel came from and interpolate there,

        C_ref(x) = C_t(x - V dt) + (E/H) dt

    This is the same operator validated for the 1 km forecaster in
    ``experiments/riyadh_1km/forecast_model.py``, where at CFL 1.2 it beat both
    the explicit residual and a purely data-driven rollout.

    Displacements are computed in grid-index units; the signed ``dy`` makes a
    descending-latitude grid come out right without a special case.
    """
    from scipy.ndimage import map_coordinates

    conc_phys = np.nan_to_num(np.asarray(conc_phys, dtype="float32"))
    if mask is not None:
        conc_phys = conc_phys * mask.astype("float32")

    n_t, n_rows, n_cols = conc_phys.shape
    rows, cols = np.meshgrid(np.arange(n_rows, dtype="float32"),
                             np.arange(n_cols, dtype="float32"), indexing="ij")

    # Grid-cell displacements over one step: rows follow the signed dy so that a
    # northward wind moves the parcel toward lower row indices on a N->S grid.
    disp_row = (np.asarray(v, dtype="float32") * dt) / dy
    disp_col = (np.asarray(u, dtype="float32") * dt) / np.asarray(dx, dtype="float32")

    out = np.empty_like(conc_phys)
    for t in range(n_t):
        coords = np.stack([rows - disp_row[t], cols - disp_col[t]])
        out[t] = map_coordinates(conc_phys[t], coords, order=1, mode="nearest")

    if emission_tendency is not None:
        out = out + np.asarray(emission_tendency, dtype="float32") * float(dt)
    if mask is not None:
        out = out * mask.astype("float32")
    return np.clip(out, 0.0, None).astype("float32")


REFERENCE_MODES = ("euler", "semilag")


def build_reference(mode, conc_phys, u, v, dx, dy, dt, emission_tendency=None,
                    divergence=True, mask=None):
    """Dispatch to the Euler (AI-DUST, faithful) or semi-Lagrangian (stable) reference."""
    if mode == "euler":
        return reference_concentration(conc_phys, u, v, dx, dy, dt,
                                       emission_tendency=emission_tendency,
                                       divergence=divergence, mask=mask)
    if mode == "semilag":
        return semilagrangian_reference(conc_phys, u, v, dx, dy, dt,
                                        emission_tendency=emission_tendency, mask=mask)
    raise ValueError(f"reference mode must be one of {REFERENCE_MODES}, got {mode!r}")


def courant_number(u, v, dx, dy, dt):
    """Diagnostic: max(|u|dt/dx, |v|dt/dy). Values >= 1 invalidate the Euler reference."""
    cx = np.abs(np.asarray(u, dtype="float32")) * dt / np.asarray(dx, dtype="float32")
    cy = np.abs(np.asarray(v, dtype="float32")) * dt / abs(float(dy))
    return np.maximum(cx, cy).astype("float32")


def to_log(conc_phys):
    """Physical ug/m^3 -> the log1p space the models learn in."""
    return np.log1p(np.clip(np.asarray(conc_phys, dtype="float32"), 0.0, None)).astype("float32")


def to_physical(conc_log):
    """log1p space -> non-negative physical ug/m^3 (inverse of :func:`to_log`)."""
    if isinstance(conc_log, np.ndarray):
        return np.expm1(np.clip(conc_log, 0.0, None)).astype("float32")
    import torch
    return torch.expm1(torch.clamp(conc_log, min=0.0))
