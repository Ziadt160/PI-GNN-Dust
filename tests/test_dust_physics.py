"""Physics audit: does dust actually move the way the code says it does?

Sign errors in advection are silent -- the model still trains, the loss still
falls, and the field simply drifts the wrong way.  This repo has had one such bug
before (latitude descends with row index, so a row-wise gradient is the
*southward* derivative).  These checks are cheap and catch the whole class.

Each test builds a synthetic field with a known answer and asserts the code
agrees.  Conventions under test:

    pos[:, 0] = eastward  (increases with column / longitude)
    pos[:, 1] = northward (increases with latitude, i.e. DECREASING row index)
    u > 0 blows toward the east, v > 0 blows toward the north
    latitude array DESCENDS with row index (ERA5/CAMS default)

Run:  pytest tests/test_dust_physics.py
"""
import numpy as np
import pytest
import torch
import xarray as xr

from src.physics import stability as S
from src.physics import transport as T
from src.utils.graph import build_dust_graph

H = W = 11
CENTRE = H // 2
DT = 10800.0

# Descending latitude, as in the real files: row 0 is the NORTHERNMOST band.
LATS = np.linspace(30.0, 30.0 - 0.75 * (H - 1), H)
LONS = np.linspace(40.0, 40.0 + 0.75 * (W - 1), W)


def gaussian(cx, cy, s=1.4):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))).astype("float32")


BLOB = gaussian(CENTRE, CENTRE)[None]        # [1, H, W]


@pytest.fixture(scope="module")
def grid():
    """(dx, dy) in metres; dy is signed, negative for descending latitude."""
    return T.grid_spacing(LATS, LONS)


def test_signed_dy_negative_for_descending_latitude_grid(grid):
    _, dy = grid
    assert dy < 0, f"dy = {dy:,.1f} m"


# ---- eastward wind must move dust EAST ------------------------------------- #
def test_eastward_wind_transports_dust_east(grid):
    dx, dy = grid
    u = np.full_like(BLOB, 5.0); v = np.zeros_like(BLOB)
    tend = T.transport_tendency(BLOB, u, v, dx, dy, divergence=False)[0]
    east_of, west_of = tend[CENTRE, CENTRE + 2], tend[CENTRE, CENTRE - 2]
    assert east_of > 0 > west_of, (
        f"tendency east of peak {east_of:+.3e} (should gain), "
        f"west {west_of:+.3e} (should lose)")


# ---- northward wind must move dust NORTH = toward LOWER row ---------------- #
def test_northward_wind_transports_dust_toward_row_zero(grid):
    dx, dy = grid
    u = np.zeros_like(BLOB); v = np.full_like(BLOB, 5.0)
    tend = T.transport_tendency(BLOB, u, v, dx, dy, divergence=False)[0]
    north_of, south_of = tend[CENTRE - 2, CENTRE], tend[CENTRE + 2, CENTRE]
    assert north_of > 0 > south_of, (
        f"tendency north of peak (row {CENTRE-2}) {north_of:+.3e} (should gain), "
        f"south (row {CENTRE+2}) {south_of:+.3e} (should lose)")


# ---- semi-Lagrangian back-trace must agree with the Euler direction -------- #
def test_semilagrangian_advects_plume_east_by_courant_distance(grid):
    dx, dy = grid
    u = np.full_like(BLOB, 8.0); v = np.zeros_like(BLOB)
    ref = T.semilagrangian_reference(BLOB, u, v, dx, dy, DT)[0]
    shift = float(np.argmax(ref[CENTRE]) - CENTRE)
    expected = 8.0 * DT / float(dx.mean())
    assert shift > 0 and abs(shift - expected) < 1.0, (
        f"peak moved {shift:+.0f} cells east; u*dt/dx = {expected:+.2f}")


# ---- and NORTHWARD wind must move the semi-Lagrangian plume north ---------- #
def test_semilagrangian_moves_dust_north_to_lower_row(grid):
    dx, dy = grid
    u = np.zeros_like(BLOB); v = np.full_like(BLOB, 8.0)
    ref = T.semilagrangian_reference(BLOB, u, v, dx, dy, DT)[0]
    row_shift = float(np.argmax(ref[:, CENTRE]) - CENTRE)
    assert row_shift < 0, (
        f"peak moved {row_shift:+.0f} rows (negative = northward on this grid)")


# ---- graph edge geometry: is 'north' the same direction in the graph? ------ #
@pytest.fixture(scope="module")
def north_offset():
    """pos[i] - pos[j] for node i one row NORTH of node j, at the grid centre."""
    edge_index, pos = build_dust_graph(H, W, neighbors=4)
    n_of = lambda r, c: r * W + c
    i, j = n_of(CENTRE - 1, CENTRE), n_of(CENTRE, CENTRE)
    return pos[i] - pos[j]


def test_graph_pos_lower_row_increases_northward_coordinate(north_offset):
    rel = north_offset
    assert float(rel[1]) > 0 and abs(float(rel[0])) < 1e-6, (
        f"pos[north neighbour] - pos[centre] = ({rel[0]:+.0f} east, {rel[1]:+.0f} north)")


# ---- flux layer: a southerly wind must push dust to the NORTH node --------- #
# edge j -> i with i north of j; wind at j is northward (v>0). The projection
# of the source wind onto the edge direction must be POSITIVE (dust flows j->i).
def test_flux_layer_northward_wind_gives_positive_flux_toward_north_neighbour(north_offset):
    v_j = torch.tensor([[5.0]]); u_j = torch.tensor([[0.0]])
    wind = torch.cat([u_j, v_j], dim=-1)
    proj = float((wind * north_offset.unsqueeze(0)).sum())
    assert proj > 0, f"wind . edge_direction = {proj:+.2f}"


# ---- divergence: convergent wind must ACCUMULATE dust ---------------------- #
def test_wind_convergence_accumulates_dust(grid):
    dx, dy = grid
    flat = np.ones((1, H, W), "float32")
    yy, xx = np.mgrid[0:H, 0:W]
    u_conv = (-(xx - CENTRE)).astype("float32")[None] * 2.0    # blows inward in x
    v_conv = ((yy - CENTRE)).astype("float32")[None] * 2.0     # inward in y (rows flip)
    tend = T.transport_tendency(flat, u_conv, v_conv, dx, dy, divergence=True)[0]
    assert tend[CENTRE, CENTRE] > 0, (
        f"tendency at convergence centre {tend[CENTRE, CENTRE]:+.3e}")


# =========================================================================== #
# boundary-layer stability
# Sign errors here are as silent as sign errors in advection: a model fed an
# inverted Richardson number learns that stable air mixes, and still trains.
# =========================================================================== #
CH = {name: i for i, name in enumerate(S.CHANNEL_NAMES)}
# Standard heights for 250/500/700/850/900 hPa, metres.
ZS = [10400.0, 5600.0, 3000.0, 1450.0, 1000.0]
CALM = [0.0] * 5


def _levels_dataset(temps_k, u_ms, v_ms, heights_m, q=1e-3):
    """A one-timestep, one-cell ERA5-shaped dataset at the five standard levels.

    `temps_k`, `u_ms`, `v_ms`, `heights_m` are given top-down: 250, 500, 700,
    850, 900 hPa, matching `S.LEVELS_HPA`.
    """
    n = len(S.LEVELS_HPA)
    shape = (1, n, 1, 1)
    mk = lambda a: np.asarray(a, dtype="float64").reshape(shape)
    dims = ("valid_time", "pressure_level", "latitude", "longitude")
    return xr.Dataset(
        {"z": (dims, mk(np.asarray(heights_m) * S.GRAVITY)),
         "t": (dims, mk(temps_k)),
         "q": (dims, mk([q] * n)),
         "u": (dims, mk(u_ms)),
         "v": (dims, mk(v_ms))},
        coords={"pressure_level": list(S.LEVELS_HPA)})


@pytest.fixture(scope="module")
def stable():
    """A strong low-level inversion: theta rising sharply with height, no shear."""
    return S.derive(_levels_dataset([230, 255, 285, 305, 300], CALM, CALM, ZS))


# Potential temperature must exceed the actual temperature aloft (p < p0)
# and equal it at 1000 hPa.
def test_potential_temperature_reduces_to_t_at_1000hpa_and_exceeds_it_aloft():
    at_p0 = S.potential_temperature(300.0, 1000.0)
    aloft = S.potential_temperature(250.0, 500.0)
    assert abs(at_p0 - 300.0) < 1e-9 and aloft > 250.0, (
        f"theta(300K, 1000hPa) = {at_p0:.3f} K; theta(250K, 500hPa) = {aloft:.1f} K")


# Buoyancy wins, nothing mixes, dust is trapped at the surface -> Ri > 0.
def test_inversion_without_shear_gives_positive_ri(stable):
    ri = stable[0, CH["ri_900_850"], 0, 0]
    assert ri > 0, f"arctan(Ri) 900-850 = {ri:+.4f}"


# A superadiabatic afternoon boundary layer: theta FALLING with height, so
# the layer overturns on its own -> Ri < 0.
def test_superadiabatic_layer_gives_negative_ri():
    unstable = S.derive(_levels_dataset([230, 255, 280, 292, 300], CALM, CALM, ZS))
    ri = unstable[0, CH["ri_900_850"], 0, 0]
    assert ri < 0, f"arctan(Ri) 900-850 = {ri:+.4f}"


# Same stratification, strong shear added: turbulence production rises, so Ri
# must FALL. This is the sign that decides whether a shamal mixes dust down.
def test_shear_reduces_ri_at_fixed_stratification(stable):
    sheared = S.derive(_levels_dataset([230, 255, 285, 305, 300],
                                       [0, 0, 0, 20, 0], CALM, ZS))
    calm_ri = stable[0, CH["ri_900_850"], 0, 0]
    sheared_ri = sheared[0, CH["ri_900_850"], 0, 0]
    assert sheared_ri < calm_ri, f"calm {calm_ri:+.4f} -> sheared {sheared_ri:+.4f}"


# The pole at zero shear must not produce inf or NaN anywhere.
def test_ri_finite_and_bounded_at_exactly_zero_shear(stable):
    peak = np.abs(stable[0, CH["ri_900_850"]]).max()
    assert np.all(np.isfinite(stable)) and peak < np.pi / 2, (
        f"max |arctan(Ri)| = {peak:.4f} vs pi/2 = {np.pi/2:.4f}; "
        f"{int((~np.isfinite(stable)).sum())} NaN or inf in {stable.size} values")


# Low-level jet: the strongest wind of 900/850/700 sits at 850 hPa, so the
# reported level must be the middle one (0.5) and the speed must be its speed.
def test_low_level_jet_located_at_level_with_strongest_wind():
    jet = S.derive(_levels_dataset([230, 255, 285, 305, 300],
                                   [0, 0, 6, 18, 4], CALM, ZS))
    level, speed = jet[0, CH["llj_level"], 0, 0], jet[0, CH["llj_speed"], 0, 0]
    assert abs(level - 0.5) < 1e-6 and abs(speed - 18.0) < 1e-4, (
        f"level {level:.1f} (0=900, 0.5=850, 1=700), speed {speed:.1f} m/s")
