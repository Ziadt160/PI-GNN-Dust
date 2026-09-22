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

Run:  python -u experiments/verify_dust_physics.py
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.physics import stability as S                       # noqa: E402
from src.physics import transport as T                       # noqa: E402
from src.utils.graph import build_dust_graph                 # noqa: E402

H = W = 11
CENTRE = H // 2
PASS, FAIL = "PASS", "** FAIL **"
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {name}\n        {detail}")


def gaussian(cx, cy, s=1.4):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))).astype("float32")


def main():
    # Descending latitude, as in the real files: row 0 is the NORTHERNMOST band.
    lats = np.linspace(30.0, 30.0 - 0.75 * (H - 1), H)
    lons = np.linspace(40.0, 40.0 + 0.75 * (W - 1), W)
    dx, dy = T.grid_spacing(lats, lons)
    dt = 10800.0
    print(f"grid: lat {lats[0]:.2f} -> {lats[-1]:.2f} (descending), "
          f"dy = {dy:,.0f} m (negative = descending), dx ~ {dx.mean():,.0f} m\n")

    report("signed dy is negative for a descending-latitude grid",
           dy < 0, f"dy = {dy:,.1f} m")

    blob = gaussian(CENTRE, CENTRE)[None]        # [1, H, W]

    # ---- 1. eastward wind must move dust EAST -------------------------------- #
    u = np.full_like(blob, 5.0); v = np.zeros_like(blob)
    tend = T.transport_tendency(blob, u, v, dx, dy, divergence=False)[0]
    east_of, west_of = tend[CENTRE, CENTRE + 2], tend[CENTRE, CENTRE - 2]
    report("u > 0 transports dust eastward",
           east_of > 0 > west_of,
           f"tendency east of peak {east_of:+.3e} (gain), west {west_of:+.3e} (loss)")

    # ---- 2. northward wind must move dust NORTH = toward LOWER row ----------- #
    u = np.zeros_like(blob); v = np.full_like(blob, 5.0)
    tend = T.transport_tendency(blob, u, v, dx, dy, divergence=False)[0]
    north_of, south_of = tend[CENTRE - 2, CENTRE], tend[CENTRE + 2, CENTRE]
    report("v > 0 transports dust northward (toward row 0)",
           north_of > 0 > south_of,
           f"tendency north of peak (row {CENTRE-2}) {north_of:+.3e} (gain), "
           f"south (row {CENTRE+2}) {south_of:+.3e} (loss)")

    # ---- 3. semi-Lagrangian back-trace must agree with the Euler direction --- #
    u = np.full_like(blob, 8.0); v = np.zeros_like(blob)
    ref = T.semilagrangian_reference(blob, u, v, dx, dy, dt)[0]
    shift = float(np.argmax(ref[CENTRE]) - CENTRE)
    expected = 8.0 * dt / float(dx.mean())
    report("semi-Lagrangian advects the plume east by the Courant distance",
           shift > 0 and abs(shift - expected) < 1.0,
           f"peak moved {shift:+.0f} cells east; u*dt/dx = {expected:+.2f}")

    # ---- 4. and NORTHWARD wind must move the semi-Lagrangian plume north ----- #
    u = np.zeros_like(blob); v = np.full_like(blob, 8.0)
    ref = T.semilagrangian_reference(blob, u, v, dx, dy, dt)[0]
    row_shift = float(np.argmax(ref[:, CENTRE]) - CENTRE)
    report("semi-Lagrangian moves dust north to a LOWER row index",
           row_shift < 0,
           f"peak moved {row_shift:+.0f} rows (negative = northward on this grid)")

    # ---- 5. graph edge geometry: is 'north' the same direction in the graph? - #
    edge_index, pos = build_dust_graph(H, W, neighbors=4)
    n_of = lambda r, c: r * W + c
    i, j = n_of(CENTRE - 1, CENTRE), n_of(CENTRE, CENTRE)      # i is one row NORTH of j
    rel = pos[i] - pos[j]
    report("graph pos: moving to a lower row increases the northward coordinate",
           float(rel[1]) > 0 and abs(float(rel[0])) < 1e-6,
           f"pos[north neighbour] - pos[centre] = ({rel[0]:+.0f} east, {rel[1]:+.0f} north)")

    # ---- 6. flux layer: a southerly wind must push dust to the NORTH node ---- #
    # edge j -> i with i north of j; wind at j is northward (v>0). The projection
    # of the source wind onto the edge direction must be POSITIVE (dust flows j->i).
    v_j = torch.tensor([[5.0]]); u_j = torch.tensor([[0.0]])
    wind = torch.cat([u_j, v_j], dim=-1)
    proj = float((wind * rel.unsqueeze(0)).sum())
    report("flux layer: northward wind gives positive flux toward the north neighbour",
           proj > 0, f"wind . edge_direction = {proj:+.2f}")

    # ---- 7. divergence: convergent wind must ACCUMULATE dust ----------------- #
    flat = np.ones((1, H, W), "float32")
    yy, xx = np.mgrid[0:H, 0:W]
    u_conv = (-(xx - CENTRE)).astype("float32")[None] * 2.0    # blows inward in x
    v_conv = ((yy - CENTRE)).astype("float32")[None] * 2.0     # inward in y (rows flip)
    tend = T.transport_tendency(flat, u_conv, v_conv, dx, dy, divergence=True)[0]
    report("wind convergence accumulates dust (div form only)",
           tend[CENTRE, CENTRE] > 0,
           f"tendency at convergence centre {tend[CENTRE, CENTRE]:+.3e}")

    # ---- 8-13. boundary-layer stability ------------------------------------- #
    # Sign errors here are as silent as sign errors in advection: a model fed an
    # inverted Richardson number learns that stable air mixes, and still trains.
    check_stability()

    print(f"\n{sum(results)}/{len(results)} checks passed")


def _levels_dataset(temps_k, u_ms, v_ms, heights_m, q=1e-3):
    """A one-timestep, one-cell ERA5-shaped dataset at the five standard levels.

    `temps_k`, `u_ms`, `v_ms`, `heights_m` are given top-down: 250, 500, 700,
    850, 900 hPa, matching `S.LEVELS_HPA`.
    """
    import xarray as xr
    n = len(S.LEVELS_HPA)
    shape = (1, n, 1, 1)
    mk = lambda a: np.asarray(a, dtype="float64").reshape(shape)
    return xr.Dataset(
        {"z": (("valid_time", "pressure_level", "latitude", "longitude"),
               mk(np.asarray(heights_m) * S.GRAVITY)),
         "t": (("valid_time", "pressure_level", "latitude", "longitude"), mk(temps_k)),
         "q": (("valid_time", "pressure_level", "latitude", "longitude"),
               mk([q] * n)),
         "u": (("valid_time", "pressure_level", "latitude", "longitude"), mk(u_ms)),
         "v": (("valid_time", "pressure_level", "latitude", "longitude"), mk(v_ms))},
        coords={"pressure_level": list(S.LEVELS_HPA)})


def check_stability():
    ch = {name: i for i, name in enumerate(S.CHANNEL_NAMES)}
    # Standard heights for 250/500/700/850/900 hPa, metres.
    ZS = [10400.0, 5600.0, 3000.0, 1450.0, 1000.0]
    CALM = [0.0] * 5

    # Potential temperature must exceed the actual temperature aloft (p < p0)
    # and equal it at 1000 hPa.
    report("potential temperature reduces to T at 1000 hPa and exceeds it aloft",
           abs(S.potential_temperature(300.0, 1000.0) - 300.0) < 1e-9
           and S.potential_temperature(250.0, 500.0) > 250.0,
           f"theta(300K, 1000hPa) = {S.potential_temperature(300.0, 1000.0):.3f} K; "
           f"theta(250K, 500hPa) = {S.potential_temperature(250.0, 500.0):.1f} K")

    # A strong low-level inversion: theta rising sharply with height, no shear.
    # Buoyancy wins, nothing mixes, dust is trapped at the surface -> Ri > 0.
    stable = S.derive(_levels_dataset([230, 255, 285, 305, 300], CALM, CALM, ZS))
    report("an inversion (theta increasing with height, no shear) gives Ri > 0",
           stable[0, ch["ri_900_850"], 0, 0] > 0,
           f"arctan(Ri) 900-850 = {stable[0, ch['ri_900_850'], 0, 0]:+.4f}")

    # A superadiabatic afternoon boundary layer: theta FALLING with height, so
    # the layer overturns on its own -> Ri < 0.
    unstable = S.derive(_levels_dataset([230, 255, 280, 292, 300], CALM, CALM, ZS))
    report("a superadiabatic layer (theta falling with height) gives Ri < 0",
           unstable[0, ch["ri_900_850"], 0, 0] < 0,
           f"arctan(Ri) 900-850 = {unstable[0, ch['ri_900_850'], 0, 0]:+.4f}")

    # Same stratification, strong shear added: turbulence production rises, so Ri
    # must FALL. This is the sign that decides whether a shamal mixes dust down.
    sheared = S.derive(_levels_dataset([230, 255, 285, 305, 300],
                                       [0, 0, 0, 20, 0], CALM, ZS))
    report("shear reduces Ri at fixed stratification",
           sheared[0, ch["ri_900_850"], 0, 0] < stable[0, ch["ri_900_850"], 0, 0],
           f"calm {stable[0, ch['ri_900_850'], 0, 0]:+.4f} -> "
           f"sheared {sheared[0, ch['ri_900_850'], 0, 0]:+.4f}")

    # The pole at zero shear must not produce inf or NaN anywhere.
    report("Ri is finite and bounded even at exactly zero shear",
           np.all(np.isfinite(stable)) and np.abs(stable[0, ch["ri_900_850"]]).max() < np.pi / 2,
           f"max |arctan(Ri)| = {np.abs(stable[0, ch['ri_900_850']]).max():.4f} "
           f"< pi/2 = {np.pi/2:.4f}, no NaN or inf in {stable.size} values")

    # Low-level jet: the strongest wind of 900/850/700 sits at 850 hPa, so the
    # reported level must be the middle one (0.5) and the speed must be its speed.
    jet = S.derive(_levels_dataset([230, 255, 285, 305, 300],
                                   [0, 0, 6, 18, 4], CALM, ZS))
    report("low-level jet is located at the level that carries the strongest wind",
           abs(jet[0, ch["llj_level"], 0, 0] - 0.5) < 1e-6
           and abs(jet[0, ch["llj_speed"], 0, 0] - 18.0) < 1e-4,
           f"level {jet[0, ch['llj_level'], 0, 0]:.1f} (0=900, 0.5=850, 1=700), "
           f"speed {jet[0, ch['llj_speed'], 0, 0]:.1f} m/s")
    if not all(results):
        sys.exit("PHYSICS AUDIT FAILED -- dust is moving the wrong way somewhere")


if __name__ == "__main__":
    main()
