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

    print(f"\n{sum(results)}/{len(results)} checks passed")
    if not all(results):
        sys.exit("PHYSICS AUDIT FAILED -- dust is moving the wrong way somewhere")


if __name__ == "__main__":
    main()
