"""Does the conservative flux layer actually conserve mass?

The claim is structural, not statistical: for every face the flux out of one cell
equals the flux into its neighbour, so the transport term sums to exactly zero
over the graph. That is testable to floating-point precision, and it is the whole
reason for preferring this layer -- a conservation LOSS only discourages
violations, which this project already measured to be ineffective (the gated
physics loss fires on 0.1-0.3% of samples).

Run:  python -u experiments/verify_conservation.py
"""
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.layers.flux import ConservativeFluxLayer, PhysicsFluxLayer   # noqa: E402
from src.utils.graph import build_dust_graph                          # noqa: E402

PASS, FAIL = "PASS", "** FAIL **"
results = []


def report(name, ok, detail):
    results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}\n        {detail}")


def main():
    torch.manual_seed(0)
    H = W = 7
    n = H * W
    edge_index, pos = build_dust_graph(H, W)[:2]
    x = torch.randn(n, 6)
    u, v = torch.randn(n, 1), torch.randn(n, 1)
    c = torch.rand(n, 1) * 5

    cons = ConservativeFluxLayer(6, 4)
    old = PhysicsFluxLayer(6, 4)

    # ---- the structural claim ---------------------------------------------- #
    net_c = cons(x, edge_index, pos, u, v, c).sum(0).abs().max().item()
    scale_c = cons(x, edge_index, pos, u, v, c).abs().max().item()
    report("conservative layer: total transport over the graph is zero",
           net_c < 1e-4 * max(scale_c, 1.0),
           f"net |sum| = {net_c:.3e} against per-node magnitude {scale_c:.3e} "
           f"-- every face contributes +F and -F")

    net_o = old(x, edge_index, pos, u, v, c).sum(0).abs().max().item()
    report("the existing layer does NOT conserve (this is the gap being closed)",
           net_o > 1e-3,
           f"net |sum| = {net_o:.3e} -- dust can be created or destroyed freely")

    # ---- antisymmetry, at the level where it is actually defined ----------- #
    # Per EDGE, not per node. Reversing the graph also changes which node each
    # message is aggregated at, so the per-node sums are not negatives of each
    # other and comparing them proves nothing. What must hold is F(j->i) =
    # -F(i->j) for one face, which is exactly what makes the graph sum vanish.
    src, dst = edge_index[0], edge_index[1]        # PyG: _j is source, _i is target
    f_ji = cons.message(x[dst], x[src], pos[dst], pos[src],
                        u[dst], u[src], v[dst], v[src], c[dst], c[src])
    f_ij = cons.message(x[src], x[dst], pos[src], pos[dst],
                        u[src], u[dst], v[src], v[dst], c[src], c[dst])
    report("the flux across one face is antisymmetric: F(j->i) = -F(i->j)",
           torch.allclose(f_ji, -f_ij, atol=1e-5),
           f"max |F(j->i) + F(i->j)| = {(f_ji + f_ij).abs().max():.3e} "
           f"against flux magnitude {f_ji.abs().max():.3e}")

    # ---- upwind, not downwind ---------------------------------------------- #
    # Eastward wind everywhere, a single spike: dust must move east (+x), so the
    # cell EAST of the spike gains and the cell WEST of it does not.
    flat = torch.zeros(n, 1)
    mid = (H // 2) * W + (W // 2)
    flat[mid] = 10.0
    ue, ve = torch.ones(n, 1) * 5.0, torch.zeros(n, 1)
    f = cons(torch.ones(n, 6), edge_index, pos, ue, ve, flat)
    east, west = f[mid + 1].sum().item(), f[mid - 1].sum().item()
    report("an eastward wind moves dust east, not west",
           abs(east) > abs(west),
           f"cell east of the spike |flux| {abs(east):.3e}, "
           f"cell west {abs(west):.3e}")

    # ---- zero wind moves nothing ------------------------------------------- #
    z = cons(x, edge_index, pos, torch.zeros(n, 1), torch.zeros(n, 1), c)
    report("no wind transports no dust",
           z.abs().max().item() < 1e-6,
           f"max |flux| = {z.abs().max():.3e} with u = v = 0")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    if not all(results):
        sys.exit("CONSERVATION AUDIT FAILED")


if __name__ == "__main__":
    main()
