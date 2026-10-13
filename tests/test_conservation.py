"""Does the conservative flux layer actually conserve mass?

The claim is structural, not statistical: for every face the flux out of one cell
equals the flux into its neighbour, so the transport term sums to exactly zero
over the graph. That is testable to floating-point precision, and it is the whole
reason for preferring this layer -- a conservation LOSS only discourages
violations, which this project already measured to be ineffective (the gated
physics loss fires on 0.1-0.3% of samples).

Run:  pytest tests/test_conservation.py
"""
from types import SimpleNamespace

import pytest
import torch

from src.layers.flux import ConservativeFluxLayer, PhysicsFluxLayer
from src.utils.graph import build_dust_graph

H = W = 7
N = H * W


@pytest.fixture(scope="module")
def toy():
    """A 7x7 graph with random features, wind and concentration, and both layers."""
    torch.manual_seed(0)
    edge_index, pos = build_dust_graph(H, W)[:2]
    x = torch.randn(N, 6)
    u, v = torch.randn(N, 1), torch.randn(N, 1)
    c = torch.rand(N, 1) * 5
    cons = ConservativeFluxLayer(6, 4)
    old = PhysicsFluxLayer(6, 4)
    return SimpleNamespace(edge_index=edge_index, pos=pos, x=x, u=u, v=v, c=c,
                           cons=cons, old=old)


# ---- the structural claim -------------------------------------------------- #
# Every face contributes +F and -F.
def test_conservative_layer_total_transport_over_graph_is_zero(toy):
    out = toy.cons(toy.x, toy.edge_index, toy.pos, toy.u, toy.v, toy.c)
    net = out.sum(0).abs().max().item()
    scale = out.abs().max().item()
    assert net < 1e-4 * max(scale, 1.0), (
        f"net |sum| = {net:.3e} against per-node magnitude {scale:.3e}")


# This is the gap being closed: dust can be created or destroyed freely.
def test_existing_layer_does_not_conserve(toy):
    net = toy.old(toy.x, toy.edge_index, toy.pos, toy.u, toy.v, toy.c).sum(0).abs().max().item()
    assert net > 1e-3, f"net |sum| = {net:.3e}"


# ---- antisymmetry, at the level where it is actually defined --------------- #
# Per EDGE, not per node. Reversing the graph also changes which node each
# message is aggregated at, so the per-node sums are not negatives of each
# other and comparing them proves nothing. What must hold is F(j->i) =
# -F(i->j) for one face, which is exactly what makes the graph sum vanish.
def test_flux_across_one_face_is_antisymmetric(toy):
    x, pos, u, v, c = toy.x, toy.pos, toy.u, toy.v, toy.c
    src, dst = toy.edge_index[0], toy.edge_index[1]   # PyG: _j is source, _i is target
    f_ji = toy.cons.message(x[dst], x[src], pos[dst], pos[src],
                            u[dst], u[src], v[dst], v[src], c[dst], c[src])
    f_ij = toy.cons.message(x[src], x[dst], pos[src], pos[dst],
                            u[src], u[dst], v[src], v[dst], c[src], c[dst])
    assert torch.allclose(f_ji, -f_ij, atol=1e-5), (
        f"max |F(j->i) + F(i->j)| = {(f_ji + f_ij).abs().max():.3e} "
        f"against flux magnitude {f_ji.abs().max():.3e}")


# ---- upwind, not downwind -------------------------------------------------- #
# Eastward wind everywhere, a single spike: dust must move east (+x), so the
# cell EAST of the spike gains and the cell WEST of it does not.
def test_eastward_wind_moves_dust_east_not_west(toy):
    flat = torch.zeros(N, 1)
    mid = (H // 2) * W + (W // 2)
    flat[mid] = 10.0
    ue, ve = torch.ones(N, 1) * 5.0, torch.zeros(N, 1)
    f = toy.cons(torch.ones(N, 6), toy.edge_index, toy.pos, ue, ve, flat)
    east, west = f[mid + 1].sum().item(), f[mid - 1].sum().item()
    assert abs(east) > abs(west), (
        f"cell east of the spike |flux| {abs(east):.3e}, cell west {abs(west):.3e}")


# ---- zero wind moves nothing ----------------------------------------------- #
def test_no_wind_transports_no_dust(toy):
    z = toy.cons(toy.x, toy.edge_index, toy.pos, torch.zeros(N, 1), torch.zeros(N, 1), toy.c)
    assert z.abs().max().item() < 1e-6, f"max |flux| = {z.abs().max():.3e} with u = v = 0"
