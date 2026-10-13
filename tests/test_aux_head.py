"""Is the CAMS head a SIBLING of the station head, or secretly a pipeline?

The whole case for the auxiliary head rests on one structural property: the
station head must read the shared backbone, never the CAMS head's output. If it
read the CAMS prediction, a damped field would propagate straight into the
severity forecast -- which is the failure the design exists to avoid.

That is testable. Perturb the CAMS head's weights and the station logits must not
move at all; perturb the shared backbone and both must move.

Run:  pytest tests/test_aux_head.py
"""
import torch

from src.models.dust_transport import DustTransportGNN
from src.utils.graph import build_dust_graph

H = W = 6
N_NODES, IN_CH, N_OUT = H * W, 9, 4


def _build():
    """Graph, input, and the models without and with the aux head, from seed 0.

    Every test rebuilds rather than sharing one model: the later checks change
    weights with noise drawn from the global RNG, so a shared model would make
    the numbers depend on which tests ran first.
    """
    torch.manual_seed(0)
    edge_index, pos = build_dust_graph(H, W)[:2]
    x = torch.randn(N_NODES, IN_CH)
    plain = DustTransportGNN(IN_CH, N_OUT, [3, 4], hidden=16, hops=2, dropout=0.0)
    m = DustTransportGNN(IN_CH, N_OUT, [3, 4], hidden=16, hops=2, dropout=0.0,
                         aux_out=2)
    return plain.eval(), m.eval(), x, edge_index, pos


def _wreck_aux_head(m):
    with torch.no_grad():
        m.aux_head.weight.add_(torch.randn_like(m.aux_head.weight) * 10.0)
        m.aux_head.bias.add_(torch.randn_like(m.aux_head.bias) * 10.0)


def _perturb_backbone(m):
    with torch.no_grad():
        m.encode[0].weight.add_(torch.randn_like(m.encode[0].weight) * 0.5)


# ---- off by default: shape and return type unchanged ----------------------- #
def test_aux_off_returns_a_single_tensor_as_before():
    plain, _, x, edge_index, pos = _build()
    out = plain(x, edge_index, pos)
    assert torch.is_tensor(out) and out.shape == (N_NODES, N_OUT), (
        f"type {type(out).__name__}, shape {tuple(getattr(out, 'shape', ()))}")


# ---- on: two heads, aux covers EVERY node ---------------------------------- #
def test_aux_head_predicts_every_node_not_just_the_core():
    _, m, x, edge_index, pos = _build()
    main, aux = m(x, edge_index, pos)
    assert aux.shape == (N_NODES, 2) and main.shape == (N_NODES, N_OUT), (
        f"station logits {tuple(main.shape)}, CAMS {tuple(aux.shape)}, "
        f"expected CAMS over all {N_NODES} nodes")


# ---- THE structural test: siblings, not a pipeline ------------------------- #
# A damped CAMS prediction must not be able to propagate into severity.
def test_wrecking_cams_head_leaves_station_logits_exactly_unchanged():
    _, m, x, edge_index, pos = _build()
    before = m(x, edge_index, pos)[0].clone()
    _wreck_aux_head(m)
    after = m(x, edge_index, pos)[0]
    assert torch.equal(before, after), (
        f"station logits moved by up to {(after - before).abs().max():.2e}")


# ---- but the shared backbone does drive both ------------------------------- #
# Each check below starts from the state the previous one left, replayed in the
# same order, so the noise drawn and the numbers match the original audit.
def test_perturbing_shared_backbone_moves_both_heads():
    _, m, x, edge_index, pos = _build()
    _wreck_aux_head(m)
    b_main, b_aux = (t.clone() for t in m(x, edge_index, pos))
    _perturb_backbone(m)
    a_main, a_aux = m(x, edge_index, pos)
    moved_main = (a_main - b_main).abs().max().item()
    moved_aux = (a_aux - b_aux).abs().max().item()
    assert moved_main > 1e-6 and moved_aux > 1e-6, (
        f"station moved {moved_main:.3e}, CAMS moved {moved_aux:.3e} "
        f"-- they should share a representation")


# ---- gradients from the aux loss reach the backbone ------------------------ #
def _aux_loss_grads():
    """(encoder grad, station head grad) under the CAMS loss alone."""
    _, m, x, edge_index, pos = _build()
    _wreck_aux_head(m)
    _perturb_backbone(m)
    m.zero_grad()
    _, aux = m(x, edge_index, pos)
    aux.pow(2).mean().backward()          # CAMS loss alone
    return m.encode[0].weight.grad, m.head.weight.grad


# This is how 251 dense nodes teach a model that only sees 11 labelled ones.
def test_cams_loss_trains_the_shared_backbone():
    g_enc, _ = _aux_loss_grads()
    assert g_enc is not None and g_enc.abs().max() > 0, (
        f"encoder grad {'None' if g_enc is None else f'{g_enc.abs().max():.3e}'}")


# The station head grad must be zero under the auxiliary loss alone, so the
# weak label cannot rewrite the answer layer.
def test_cams_loss_does_not_reach_the_station_head():
    _, g_head = _aux_loss_grads()
    assert g_head is None or g_head.abs().max() == 0, (
        f"station head grad {g_head.abs().max():.3e} under the CAMS loss alone")
