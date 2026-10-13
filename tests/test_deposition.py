"""Does the removal law actually remove dust, and in the right direction?

`keep = exp(-(v_dry + lambda*P) * dt / H)` replaces a single global scalar. Every
sign in it is silent if wrong: a model whose rain term has the wrong sign still
trains, still converges, and simply believes rain creates dust.

Four directions must hold, and none of them is enforced by the optimiser:

    more rain          -> less dust survives
    deeper mixing      -> more survives (the same removal thins a deeper layer
                          more slowly)
    longer hop         -> less survives
    always in (0, 1]   -> a survival fraction is a fraction

Run:  pytest tests/test_deposition.py
"""
import pytest
import torch

from src.models.dust_transport import DustTransportGNN

TP, BLH = 6, 7          # channel slots used by the toy feature vector


def make(dt_hop=10800.0, hidden=8):
    torch.manual_seed(0)
    m = DustTransportGNN(12, 4, [3, 4], hidden=hidden, hops=2, dropout=0.0,
                         deposition_idx=[TP, BLH], dt_hop=dt_hop)
    # Push the rates up from their near-zero init so the effects are visible.
    with torch.no_grad():
        m.v_dry.fill_(-1.0)
        m.scavenge.fill_(1.0)
    return m


def feats(precip_mm, blh_m, n=5):
    """One toy node vector; precipitation is stored as log1p(mm), like the field."""
    x = torch.zeros(n, 12)
    x[:, TP] = torch.log1p(torch.tensor(float(precip_mm)))
    x[:, BLH] = float(blh_m)
    return x


@pytest.fixture(scope="module")
def model():
    return make()


# Scavenging is multiplicative, which the old additive feature could not express.
def test_rain_removes_dust(model):
    dry = model._survival(feats(0.0, 1000.0)).mean().item()
    wet = model._survival(feats(5.0, 1000.0)).mean().item()
    assert wet < dry, f"0 mm keeps {dry:.4f}, 5 mm keeps {wet:.4f}"


def test_deeper_mixed_layer_dilutes_removal(model):
    shallow = model._survival(feats(0.0, 200.0)).mean().item()
    deep = model._survival(feats(0.0, 3000.0)).mean().item()
    assert deep > shallow, f"200 m keeps {shallow:.4f}, 3000 m keeps {deep:.4f}"


def test_longer_hop_removes_more():
    short = make(dt_hop=3600.0)._survival(feats(0.0, 1000.0)).mean().item()
    long = make(dt_hop=21600.0)._survival(feats(0.0, 1000.0)).mean().item()
    assert long < short, f"1 h keeps {short:.4f}, 6 h keeps {long:.4f}"


def test_survival_stays_a_fraction_under_extremes(model):
    extreme = model._survival(feats(200.0, 50.0))
    none = model._survival(feats(0.0, 6000.0))
    assert bool((extreme > 0).all() and (extreme <= 1).all()
                and (none > 0).all() and (none <= 1).all()), (
        f"torrential+shallow {extreme.min():.2e}..{extreme.max():.2e}, "
        f"dry+deep {none.min():.4f}..{none.max():.4f}")


# Per-channel rates: the latent stand-in for coarse dust dropping out fast
# while fine dust lingers. A single shared scalar forced them to be equal.
# PM10 and PM2.5 settle orders of magnitude apart.
def test_hidden_channels_can_decay_at_different_rates():
    m = make()
    with torch.no_grad():
        m.v_dry[0] = 2.0          # one fast-settling component
        m.v_dry[1:] = -3.0        # the rest slow
    k = m._survival(feats(0.0, 1000.0))[0]
    assert k[0] < k[1] * 0.5, (
        f"fast channel keeps {k[0]:.4f}, slow channel keeps {k[1]:.4f}")


# Default path must be untouched: without --deposition the scalar ignores rain
# and depth exactly as before.
def test_without_deposition_old_global_scalar_is_used_unchanged():
    plain = DustTransportGNN(12, 4, [3, 4], hidden=8, hops=2, dropout=0.0)
    s = plain._survival(feats(99.0, 10.0))
    assert s.dim() == 0 and abs(s.item() - torch.sigmoid(plain.decay).item()) < 1e-9, (
        f"survival has shape {tuple(s.shape)}, value {s.flatten()[0].item():.4f}; "
        f"expected the scalar sigmoid(decay) = {torch.sigmoid(plain.decay).item():.4f}")
