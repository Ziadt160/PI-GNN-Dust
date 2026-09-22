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

Run:  python -u experiments/verify_deposition.py
"""
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.dust_transport import DustTransportGNN     # noqa: E402

PASS, FAIL = "PASS", "** FAIL **"
results = []
TP, BLH = 6, 7          # channel slots used by the toy feature vector


def report(name, ok, detail):
    results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}\n        {detail}")


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


def main():
    m = make()

    dry = m._survival(feats(0.0, 1000.0)).mean().item()
    wet = m._survival(feats(5.0, 1000.0)).mean().item()
    report("rain REMOVES dust (more precipitation -> less survives)",
           wet < dry,
           f"0 mm keeps {dry:.4f}, 5 mm keeps {wet:.4f} "
           f"-- scavenging is multiplicative, which the old additive feature "
           f"could not express")

    shallow = m._survival(feats(0.0, 200.0)).mean().item()
    deep = m._survival(feats(0.0, 3000.0)).mean().item()
    report("a deeper mixed layer dilutes removal (more survives)",
           deep > shallow,
           f"200 m keeps {shallow:.4f}, 3000 m keeps {deep:.4f}")

    short = make(dt_hop=3600.0)._survival(feats(0.0, 1000.0)).mean().item()
    long = make(dt_hop=21600.0)._survival(feats(0.0, 1000.0)).mean().item()
    report("a longer hop removes more",
           long < short, f"1 h keeps {short:.4f}, 6 h keeps {long:.4f}")

    extreme = m._survival(feats(200.0, 50.0))
    none = m._survival(feats(0.0, 6000.0))
    report("survival stays a fraction in (0, 1] under extremes",
           bool((extreme > 0).all() and (extreme <= 1).all()
                and (none > 0).all() and (none <= 1).all()),
           f"torrential+shallow {extreme.min():.2e}..{extreme.max():.2e}, "
           f"dry+deep {none.min():.4f}..{none.max():.4f}")

    # Per-channel rates: the latent stand-in for coarse dust dropping out fast
    # while fine dust lingers. A single shared scalar forced them to be equal.
    with torch.no_grad():
        m.v_dry[0] = 2.0          # one fast-settling component
        m.v_dry[1:] = -3.0        # the rest slow
    k = m._survival(feats(0.0, 1000.0))[0]
    report("different hidden channels can decay at different rates",
           k[0] < k[1] * 0.5,
           f"fast channel keeps {k[0]:.4f}, slow channel keeps {k[1]:.4f} "
           f"-- PM10 and PM2.5 settle orders of magnitude apart")

    # Default path must be untouched.
    plain = DustTransportGNN(12, 4, [3, 4], hidden=8, hops=2, dropout=0.0)
    s = plain._survival(feats(99.0, 10.0))
    report("without --deposition the old global scalar is used, unchanged",
           s.dim() == 0 and abs(s.item() - torch.sigmoid(plain.decay).item()) < 1e-9,
           f"scalar {s.item():.4f}, ignores rain and depth exactly as before")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    if not all(results):
        sys.exit("DEPOSITION AUDIT FAILED -- removal is not behaving physically")


if __name__ == "__main__":
    main()
