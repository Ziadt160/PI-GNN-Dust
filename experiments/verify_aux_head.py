"""Is the CAMS head a SIBLING of the station head, or secretly a pipeline?

The whole case for the auxiliary head rests on one structural property: the
station head must read the shared backbone, never the CAMS head's output. If it
read the CAMS prediction, a damped field would propagate straight into the
severity forecast -- which is the failure the design exists to avoid.

That is testable. Perturb the CAMS head's weights and the station logits must not
move at all; perturb the shared backbone and both must move.

Run:  python -u experiments/verify_aux_head.py
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.dust_transport import DustTransportGNN     # noqa: E402
from src.utils.graph import build_dust_graph               # noqa: E402

PASS, FAIL = "PASS", "** FAIL **"
results = []


def report(name, ok, detail):
    results.append(bool(ok))
    print(f"  [{PASS if ok else FAIL}] {name}\n        {detail}")


def main():
    torch.manual_seed(0)
    H = W = 6
    n_nodes, in_ch, n_out = H * W, 9, 4
    edge_index, pos = build_dust_graph(H, W)[:2]
    x = torch.randn(n_nodes, in_ch)

    # ---- 1. off by default: shape and return type unchanged ----------------- #
    plain = DustTransportGNN(in_ch, n_out, [3, 4], hidden=16, hops=2, dropout=0.0)
    plain.eval()
    out = plain(x, edge_index, pos)
    report("aux_out=0 returns a single tensor, as before",
           torch.is_tensor(out) and out.shape == (n_nodes, n_out),
           f"type {type(out).__name__}, shape {tuple(out.shape)}")

    # ---- 2. on: two heads, aux covers EVERY node ---------------------------- #
    m = DustTransportGNN(in_ch, n_out, [3, 4], hidden=16, hops=2, dropout=0.0,
                         aux_out=2)
    m.eval()
    main, aux = m(x, edge_index, pos)
    report("aux head returns a prediction at every node, not just the core",
           aux.shape == (n_nodes, 2) and main.shape == (n_nodes, n_out),
           f"station logits {tuple(main.shape)}, CAMS {tuple(aux.shape)} "
           f"over all {n_nodes} nodes")

    # ---- 3. THE structural test: siblings, not a pipeline ------------------- #
    before = m(x, edge_index, pos)[0].clone()
    with torch.no_grad():
        m.aux_head.weight.add_(torch.randn_like(m.aux_head.weight) * 10.0)
        m.aux_head.bias.add_(torch.randn_like(m.aux_head.bias) * 10.0)
    after = m(x, edge_index, pos)[0]
    report("wrecking the CAMS head leaves the station logits EXACTLY unchanged",
           torch.equal(before, after),
           "a damped CAMS prediction cannot propagate into severity; "
           f"max |change| = {(after - before).abs().max():.2e}")

    # ---- 4. but the shared backbone does drive both ------------------------- #
    b_main, b_aux = (t.clone() for t in m(x, edge_index, pos))
    with torch.no_grad():
        m.encode[0].weight.add_(torch.randn_like(m.encode[0].weight) * 0.5)
    a_main, a_aux = m(x, edge_index, pos)
    moved_main = (a_main - b_main).abs().max().item()
    moved_aux = (a_aux - b_aux).abs().max().item()
    report("perturbing the shared backbone moves BOTH heads",
           moved_main > 1e-6 and moved_aux > 1e-6,
           f"station moved {moved_main:.3e}, CAMS moved {moved_aux:.3e} "
           f"-- they genuinely share a representation")

    # ---- 5. gradients from the aux loss reach the backbone ------------------ #
    m.zero_grad()
    _, aux2 = m(x, edge_index, pos)
    aux2.pow(2).mean().backward()          # CAMS loss alone
    g_enc = m.encode[0].weight.grad
    g_head = m.head.weight.grad
    report("the CAMS loss trains the shared backbone",
           g_enc is not None and g_enc.abs().max() > 0,
           f"encoder grad {g_enc.abs().max():.3e} -- this is how 251 dense nodes "
           f"teach a model that only sees 11 labelled ones")
    report("the CAMS loss does NOT reach into the station head",
           g_head is None or g_head.abs().max() == 0,
           "station head grad is zero under the auxiliary loss alone, "
           "so the weak label cannot rewrite the answer layer")

    print(f"\n{sum(results)}/{len(results)} checks passed")
    if not all(results):
        sys.exit("AUX HEAD AUDIT FAILED -- the two heads are not wired as siblings")


if __name__ == "__main__":
    main()
