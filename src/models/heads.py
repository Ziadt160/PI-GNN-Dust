"""The unified prediction contract every model in the benchmark obeys.

Your model and AI-DUST both predict "**anchor + learned correction**".  They
differ only in *which* anchor and *what kind* of correction:

    anchor  A in { C_t                      persistence            (yours)
                   C_ref = C_t + (-div(VC) + E/H) dt   physics-advanced  (AI-DUST) }

    head       C = A + d                    additive               (yours)
               C = 2 sigma(z) . A           multiplicative bound   (AI-DUST Eq. 1)

                      additive A + d        multiplicative 2s(z).A
        A = C_t       SRC  (your model)     H1
        A = C_ref     H2                    AID  (AI-DUST)

Those two models are the diagonal of one 2x2; the off-diagonal entries are new
hybrids.  Making this an explicit axis is what stops "AI-DUST beats your model"
from silently meaning "AI-DUST used a different output parameterisation".

**Both heads zero-initialise to the anchor.**  The additive head starts at
``A + 0``; the multiplicative head starts at ``2 sigma(0) A = A``.  So every
model in the roster begins training from the same well-posed solution and the
learning curves are directly comparable.

Non-negativity: targets are ``log1p(ug/m^3) >= 0``.  The multiplicative head is
non-negative by construction (``A >= 0``, ``2 sigma > 0``).  The additive head can
go negative in log space, and -- as established during the earlier refactor -- is
deliberately *not* passed through Softplus, which would impose a spurious
``ln 2`` floor; physical non-negativity is applied on conversion instead.
"""
import torch
import torch.nn as nn

ANCHORS = ("persistence", "physics")
HEADS = ("additive", "multiplicative")


class UnifiedHead(nn.Module):
    """Maps a node embedding to the next-step state, given an anchor.

    Args:
        in_dim: width of the incoming node embedding.
        n_targets: number of predicted species (2 = PM10 + PM2.5).
        mode: ``"additive"`` or ``"multiplicative"``.
    """

    def __init__(self, in_dim, n_targets=2, mode="additive"):
        super().__init__()
        if mode not in HEADS:
            raise ValueError(f"mode must be one of {HEADS}, got {mode!r}")
        self.mode = mode
        self.n_targets = n_targets
        self.proj = nn.Linear(in_dim, n_targets)
        # Zero-init => identity-on-anchor at step 0, for BOTH modes.
        nn.init.constant_(self.proj.weight, 0.0)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, h, anchor):
        """
        Args:
            h: ``[N, in_dim]`` node embeddings.
            anchor: ``[N, n_targets]`` anchor state in log1p space.
        Returns:
            ``[N, n_targets]`` predicted state at ``t + dt``, log1p space.
        """
        z = self.proj(h)
        if self.mode == "additive":
            return anchor + z
        # AI-DUST Eq. 1: f = sigmoid(z) in (0,1), scaled by 2 so the anchor can be
        # revised upward as well as downward. Hard-bounds the output to [0, 2A].
        return 2.0 * torch.sigmoid(z) * anchor


def select_anchor(anchor_kind, c_t, c_ref):
    """Pick the anchor tensor. Both are supplied per-sample by the dataset.

    Args:
        anchor_kind: ``"persistence"`` -> ``C_t``; ``"physics"`` -> ``C_ref``.
        c_t: ``[N, n_targets]`` current state, log1p space.
        c_ref: ``[N, n_targets]`` physics-advanced state, log1p space.
    """
    if anchor_kind == "persistence":
        return c_t
    if anchor_kind == "physics":
        return c_ref
    raise ValueError(f"anchor must be one of {ANCHORS}, got {anchor_kind!r}")
