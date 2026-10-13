"""Loss functions for the benchmark, including AI-DUST's gated physics term.

Three modes, selectable per model so the loss becomes an ablation axis rather
than a fixed property of each architecture:

``mse``            plain supervised MSE (this repo's current default)
``gated``          AI-DUST Eq. 2 + Eq. 3
``homoscedastic``  learned uncertainty weighting (this repo's earlier default)

A finding that shaped the design
--------------------------------
Working AI-DUST's two equations against each other shows their physics penalty
is **almost always inert**:

    Eq. 1 sets  C_p = 2 f C_ref  with f = sigmoid(.) in (0, 1)
    => the Eq. 3 gate ratio  C_p / C_ref = 2f  lies in (0, 2)
    => the upper test `ratio < 10` is unconditionally true
    => alpha fires only when 2f < 0.1, i.e. f < 0.05

So with AI-DUST's own bounded head, the physics term is not a regulariser -- it
is an **anti-collapse floor**, active only when the network tries to wipe the
concentration below 5% of the physical reference.  In normal operation AI-DUST
trains on pure MSE.

Two consequences, both encoded here:

1. The transferable idea is the *anchor* (see :mod:`src.models.heads`), not the
   penalty.  This repo's earlier result that physics-as-loss degrades accuracy
   monotonically (R^2 0.622 -> 0.471 -> 0.156) is consistent with AI-DUST rather
   than contradicted by it: they never pay that cost.
2. The gate only carries information when the anchor is ``C_t``, where the ratio
   is unbounded.  :func:`gated_physics_loss` therefore reports its activation
   rate, so a run where the gate never fires is visible in the logs instead of
   being mistaken for "physics helped".
"""
import torch
import torch.nn as nn

LOSS_MODES = ("mse", "gated", "homoscedastic")

# AI-DUST Eq. 3
GATE_LO = 0.1
GATE_HI = 10.0
ALPHA_ON = 10.0
EPS = 1e-6


def gated_physics_loss(pred, target, c_ref, gate_lo=GATE_LO, gate_hi=GATE_HI,
                       alpha_on=ALPHA_ON):
    """AI-DUST Eq. 2 with the Eq. 3 adaptive weight.

        L     = (C_p - C_true)^2 + alpha (C_p - C_ref)^2
        alpha = 0     if gate_lo < C_p / C_ref < gate_hi
                10    otherwise

    Returns:
        loss: scalar
        info: dict with the data/physics split and the gate activation rate --
            log this; a 0% rate means the physics term did nothing.
    """
    data_term = (pred - target) ** 2

    ratio = pred / torch.clamp(c_ref.abs(), min=EPS)
    gate_open = (ratio > gate_lo) & (ratio < gate_hi)
    alpha = torch.where(gate_open, torch.zeros_like(ratio),
                        torch.full_like(ratio, alpha_on))

    phys_term = alpha * (pred - c_ref) ** 2
    loss = (data_term + phys_term).mean()

    return loss, {
        "data": float(data_term.mean().detach()),
        "physics": float(phys_term.mean().detach()),
        "gate_fired_frac": float((~gate_open).float().mean().detach()),
    }


class HomoscedasticLoss(nn.Module):
    """Learned uncertainty weighting over (supervised, physics) -- the repo's prior default.

    Kept so the ablation can attribute any change to AI-DUST's *gate* rather than
    to simply having removed this weighting.
    """

    def __init__(self, n_terms=2, device=None):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(n_terms, device=device))

    def forward(self, *terms):
        total = 0.0
        for i, term in enumerate(terms):
            total = total + 0.5 * torch.exp(-self.log_vars[i]) * term + 0.5 * self.log_vars[i]
        return total


class BenchmarkLoss(nn.Module):
    """Single entry point so every model in the roster is trained through one path.

    Args:
        mode: one of :data:`LOSS_MODES`.
    """

    def __init__(self, mode="mse", device=None):
        super().__init__()
        if mode not in LOSS_MODES:
            raise ValueError(f"mode must be one of {LOSS_MODES}, got {mode!r}")
        self.mode = mode
        self.mse = nn.MSELoss()
        self.homo = HomoscedasticLoss(2, device) if mode == "homoscedastic" else None

    def forward(self, pred, target, c_ref=None, physics_residual=None):
        if self.mode == "mse":
            return self.mse(pred, target), {"data": float(self.mse(pred, target).detach())}

        if self.mode == "gated":
            if c_ref is None:
                raise ValueError("gated loss requires c_ref")
            return gated_physics_loss(pred, target, c_ref)

        l_sup = self.mse(pred, target)
        l_phys = physics_residual if physics_residual is not None else torch.zeros_like(l_sup)
        return self.homo(l_sup, l_phys), {
            "data": float(l_sup.detach()), "physics": float(l_phys.detach()),
        }
