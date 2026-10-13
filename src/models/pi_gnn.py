import torch
import torch.nn as nn
from src.layers.flux import PhysicsFluxLayer


class PIDustModel(nn.Module):
    """
    Physics-Informed Graph Neural Network (PI-GNN) for dust forecasting.

    Predicts the temporal change (delta) in log-space PM10 with a flux-based GNN
    and forms the next state as a residual on the current one
    (``C_{t+1} = C_t + delta``). The output head is zero-initialised so training
    starts from the identity / persistence solution and learns corrections.

    Input node features are expected with the *current* frame's physical values at
    indices ``[0:4] = [log_pm10, u10, v10, t2m]`` (see
    :class:`~src.training.dataset.GraphDustDataset`); any additional history /
    derived channels follow. Features are standardised internally using
    train-split statistics (registered buffers), while the physical current-frame
    PM10 / winds (indices 0/1/2) are forwarded unchanged to the flux layer and the
    residual update so the physics terms stay dimensionally correct and leakage-free.

    ``hops`` stacks additional flux layers (with residual connections) to widen the
    spatial receptive field beyond a single neighbour.
    """

    def __init__(self, in_channels=4, hidden_channels=64, hops=1):
        super().__init__()
        self.in_channels = in_channels
        self.hops = max(1, int(hops))
        self.flux_in = PhysicsFluxLayer(in_channels=in_channels, out_channels=hidden_channels)
        self.flux_hidden = nn.ModuleList(
            PhysicsFluxLayer(in_channels=hidden_channels, out_channels=hidden_channels)
            for _ in range(self.hops - 1)
        )
        self.output_head = nn.Linear(hidden_channels, 1)  # Predicts delta(log PM10)

        self.register_buffer("feat_mean", torch.zeros(in_channels))
        self.register_buffer("feat_std", torch.ones(in_channels))

        nn.init.constant_(self.output_head.weight, 0.0)
        nn.init.constant_(self.output_head.bias, 0.0)

    def set_norm_stats(self, mean, std):
        """Set input-feature normalisation statistics (fitted on the train split)."""
        with torch.no_grad():
            self.feat_mean.copy_(torch.as_tensor(mean, dtype=self.feat_mean.dtype))
            std = torch.as_tensor(std, dtype=self.feat_std.dtype)
            self.feat_std.copy_(torch.clamp(std, min=1e-6))

    def forward(self, data):
        """
        Returns:
            pm10_next (Tensor): Predicted log-space PM10 at t+1 [N, 1].
            delta_pm10 (Tensor): Predicted change in log-space PM10 [N, 1].
        """
        x, edge_index, pos = data.x, data.edge_index, data.pos

        # Current-frame physical quantities (un-normalised) for physics / residual.
        pm10_phys = x[:, 0:1]
        u10, v10 = x[:, 1:2], x[:, 2:3]

        x_norm = (x - self.feat_mean) / self.feat_std

        h = torch.relu(self.flux_in(x_norm, edge_index, pos, u10, v10, pm10_phys))
        for flux in self.flux_hidden:
            h = h + torch.relu(flux(h, edge_index, pos, u10, v10, pm10_phys))

        delta_pm10 = self.output_head(h)

        # Residual update in log space (targets are log1p(PM10) >= 0, so no
        # Softplus floor is applied); physical non-negativity via `to_physical`.
        pm10_next = pm10_phys + delta_pm10
        return pm10_next, delta_pm10

    @staticmethod
    def to_physical(log_pm10):
        """Map a log-space prediction back to non-negative physical PM10 (ug/m^3)."""
        return torch.expm1(torch.clamp(log_pm10, min=0.0))
