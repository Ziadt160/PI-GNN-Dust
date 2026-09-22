import torch
import torch.nn as nn
from tqdm import tqdm
from src.physics.loss import physics_loss


class Trainer:
    """
    Orchestrates PI-GNN training with homoscedastic uncertainty weighting
    (Kendall et al., 2018) to balance the supervised (data) loss and the physics
    residual loss automatically:

        L = sum_k [ 0.5 * exp(-s_k) * L_k + 0.5 * s_k ],   s_k = log(sigma_k^2)

    The combined weighted loss is the optimisation objective, but model selection
    is done on the raw validation **MSE** (interpretable and bounded below by 0),
    not the weighted loss (which can go negative as s_k decreases).
    """

    def __init__(self, model, optimizer, device, physics_cfg=None, log_vars=None, use_physics=True):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.physics_cfg = physics_cfg or {}
        self.use_physics = use_physics  # False -> pure data-driven (MSE only), for ablation

        if log_vars is None:
            self.log_vars = nn.Parameter(torch.zeros(2, device=device))
            if use_physics:
                self.optimizer.add_param_group({"params": [self.log_vars]})
        else:
            self.log_vars = log_vars

    def _physics(self, data, pm10_next):
        return physics_loss(
            data.x[:, 0:1], pm10_next, data.x[:, 1], data.x[:, 2],
            delta_t=self.physics_cfg.get("delta_t", 43200.0),
            dx=self.physics_cfg.get("dx", 74880.0),
            dy=self.physics_cfg.get("dy", 83490.0),
            grid_size=tuple(self.physics_cfg.get("grid_size", (7, 12))),
        )

    def _combined(self, l_sup, l_phys):
        if not self.use_physics:
            return l_sup
        return (0.5 * torch.exp(-self.log_vars[0]) * l_sup + 0.5 * self.log_vars[0]) + \
               (0.5 * torch.exp(-self.log_vars[1]) * l_phys + 0.5 * self.log_vars[1])

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for data in tqdm(loader, desc="Training", leave=False):
            data = data.to(self.device)
            self.optimizer.zero_grad()

            pm10_next, _ = self.model(data)
            l_sup = self.mse_loss(pm10_next, data.y)
            l_phys = self._physics(data, pm10_next) if self.use_physics else l_sup
            loss = self._combined(l_sup, l_phys)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def validate(self, loader):
        """Returns (weighted_loss, supervised_mse) averaged over the loader."""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        for data in loader:
            data = data.to(self.device)
            pm10_next, _ = self.model(data)
            l_sup = self.mse_loss(pm10_next, data.y)
            l_phys = self._physics(data, pm10_next)
            total_loss += self._combined(l_sup, l_phys).item()
            total_mse += l_sup.item()
        n = len(loader)
        return total_loss / n, total_mse / n
