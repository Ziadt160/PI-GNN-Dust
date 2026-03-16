import torch
import torch.nn as nn
from src.layers.flux import PhysicsFluxLayer

class PIDustModel(nn.Module):
    """
    Physics-Informed Graph Neural Network (PI-GNN) for Dust Forecasting.
    Predicts the temporal change (delta) in PM10 concentration using a flux-based GNN architecture.
    """
    def __init__(self, in_channels=4, hidden_channels=64):
        super().__init__()
        self.flux_layer = PhysicsFluxLayer(in_channels=in_channels, out_channels=hidden_channels)
        self.output_head = nn.Linear(hidden_channels, 1) # Predicts Delta PM10
        
        # Initialize output to zero (identity prediction: next state = current state)
        nn.init.constant_(self.output_head.weight, 0.0)
        nn.init.constant_(self.output_head.bias, 0.0)

    def forward(self, data):
        """
        Args:
            data: PyG Data object containing x, edge_index, and pos.
        Returns:
            pm10_next (Tensor): Predicted PM10 at t+1.
            delta_pm10 (Tensor): Predicted change in PM10.
        """
        x, edge_index, pos = data.x, data.edge_index, data.pos
        
        # Assume feature order: [pm10, u10, v10, t2m]
        u10, v10 = x[:, 1], x[:, 2] 
        
        # Compute spatial flux contributions
        flux = self.flux_layer(x, edge_index, pos, u10, v10)
        
        # Predict the change (delta)
        delta_pm10 = self.output_head(flux)
        
        # Apply the update and ensure non-negativity using Softplus
        pm10_next = nn.functional.softplus(x[:, 0:1] + delta_pm10)
        
        return pm10_next, delta_pm10
