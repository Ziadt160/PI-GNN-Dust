import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class PhysicsFluxLayer(MessagePassing):
    """
    Physics-Informed Flux Layer that implements message passing based on wind-driven transport.
    Learns to balance advection, diffusion, and source/sink terms.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add') # Sum incoming fluxes
        # Inputs: x_i + x_j + edge_wind + advection_term
        # in_channels * 2 (i and j nodes) + 1 (edge wind) + 1 (advection term)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1 + 1, out_channels), 
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, pos, u10, v10):
        """
        Forward pass for computing spatio-temporal fluxes.
        
        Args:
            x (Tensor): Node features [N, in_channels]
            edge_index (Tensor): Graph connectivity [2, E]
            pos (Tensor): Node positions [N, 2]
            u10 (Tensor): Eastward wind component [N]
            v10 (Tensor): Northward wind component [N]
        """
        row, col = edge_index
        
        # Vector pointing from source (j) to target (i)
        rel_pos = pos[row] - pos[col] 
        
        # Use wind at target node i to determine acceptance of flow
        wind_vec = torch.stack([u10[row], v10[row]], dim=-1) # [E, 2]
        
        # Projection of wind onto the edge direction
        edge_wind = torch.sum(wind_vec * rel_pos, dim=-1).unsqueeze(-1) # [E, 1]
        
        return self.propagate(edge_index, x=x, edge_attr=edge_wind)

    def message(self, x_i, x_j, edge_attr):
        """
        Computes the flux (message) from node j to node i.
        Incorporates an explicit advection term: Flux is proportional to wind speed and concentration gradient.
        """
        # pm10 is assumed to be the first feature (index 0)
        pm10_diff = x_j[:, 0:1] - x_i[:, 0:1]
        advection = edge_attr * pm10_diff
        
        return self.mlp(torch.cat([x_i, x_j, edge_attr, advection], dim=-1))
