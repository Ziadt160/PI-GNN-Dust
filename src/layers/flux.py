import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class PhysicsFluxLayer(MessagePassing):
    """
    Physics-informed flux layer implementing wind-driven message passing.

    For each directed edge j -> i (PyG convention: ``x_j`` is the source /
    neighbour, ``x_i`` is the target / central node), the message combines:

      * the (normalised) node features of both endpoints, for representational
        capacity, and
      * an explicit, physically-meaningful advection term computed from the
        *physical* wind at the source cell and the *physical* concentration
        gradient along the edge.

    The advective contribution from j into i is proportional to the projection
    of the source wind onto the edge direction (j -> i) times the concentration
    difference, mirroring an upwind flux ``(u . n_{j->i}) * (C_j - C_i)``.
    Messages are summed (``aggr='add'``) so each node accumulates net inflow.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        # MLP input: x_i + x_j (node features) + edge_wind (1) + advection (1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1 + 1, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, x, edge_index, pos, u, v, pm10):
        """
        Args:
            x (Tensor):    Normalised node features [N, in_channels] (MLP input).
            edge_index (Tensor): Graph connectivity [2, E].
            pos (Tensor):  Geographically-oriented node positions [N, 2] = (east, north).
            u (Tensor):    Physical eastward wind component [N, 1] (m/s).
            v (Tensor):    Physical northward wind component [N, 1] (m/s).
            pm10 (Tensor): Physical (log-space) PM10 concentration [N, 1].
        """
        return self.propagate(edge_index, x=x, pos=pos, u=u, v=v, pm10=pm10)

    def message(self, x_i, x_j, pos_i, pos_j, u_j, v_j, pm10_i, pm10_j):
        # Unit-step direction from source j to target i.
        rel_pos = pos_i - pos_j  # [E, 2] = (east, north)

        # Wind at the source cell carries the dust toward the target.
        wind_vec = torch.cat([u_j, v_j], dim=-1)  # [E, 2]

        # Projection of source wind onto the edge direction (signed).
        edge_wind = torch.sum(wind_vec * rel_pos, dim=-1, keepdim=True)  # [E, 1]

        # Upwind advective flux contribution: wind projection * concentration drop.
        advection = edge_wind * (pm10_j - pm10_i)  # [E, 1]

        return self.mlp(torch.cat([x_i, x_j, edge_wind, advection], dim=-1))
