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


class ConservativeFluxLayer(MessagePassing):
    """Transport that cannot create or destroy dust, by construction.

    `PhysicsFluxLayer` builds each message with an MLP over
    ``[x_i, x_j, edge_wind, advection]``.  That carries the right directional
    prior, but nothing ties what LEAVES cell j to what ARRIVES at cell i: the two
    directed edges of one face are computed independently, so the network is free
    to invent dust at one end and lose it at the other.  Mass conservation is
    available to it, never required of it.

    Here the flux is antisymmetric by construction.  For the face between i and
    j, with the unit normal oriented j -> i::

        u_n   = (u_i + u_j)/2 . n_{j->i}        midpoint normal velocity
        c_up  = c_j if u_n > 0 else c_i         upwind concentration
        F     = gate(x_i + x_j) * u_n * c_up    what crosses the face

    Swapping i and j flips ``n``, so ``u_n`` changes sign; ``x_i + x_j`` is
    symmetric so ``gate`` is identical for both directions; and ``c_up`` selects
    the same physical cell either way.  Therefore ``F_{i->j} = -F_{j->i}``
    exactly.  Summed over the whole graph every face contributes ``+F`` and
    ``-F``, so the transport term sums to zero: this is a finite-volume scheme,
    not a penalty that hopes for one.

    The upwind choice is the other piece of physics.  Dust crossing a face came
    from whichever side the wind is blowing FROM, so taking ``c`` from the
    downwind cell would diffuse a plume backwards.

    One honest limit: the domain is open, and dust genuinely leaves through the
    boundary.  Conservation here is over the interior, with boundary loss real
    rather than an error -- which is why the readout is restricted to core nodes
    anyway.
    """

    # Characteristic scales that turn the raw physical product into an O(1)
    # quantity. Constants, so they cannot disturb antisymmetry.
    U_SCALE = 10.0      # m/s, a strong wind over this domain
    C_SCALE = 5.0       # log-space concentration, near the field's spread

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="add")
        # Acts on the SYMMETRIC combination x_i + x_j, so both directions of a
        # face get the same gate and antisymmetry survives.
        self.gate = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        # Overall strength, learnable but starting small so the operator begins
        # nearly inert and grows into its role instead of detonating at hop 1.
        self.strength = nn.Parameter(torch.tensor(-2.0))

    def forward(self, x, edge_index, pos, u, v, pm10):
        return self.propagate(edge_index, x=x, pos=pos, u=u, v=v, pm10=pm10)

    def message(self, x_i, x_j, pos_i, pos_j, u_i, u_j, v_i, v_j, pm10_i, pm10_j):
        rel = pos_i - pos_j                                   # direction j -> i
        norm = torch.linalg.norm(rel, dim=-1, keepdim=True).clamp(min=1e-6)
        n_hat = rel / norm

        # Midpoint wind: symmetric in (i, j), so only `n_hat` carries the sign.
        wind = torch.cat([0.5 * (u_i + u_j), 0.5 * (v_i + v_j)], dim=-1)
        u_n = (wind * n_hat).sum(dim=-1, keepdim=True) / self.U_SCALE

        # Upwind cell -- invariant under swapping the endpoints.
        c_up = torch.where(u_n > 0, pm10_j, pm10_i) / self.C_SCALE

        # The first version multiplied an UNBOUNDED gate by a raw wind and a raw
        # concentration, summed that over 8 neighbours, and compounded it every
        # hop. Measured against PhysicsFluxLayer on the same input it ran 388x
        # larger at 2 hops and 1.8e16 larger at 12 -- an exponential blow-up in
        # hop count, which is exactly why 48 h and 24 h diverged while 3 h did
        # not. Bounding the gate and scaling the physical factors fixes the
        # magnitude; all three are symmetric or constant, so F(j->i) = -F(i->j)
        # still holds exactly and the scheme stays conservative.
        gate = torch.tanh(self.gate(x_i + x_j))
        return nn.functional.softplus(self.strength) * gate * u_n * c_up
