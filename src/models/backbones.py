"""The three message-passing backbones in the benchmark.

``FluxBackbone``     your model: sum-aggregated MLP over a hand-built,
                     wind-projected flux message.  The upwind weighting is
                     *specified*, not learned.

``GATBackbone``      AI-DUST: stacked graph attention.  The neighbour weighting is
                     *learned*, but from node features alone -- it has no direct
                     view of the wind or the concentration gradient.

``FluxGATBackbone``  the hybrid (model M23).  Attention coefficients are
                     conditioned on your physical edge terms::

                         a_ij = softmax_j( LeakyReLU( a^T [ Wx_i | Wx_j | wind_ij | adv_ij ] ) )
                         m_ij = a_ij * MLP([ x_j | wind_ij | adv_ij ])

                     so the network *learns which neighbour is upwind* instead of
                     having it hard-coded, but the search is biased by the
                     physical projection rather than starting blind.

Two AI-DUST structural devices are available to all three:

``reinject``       re-apply separate learnable projections of the key physical
                   inputs (advection, emission, C_t) at every layer after the
                   first.  AI-DUST does this for the patch's centre node only; on
                   the full-domain graph the same mechanism applies to all nodes.

``concat_layers``  form the output from ``[h1 | h2 | ... | x_key]`` instead of the
                   last layer alone, preserving shallow (local-flux) and deep
                   (regional-transport) signal together.

Every backbone exposes the same call signature and an ``out_dim``, so the model
assembly in :mod:`src.models.unified` can swap them freely.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, MessagePassing
from torch_geometric.utils import softmax

from src.layers.flux import PhysicsFluxLayer

LEAKY_SLOPE = 0.2


def _edge_physics(pos_i, pos_j, u_j, v_j, c_i, c_j):
    """Shared physical edge quantities: wind projection and the advective flux.

    ``wind_ij`` is the source-cell wind projected onto the edge direction j -> i;
    ``adv_ij`` multiplies it by the concentration drop, mirroring an upwind flux
    ``(u . n) (C_j - C_i)``.  Identical to the terms in :mod:`src.layers.flux`.
    """
    rel_pos = pos_i - pos_j                                   # [E, 2] (east, north)
    wind = torch.cat([u_j, v_j], dim=-1)                      # [E, 2]
    wind_ij = torch.sum(wind * rel_pos, dim=-1, keepdim=True)  # [E, 1]
    adv_ij = wind_ij * (c_j - c_i)                            # [E, 1]
    return wind_ij, adv_ij


class FluxGATLayer(MessagePassing):
    """Physics-conditioned graph attention -- the M23 hybrid layer."""

    def __init__(self, in_channels, out_channels, heads=4, dropout=0.1):
        super().__init__(aggr="add", node_dim=0)
        self.heads = heads
        self.out_channels = out_channels
        self.dropout = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        # Attention reads [Wx_i | Wx_j | wind | adv] -> 2*out + 2 per head.
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels + 2))
        # The message itself keeps your flux form: neighbour features + physics.
        self.msg = nn.Linear(in_channels + 2, heads * out_channels)
        self.out_proj = nn.Linear(heads * out_channels, out_channels)

        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.msg.weight)
        nn.init.zeros_(self.msg.bias)

    def forward(self, x, edge_index, pos, u, v, c):
        out = self.propagate(edge_index, x=x, pos=pos, u=u, v=v, c=c)
        return self.out_proj(out)

    def message(self, x_i, x_j, pos_i, pos_j, u_j, v_j, c_i, c_j, index, size_i):
        wind_ij, adv_ij = _edge_physics(pos_i, pos_j, u_j, v_j, c_i, c_j)
        e = x_i.size(0)

        xi = self.lin(x_i).view(e, self.heads, self.out_channels)
        xj = self.lin(x_j).view(e, self.heads, self.out_channels)
        phys = torch.cat([wind_ij, adv_ij], dim=-1).unsqueeze(1).expand(e, self.heads, 2)

        logits = F.leaky_relu((torch.cat([xi, xj, phys], dim=-1) * self.att).sum(-1),
                              LEAKY_SLOPE)
        alpha = softmax(logits, index, num_nodes=size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        msg = self.msg(torch.cat([x_j, wind_ij, adv_ij], dim=-1))
        msg = msg.view(e, self.heads, self.out_channels)
        return (msg * alpha.unsqueeze(-1)).reshape(e, self.heads * self.out_channels)


class _BaseBackbone(nn.Module):
    """Shared plumbing: layer stack, optional re-injection, optional concat output."""

    def __init__(self, in_channels, hidden, layers, key_dim, reinject, concat_layers,
                 dropout):
        super().__init__()
        self.hidden = hidden
        self.n_layers = max(1, int(layers))
        self.reinject = bool(reinject) and key_dim > 0
        self.concat_layers = bool(concat_layers)
        self.key_dim = key_dim
        self.dropout = dropout

        if self.reinject:
            # AI-DUST uses a *separate* learnable projection per key input, per layer.
            self.reinjectors = nn.ModuleList(
                nn.ModuleList(nn.Linear(1, hidden) for _ in range(key_dim))
                for _ in range(self.n_layers - 1)
            )

    @property
    def out_dim(self):
        if self.concat_layers:
            return self.hidden * self.n_layers + self.key_dim
        return self.hidden

    def _apply_reinject(self, h, key_feats, layer_idx):
        if not self.reinject or layer_idx == 0:
            return h
        projs = self.reinjectors[layer_idx - 1]
        for k, proj in enumerate(projs):
            h = h + proj(key_feats[:, k:k + 1])
        return h

    def _finalise(self, states, key_feats):
        if not self.concat_layers:
            return states[-1]
        parts = list(states)
        if self.key_dim > 0 and key_feats is not None:
            parts.append(key_feats)
        return torch.cat(parts, dim=-1)


class FluxBackbone(_BaseBackbone):
    """Your PhysicsFluxLayer stack (residual between layers), optionally augmented."""

    def __init__(self, in_channels, hidden=64, layers=2, key_dim=0, reinject=False,
                 concat_layers=False, dropout=0.0):
        super().__init__(in_channels, hidden, layers, key_dim, reinject, concat_layers,
                         dropout)
        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(PhysicsFluxLayer(in_channels if i == 0 else hidden, hidden))

    def forward(self, x, edge_index, pos, u, v, c, key_feats=None):
        states, h = [], None
        for i, layer in enumerate(self.layers):
            inp = x if i == 0 else h
            out = torch.relu(layer(inp, edge_index, pos, u, v, c))
            h = out if i == 0 else h + out          # residual between hidden layers
            h = self._apply_reinject(h, key_feats, i)
            states.append(h)
        return self._finalise(states, key_feats)


class GATBackbone(_BaseBackbone):
    """AI-DUST's stack: graph attention, LeakyReLU, dropout (3 layers in the paper).

    Layer 1 runs on the raw node features with no fusion; layers 2+ receive the
    re-injected key physical inputs, exactly as described in their Methods.
    """

    def __init__(self, in_channels, hidden=64, layers=3, key_dim=0, reinject=True,
                 concat_layers=True, dropout=0.1, heads=4):
        super().__init__(in_channels, hidden, layers, key_dim, reinject, concat_layers,
                         dropout)
        self.layers = nn.ModuleList()
        self.projs = nn.ModuleList()
        for i in range(self.n_layers):
            self.layers.append(
                GATConv(in_channels if i == 0 else hidden, hidden, heads=heads,
                        concat=True, dropout=dropout))
            self.projs.append(nn.Linear(hidden * heads, hidden))

    def forward(self, x, edge_index, pos, u, v, c, key_feats=None):
        states, h = [], x
        for i, (layer, proj) in enumerate(zip(self.layers, self.projs)):
            h = proj(layer(h, edge_index))
            h = F.dropout(F.leaky_relu(h, LEAKY_SLOPE), p=self.dropout,
                          training=self.training)
            h = self._apply_reinject(h, key_feats, i)
            states.append(h)
        return self._finalise(states, key_feats)


class FluxGATBackbone(_BaseBackbone):
    """M23: attention conditioned on the physical edge terms (the flagship hybrid)."""

    def __init__(self, in_channels, hidden=64, layers=3, key_dim=0, reinject=True,
                 concat_layers=True, dropout=0.1, heads=4):
        super().__init__(in_channels, hidden, layers, key_dim, reinject, concat_layers,
                         dropout)
        self.layers = nn.ModuleList(
            FluxGATLayer(in_channels if i == 0 else hidden, hidden, heads=heads,
                         dropout=dropout)
            for i in range(self.n_layers)
        )

    def forward(self, x, edge_index, pos, u, v, c, key_feats=None):
        states, h = [], x
        for i, layer in enumerate(self.layers):
            out = F.dropout(F.leaky_relu(layer(h, edge_index, pos, u, v, c), LEAKY_SLOPE),
                            p=self.dropout, training=self.training)
            h = out if i == 0 else h + out
            h = self._apply_reinject(h, key_feats, i)
            states.append(h)
        return self._finalise(states, key_feats)


BACKBONES = {"flux": FluxBackbone, "gat": GATBackbone, "fluxgat": FluxGATBackbone}


def build_backbone(name, **kwargs):
    if name not in BACKBONES:
        raise ValueError(f"backbone must be one of {sorted(BACKBONES)}, got {name!r}")
    return BACKBONES[name](**kwargs)
