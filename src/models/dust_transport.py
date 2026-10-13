"""A dust-transport simulator on the graph: emit everywhere, advect, read out inside.

Two measured failures motivate this architecture, and both are about reach rather
than capacity.

**1. The receptive field was far too small for the forecast horizon.**
The model carried ``hops=2`` from the original 3-hour experiment.  Two hops is a
152 km receptive field, which is fine at 3 h (dust travels ~60-90 km) and badly
wrong at 24 h, where p90 winds carry dust 500 km and storm winds 700 km -- 6 to 9
cells.  A 2-hop network at 24 h cannot see where its dust came from however large
the domain is.  Hops are therefore derived from the horizon:

    hops ~= wind x lead_time / cell_size

**2. Only interior cells can be predicted, but every cell must participate.**
Boundary cells have a truncated stencil and no inflow, so they are poor targets.
They are excellent *sources*: 42% of observed dust storms had no local emission,
68% arrive on a northerly wind, and an interior storm is preceded by a northern
station's storm 48.8% of the time within 24 h (versus 21.8% from the south -- so
the signal is directional transport, not widespread dust).  Excluding boundary
cells from the simulation throws that away.

So every node emits and transports; only interior nodes are read out.

Structure
---------
    h_i(0)   = encode(local features at i)                    all nodes
    for k in 1..K:
        inflow_i = sum_j  a_ij * flux(h_j | wind, edge geometry)
        h_i(k)   = h_i(k-1) + inflow_i + source_i             emission re-injected
    logits   = head(h_i(K))                                   interior nodes only

The source term is re-injected at **every** hop rather than only at the input,
because dust is emitted continuously along the transport path, not just at the
origin.  A single injection would model a puff released at t=0; re-injection
models a line of sources under a moving wind field, which is what a shamal is.

Transport uses `PhysicsFluxLayer`, so each message already carries the source
cell's wind projected onto the edge direction and the concentration difference
along it -- the directional prior is built in rather than learned from scratch.
"""
import math

import torch
import torch.nn as nn

from src.layers.flux import ConservativeFluxLayer, PhysicsFluxLayer

# 0.75 deg cell at this latitude; used to turn a lead time into a hop count.
CELL_KM = 76.0
# p90 10 m wind over the domain, m/s -- the speed the hop count is sized for.
DESIGN_WIND_MS = 5.8
MAX_HOPS = 12          # compute ceiling; 48 h at storm winds would ask for ~19


def hops_for_horizon(lead_hours, wind_ms=DESIGN_WIND_MS, cell_km=CELL_KM,
                     max_hops=MAX_HOPS):
    """Message-passing hops needed for dust to cross the distance it really travels.

    Returns at least 2, and is capped: the honest reading of a capped value is
    that the model still cannot see the full source region at that lead time.
    """
    km = wind_ms * lead_hours * 3.6
    return max(2, min(max_hops, int(math.ceil(km / cell_km))))


class DustTransportGNN(nn.Module):
    """Emit at every node, advect along the graph, predict interior nodes only.

    Args:
        in_channels: assembled node features.
        n_out: output logits per node (one per event threshold).
        emission_idx: indices of the emission-related channels, re-injected at
            each hop as the local source term.
        hops: transport steps; use :func:`hops_for_horizon`.
        aux_out: width of an optional SECOND head read from the same backbone.
            0 disables it and ``forward`` returns a single tensor, as before.

    The auxiliary head exists to break the project's binding constraint, which is
    labels rather than capacity.  The station target supervises **11 nodes**; the
    CAMS field supervises **all 251, at every one of 64,288 timesteps** -- about
    thirty times more signal, covering the whole country instead of eleven dots.

    CAMS cannot be the primary target: measured over 234,512 matched
    station-hours it under-reports storm dust by 15-49x, so a model trained on it
    learns to predict a damped field.  But it is not noise.  Its *ordering*
    survives: across observed severity quartiles within real storms its median
    rises 177 / 197 / 223 / 262, monotone, and rank correlation against observed
    extinction is 0.30 in dust-coded hours.  It compresses a 4.02x real range
    into 1.48x rather than destroying it.

    So it is used the way a weak label should be: as a dense auxiliary task that
    shapes the shared representation, at a low weight, while the sparse observed
    target decides the answer.  Crucially the two heads are SIBLINGS, not a
    pipeline -- the station head reads the backbone directly, never the CAMS
    head's output, so a bad PM10 prediction cannot propagate into the severity
    prediction.  Sarafian et al. 2023 use the same arrangement for this task and
    weight the auxiliary loss at 0.05.
    """

    def __init__(self, in_channels, n_out, emission_idx, hidden=48, hops=7,
                 dropout=0.3, aux_out=0, deposition_idx=None, dt_hop=10800.0,
                 conservative=False):
        super().__init__()
        self.hops = max(1, int(hops))
        self.emission_idx = list(emission_idx)

        self.encode = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU())
        # One shared transport operator applied repeatedly: the physics of a hop
        # does not change with the hop index, and sharing keeps the parameter
        # count flat as the horizon (and so the hop count) grows.
        self.transport = (ConservativeFluxLayer(hidden, hidden) if conservative
                          else PhysicsFluxLayer(hidden, hidden))
        # Local emission source, re-injected every hop.
        self.source = nn.Sequential(nn.Linear(len(self.emission_idx), hidden), nn.ReLU())
        # --- removal ---------------------------------------------------------- #
        # `decay` alone was ONE global scalar: the same survival fraction for all
        # 251 cells, every hop, in every weather condition. Rain entered only as
        # an additive input feature, but scavenging is MULTIPLICATIVE -- rain
        # removes a fraction of the dust -- so the additive form could not express
        # it without learning a contorted interaction.
        #
        # With `deposition_idx` the survival fraction becomes physical per node:
        #
        #     keep = exp( -(v_dry + lambda * P) * dt / H )
        #
        # P precipitation rate, H mixing depth, dt the seconds one hop represents.
        # Both rates are per-hidden-channel rather than scalar, so the network can
        # hold slow- and fast-settling components at once -- the latent stand-in
        # for coarse dust dropping out in minutes while fine dust hangs for days,
        # which a single shared decay forced to be identical.
        self.decay = nn.Parameter(torch.tensor(0.0))     # fallback: global scalar
        self.deposition_idx = list(deposition_idx or [])
        self.dt_hop = float(dt_hop)
        if self.deposition_idx:
            self.v_dry = nn.Parameter(torch.full((hidden,), -4.0))
            self.scavenge = nn.Parameter(torch.full((hidden,), -4.0))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_out)
        # Sibling head, not a second stage. Reads the same `h`.
        self.aux_out = int(aux_out)
        self.aux_head = nn.Linear(hidden, self.aux_out) if self.aux_out else None

        self.register_buffer("mean", torch.zeros(in_channels))
        self.register_buffer("std", torch.ones(in_channels))

    def _survival(self, x):
        """Fraction of dust surviving one hop: scalar, or physical per node."""
        if not self.deposition_idx:
            return torch.sigmoid(self.decay)
        # Unnormalised, like the wind and concentration the flux layer reads:
        # these are physical quantities, not standardised features.
        tp_log = x[:, self.deposition_idx[0]:self.deposition_idx[0] + 1]
        blh = x[:, self.deposition_idx[1]:self.deposition_idx[1] + 1]
        precip = torch.expm1(tp_log.clamp(min=0.0, max=20.0))     # -> mm
        depth = blh.clamp(min=50.0, max=6000.0)                   # -> m
        rate = (nn.functional.softplus(self.v_dry)
                + nn.functional.softplus(self.scavenge) * precip)
        return torch.exp(-rate * self.dt_hop / depth).clamp(1e-4, 1.0)

    def set_stats(self, m, s):
        with torch.no_grad():
            self.mean.copy_(torch.as_tensor(m, dtype=torch.float32))
            s = torch.as_tensor(s, dtype=torch.float32)
            # See StationDustModel.set_stats: a 1e-6 floor turns a channel that
            # is constant in the stats subsample into a 1e6 amplifier.
            self.std.copy_(torch.where(s < 1e-3, torch.ones_like(s), s))

    def forward(self, x, edge_index, pos, readout=None):
        """
        Args:
            x: ``[N, in_channels]`` features for **all** nodes, boundary included.
            readout: node indices to return; ``None`` returns every node.

        Returns the event logits, or ``(event_logits, aux_pred)`` when an
        auxiliary head is configured.  The auxiliary prediction always covers
        **every** node, because the CAMS field is dense and its whole value is
        that it supervises the cells the stations cannot reach.
        """
        pm10, u, v = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        xn = (x - self.mean) / self.std

        h = self.encode(xn)
        src = self.source(xn[:, self.emission_idx])
        keep = self._survival(x)              # in (0,1): what survives one hop

        for _ in range(self.hops):
            inflow = self.transport(h, edge_index, pos, u, v, pm10)
            h = keep * h + torch.relu(inflow) + src

        h = self.drop(h)
        main = self.head(h if readout is None else h[readout])
        if self.aux_head is None:
            return main
        return main, self.aux_head(h)
