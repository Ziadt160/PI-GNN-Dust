"""Model assembly: backbone + unified head, under one forward contract.

Every one of the 26 benchmark models is an instance of :class:`UnifiedDustModel`
(optionally wrapped in :class:`RegimeSplitModel`) built from a
:class:`ModelConfig`.  That is what makes the comparison an ablation rather than
a collection of separately-written networks: any two rows in the results table
differ only by the config fields that name their difference.

Contract, identical for both topologies:

    forward(data) -> [n_predicted, n_targets]   in log1p(ug/m^3)

For ``topology="full"`` that is every node of every graph in the batch; for
``topology="patch"`` it is one row per graph -- the centre node.  The targets
supplied by the datasets already match, so the training loop is topology-blind.

Normalisation is applied *inside* the model from registered buffers fitted on
the training split only.  The physical channels (concentration, wind) are passed
to the backbone un-normalised, because the flux/attention terms are physical
quantities and must keep their units.
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.models.backbones import build_backbone
from src.models.heads import UnifiedHead, select_anchor

KEY_DIM = 3          # transport, emission tendency, C_t -- AI-DUST's re-injected inputs

# Physical channel positions in the assembled feature matrix (see FeatureAssembler).
CH_PM10, CH_U, CH_V = 0, 1, 2


@dataclass
class ModelConfig:
    """One row of the benchmark roster."""
    name: str
    note: str = ""

    # architecture
    backbone: str = "flux"           # flux | gat | fluxgat
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0
    reinject: bool = False
    concat_layers: bool = False

    # prediction contract
    anchor: str = "persistence"      # persistence | physics
    head: str = "additive"           # additive | multiplicative

    # objective
    loss: str = "mse"                # mse | gated | homoscedastic

    # data view (overrides on DataConfig)
    data: dict = field(default_factory=dict)

    # training regime
    split_emission: bool = False
    stratified: bool = False

    @property
    def topology(self):
        return self.data.get("topology", "full")


class UnifiedDustModel(nn.Module):
    """Backbone + unified head, with train-split feature normalisation."""

    def __init__(self, cfg: ModelConfig, in_channels, n_targets=2, topology="full"):
        super().__init__()
        self.cfg = cfg
        self.topology = topology
        self.n_targets = n_targets

        self.backbone = build_backbone(
            cfg.backbone,
            in_channels=in_channels,
            hidden=cfg.hidden,
            layers=cfg.layers,
            key_dim=KEY_DIM if cfg.reinject or cfg.concat_layers else 0,
            reinject=cfg.reinject,
            concat_layers=cfg.concat_layers,
            dropout=cfg.dropout,
            **({"heads": cfg.heads} if cfg.backbone in ("gat", "fluxgat") else {}),
        )
        self.head = UnifiedHead(self.backbone.out_dim, n_targets, cfg.head)

        self.register_buffer("feat_mean", torch.zeros(in_channels))
        self.register_buffer("feat_std", torch.ones(in_channels))

    def set_norm_stats(self, mean, std):
        """Fit-once normalisation from the training split (never val/test)."""
        with torch.no_grad():
            self.feat_mean.copy_(torch.as_tensor(mean, dtype=self.feat_mean.dtype))
            self.feat_std.copy_(torch.clamp(
                torch.as_tensor(std, dtype=self.feat_std.dtype), min=1e-6))

    def _centre_rows(self, data):
        """Global node indices of each graph's centre node (patch topology)."""
        return data.ptr[:-1] + data.centre_idx

    def forward(self, data):
        x = data.x
        # Physical quantities keep their units for the flux / attention terms.
        pm10 = x[:, CH_PM10:CH_PM10 + 1]
        u = x[:, CH_U:CH_U + 1]
        v = x[:, CH_V:CH_V + 1]

        x_norm = (x - self.feat_mean) / self.feat_std
        key = getattr(data, "key", None)

        h = self.backbone(x_norm, data.edge_index, data.pos, u, v, pm10, key)

        if self.topology == "patch":
            h = h[self._centre_rows(data)]

        anchor = select_anchor(self.cfg.anchor, data.c_t, data.c_ref)
        return self.head(h, anchor)


class RegimeSplitModel(nn.Module):
    """AI-DUST's two-model design: one for emission-active cells, one for the rest.

    Both sub-models see the same inputs; at inference each prediction is taken
    from the sub-model matching that cell's emission state.  During training both
    receive the full batch and are each optimised on their own subset, which keeps
    the step count identical to a single model and the comparison fair.
    """

    def __init__(self, cfg: ModelConfig, in_channels, n_targets=2, topology="full"):
        super().__init__()
        self.active = UnifiedDustModel(cfg, in_channels, n_targets, topology)
        self.inactive = UnifiedDustModel(cfg, in_channels, n_targets, topology)
        self.cfg = cfg
        self.topology = topology

    def set_norm_stats(self, mean, std):
        self.active.set_norm_stats(mean, std)
        self.inactive.set_norm_stats(mean, std)

    def forward(self, data):
        pred_a = self.active(data)
        pred_i = self.inactive(data)
        mask = data.emis_active.reshape(-1, 1).to(pred_a.dtype)
        return mask * pred_a + (1.0 - mask) * pred_i


def build_model(cfg: ModelConfig, in_channels, n_targets=2, topology="full"):
    cls = RegimeSplitModel if cfg.split_emission else UnifiedDustModel
    return cls(cfg, in_channels, n_targets, topology)
