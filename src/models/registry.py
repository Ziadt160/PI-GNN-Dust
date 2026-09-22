"""The benchmark roster: one :class:`ModelConfig` per model.

Organised as four groups, so every row in the results table answers a question:

  A  reference points        AI-DUST as published, and your model as it stands
  B  one AI-DUST idea -> your model        "which of their ideas is worth taking?"
  C  one of your ideas -> AI-DUST          "which of yours improves theirs?"
  D  hybrids                               the mixed models

Group B and C are deliberately *single-idea* diffs against their group's
reference, so an improvement is attributable.  Anything that changes two things
at once belongs in group D.

Adding a model means adding a config here -- no new architecture code.
"""
from src.models.unified import ModelConfig

# --------------------------------------------------------------------------- #
# Reference data views
# --------------------------------------------------------------------------- #
# Your model as it stands: history window, no future met, no surface physics.
BASE_DATA = dict(
    cadence="3h", window=3, future_met=False, courant=False, relief=False,
    emission_features=False, transport_feature=False, divergence=True,
    multitask=True, topology="full", neighbors=8,
)

# AI-DUST as published, adapted to a surface-only, 0.75 deg domain:
# concentration at t only (no history), met at t and t+dt, Courant winds,
# relative elevation, explicit emission, 4-connected patch, dust (PM10) only.
AIDUST_DATA = dict(
    cadence="3h", window=1, future_met=True, courant=True, relief=True,
    emission_features=True, transport_feature=True, divergence=True,
    multitask=False, topology="patch", patch=7, neighbors=4,
)


def _d(base, **over):
    out = dict(base)
    out.update(over)
    return out


# --------------------------------------------------------------------------- #
# Group A -- reference points
# --------------------------------------------------------------------------- #
GROUP_A = [
    ModelConfig(
        name="M1_AIDUST", note="AI-DUST as published (surface-only adaptation)",
        backbone="gat", layers=3, hidden=64, heads=4, dropout=0.1,
        reinject=True, concat_layers=True,
        anchor="physics", head="multiplicative", loss="gated",
        split_emission=True, stratified=True, data=AIDUST_DATA,
    ),
    ModelConfig(
        name="M2_SRC", note="your current model, unchanged",
        backbone="flux", layers=2, hidden=64,
        anchor="persistence", head="additive", loss="mse",
        data=BASE_DATA,
    ),
]

# --------------------------------------------------------------------------- #
# Group B -- one AI-DUST idea grafted onto your model
# --------------------------------------------------------------------------- #
GROUP_B = [
    ModelConfig(name="M3_SRC+anchor", note="predict from C_ref instead of C_t",
                backbone="flux", layers=2, anchor="physics", head="additive",
                loss="mse", data=BASE_DATA),
    ModelConfig(name="M4_SRC+bounded", note="bounded head 2.sigma(z).A",
                backbone="flux", layers=2, anchor="persistence",
                head="multiplicative", loss="mse", data=BASE_DATA),
    ModelConfig(name="M5_SRC+emission", note="Shao emission E/H and u* as inputs",
                backbone="flux", layers=2, anchor="persistence", head="additive",
                loss="mse", data=_d(BASE_DATA, emission_features=True)),
    ModelConfig(name="M6_SRC+divergence", note="transport channel as div(VC)",
                backbone="flux", layers=2, anchor="persistence", head="additive",
                loss="mse", data=_d(BASE_DATA, transport_feature=True, divergence=True)),
    ModelConfig(name="M6b_SRC+advection", note="control for M6: same channel as u.grad C",
                backbone="flux", layers=2, anchor="persistence", head="additive",
                loss="mse", data=_d(BASE_DATA, transport_feature=True, divergence=False)),
    ModelConfig(name="M7_SRC+gated", note="AI-DUST Eq.3 gate (live: anchor is C_t)",
                backbone="flux", layers=2, anchor="persistence", head="additive",
                loss="gated", data=BASE_DATA),
    ModelConfig(name="M8_SRC+courant", note="Courant wind normalisation u/dx, v/dy",
                backbone="flux", layers=2, anchor="persistence", head="additive",
                loss="mse", data=_d(BASE_DATA, courant=True)),
    ModelConfig(name="M9_SRC+relief", note="relative elevation as an input",
                backbone="flux", layers=2, anchor="persistence", head="additive",
                loss="mse", data=_d(BASE_DATA, relief=True)),
    ModelConfig(name="M10_SRC+attention", note="GAT replaces the flux message",
                backbone="gat", layers=2, heads=4, dropout=0.1,
                anchor="persistence", head="additive", loss="mse", data=BASE_DATA),
    ModelConfig(name="M11_SRC+reinject", note="per-layer re-injection of {adv, E, C_t}",
                backbone="flux", layers=2, reinject=True,
                anchor="persistence", head="additive", loss="mse", data=BASE_DATA),
    ModelConfig(name="M12_SRC+concat", note="multi-scale concat output head",
                backbone="flux", layers=2, concat_layers=True,
                anchor="persistence", head="additive", loss="mse", data=BASE_DATA),
    ModelConfig(name="M13_SRC+stratified", note="AI-DUST stratified weighting",
                backbone="flux", layers=2, anchor="persistence", head="additive",
                loss="mse", stratified=True, data=BASE_DATA),
    ModelConfig(name="M14_SRC+split", note="separate emission-active / inactive models",
                backbone="flux", layers=2, anchor="persistence", head="additive",
                loss="mse", split_emission=True, data=BASE_DATA),
    ModelConfig(name="M15_SRC+patch", note="patch / centre-node topology",
                backbone="flux", layers=2, anchor="persistence", head="additive",
                loss="mse", data=_d(BASE_DATA, topology="patch", patch=7)),
    ModelConfig(name="M16_SRC+futuremet", note="meteorology at t+dt as forcing",
                backbone="flux", layers=2, anchor="persistence", head="additive",
                loss="mse", data=_d(BASE_DATA, future_met=True)),
]

# --------------------------------------------------------------------------- #
# Group C -- one of your ideas grafted onto AI-DUST
# --------------------------------------------------------------------------- #
_AID = dict(backbone="gat", layers=3, hidden=64, heads=4, dropout=0.1,
            reinject=True, concat_layers=True, anchor="physics",
            head="multiplicative", loss="gated", split_emission=True, stratified=True)

GROUP_C = [
    ModelConfig(name="M17_AID+fulldomain", note="AI-DUST net on your full-domain graph",
                **_AID, data=_d(AIDUST_DATA, topology="full", neighbors=4)),
    ModelConfig(name="M18_AID+fluxmsg", note="your flux message replaces blind attention",
                **{**_AID, "backbone": "flux"}, data=AIDUST_DATA),
    ModelConfig(name="M19_AID+8nbr", note="your 8-neighbour connectivity (diagonals)",
                **_AID, data=_d(AIDUST_DATA, neighbors=8)),
    ModelConfig(name="M20_AID+additive", note="your additive residual head",
                **{**_AID, "head": "additive"}, data=AIDUST_DATA),
    ModelConfig(name="M21_AID+history", note="your temporal history window (sees dC/dt)",
                **_AID, data=_d(AIDUST_DATA, window=3)),
    ModelConfig(name="M22_AID+multitask", note="your joint PM10 + PM2.5 heads",
                **_AID, data=_d(AIDUST_DATA, multitask=True)),
    ModelConfig(name="M27_AID+semilag",
                note="your CFL-stable semi-Lagrangian reference replaces Euler",
                **_AID, data=_d(AIDUST_DATA, reference_mode="semilag")),
]

# --------------------------------------------------------------------------- #
# Group D -- hybrids
# --------------------------------------------------------------------------- #
# The hybrids use the CFL-stable reference: at 0.75 deg / 3 h the Euler reference
# is invalid for 4% of cell-times, and there is no reason to inherit that.
FULL_FEATURES = _d(BASE_DATA, future_met=True, courant=True, relief=True,
                   emission_features=True, transport_feature=True, divergence=True,
                   reference_mode="semilag")

# Ideas that actually paid in the screening pass, stacked on the PLAIN flux
# backbone.  M25_MIX-FULL scored only +0.010 despite combining ideas worth +0.082,
# +0.069 and +0.041 individually, because it also inherited the two that did
# nothing: the physics anchor (+0.003, RMSE worse) and the FluxGAT backbone
# (+0.002).  These two configs retest the stack without those.
#
# `reference_mode` stays "euler" deliberately: with a persistence anchor and MSE
# loss, C_ref is never read, so matching BASE_DATA keeps them on the same cached
# SaudiFields instead of forcing another preprocessing pass.
MIX_LITE_DATA = _d(BASE_DATA, future_met=True, emission_features=True, courant=True,
                   relief=True, transport_feature=True, divergence=True)

GROUP_D = [
    ModelConfig(
        name="M24b_MIX-CORE", note="the two biggest wins only: future met + emission",
        backbone="flux", layers=2, hidden=64,
        anchor="persistence", head="additive", loss="mse",
        data=_d(BASE_DATA, future_met=True, emission_features=True),
    ),
    ModelConfig(
        name="M24_MIX-LITE", note="every idea worth >=+0.025 TS, on the plain backbone",
        backbone="flux", layers=2, hidden=64, reinject=True,
        anchor="persistence", head="multiplicative", loss="mse",
        stratified=True, data=MIX_LITE_DATA,
    ),
    ModelConfig(
        name="M23_FLUXGAT", note="attention conditioned on your physical edge terms",
        backbone="fluxgat", layers=3, hidden=64, heads=4, dropout=0.1,
        reinject=True, concat_layers=True,
        anchor="physics", head="additive", loss="mse",
        data=FULL_FEATURES,
    ),
    ModelConfig(
        name="M25_MIX-FULL", note="every idea on, your full-domain graph",
        backbone="fluxgat", layers=3, hidden=64, heads=4, dropout=0.1,
        reinject=True, concat_layers=True,
        anchor="physics", head="additive", loss="mse",
        split_emission=True, stratified=True, data=FULL_FEATURES,
    ),
    ModelConfig(
        name="M26_MIX-PATCH", note="MIX-FULL on the patch topology (transferability)",
        backbone="fluxgat", layers=3, hidden=64, heads=4, dropout=0.1,
        reinject=True, concat_layers=True,
        anchor="physics", head="additive", loss="mse",
        split_emission=True, stratified=True,
        data=_d(FULL_FEATURES, topology="patch", patch=7),
    ),
]

# M24_MIX-LITE is composed from the wave-1 winners -- see `compose_mix_lite`.
ALL_MODELS = GROUP_A + GROUP_B + GROUP_C + GROUP_D
BY_NAME = {m.name: m for m in ALL_MODELS}

GROUPS = {"A": GROUP_A, "B": GROUP_B, "C": GROUP_C, "D": GROUP_D}

# The smallest subset that still answers every question asked.
CORE = ["M1_AIDUST", "M2_SRC", "M3_SRC+anchor", "M5_SRC+emission",
        "M23_FLUXGAT", "M25_MIX-FULL"]


def compose_mix_lite(winner_names, base_name="M2_SRC", name="M24_MIX-LITE"):
    """Build MIX-LITE by stacking the wave-1 winners onto your model.

    Each winning single-idea config contributes the field(s) that made it differ
    from ``M2_SRC``; the result is the combination of the ideas that actually paid.
    """
    base = BY_NAME[base_name]
    cfg = ModelConfig(name=name, note="top wave-1 ideas stacked on SRC",
                      backbone=base.backbone, layers=base.layers, hidden=base.hidden,
                      heads=base.heads, dropout=base.dropout,
                      anchor=base.anchor, head=base.head, loss=base.loss,
                      data=dict(base.data))
    for win in winner_names:
        w = BY_NAME[win]
        for fld in ("backbone", "layers", "heads", "dropout", "reinject",
                    "concat_layers", "anchor", "head", "loss", "split_emission",
                    "stratified"):
            if getattr(w, fld) != getattr(base, fld):
                setattr(cfg, fld, getattr(w, fld))
        for k, v in w.data.items():
            if v != base.data.get(k):
                cfg.data[k] = v
    return cfg


def select(names=None, groups=None):
    """Resolve a roster selection from names and/or group letters."""
    if names:
        return [BY_NAME[n] for n in names]
    if groups:
        out = []
        for g in groups:
            out.extend(GROUPS[g.upper()])
        return out
    return ALL_MODELS
