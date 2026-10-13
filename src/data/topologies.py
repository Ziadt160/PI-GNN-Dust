"""Feature assembly, the two graph topologies, and AI-DUST's stratified sampler.

Built on the cached node-level fields from :class:`src.data.saudi_dataset.SaudiFields`.

``FullDomainDataset``  your layout -- one graph per timestep over all 251 cells,
                       every cell predicted in a single pass.
``PatchDataset``       AI-DUST's layout -- a P x P patch per (timestep, centre
                       cell), only the centre node predicted, zero-padded at the
                       domain edge.

Both emit the *same* fields on the PyG ``Data`` object (``x``, ``key``, ``c_t``,
``c_ref``, ``y``, ``emis_active``), so the model assembly does not need to know
which topology it is being fed.
"""
import numpy as np
import torch
from torch_geometric.data import Data

from src.data.saudi_dataset import SaudiFields


class FeatureAssembler:
    """Builds the node feature matrix for a timestep, per the config switches.

    Channel order is fixed and named: the flux layers need the *un-normalised*
    wind and concentration by index, and the patch topology needs to locate the
    relief channel to re-reference it to the patch centre.
    """

    def __init__(self, fields: SaudiFields, cfg=None):
        # `cfg` may override `fields.cfg` so one cached SaudiFields can serve
        # several models whose views differ only in assembly-time switches.
        self.f = fields
        self.cfg = cfg if cfg is not None else fields.cfg
        self.channels = self._channel_names()

    def _channel_names(self):
        cfg = self.cfg
        names = ["log_pm10", "u", "v", "log_pm25", "t2m", "rh", "log_tp", "speed"]
        for j in range(1, cfg.window):
            names += [f"log_pm10_-{j}", f"log_pm25_-{j}", f"u_-{j}", f"v_-{j}"]
        if cfg.future_met:
            names += ["u_+1", "v_+1", "t2m_+1", "rh_+1", "speed_+1", "log_tp_+1"]
        if cfg.emission_features:
            names += ["ustar_+1", "log_emission_+1", "emis_tendency_+1"]
        if cfg.relief:
            names += ["rel_elev"]
        if cfg.transport_feature:
            names += ["transport"]
        if cfg.physics_features:
            # Four things the model previously could not represent at all, rather
            # than represented poorly. See src/data/saudi_dataset.py for each.
            names += ["hour_sin", "hour_cos",      # diurnal cycle: was absent
                      "blh_+1",                    # mixing depth at the valid time
                      "ustar_excess_+1",           # how far above the saltation switch
                      "wetness"]                   # days-long soil drying memory
        # Stability block: present only when the levels archive is complete, so a
        # partial download changes nothing rather than silently changing the
        # feature set. `has_levels` is the single source of truth for both the
        # names and the assembly below.
        if self.has_levels:
            names += [f"{c}_+1" for c in self.f.level_channels]
        names += ["doy_sin", "doy_cos"]
        return names

    @property
    def has_levels(self):
        return bool(self.cfg.pressure_levels
                    and getattr(self.f, "levels", None) is not None)

    @property
    def n_channels(self):
        return len(self.channels)

    def assemble(self, idx, met_offset=1):
        """``[n_nodes, n_channels]`` features for sample ``idx``.

        ``met_offset`` selects which future step the forward-meteorology channels
        are drawn from.  It must equal the forecast horizon: a 24 h dust forecast
        needs the wind at +24 h, not at +3 h, because the dust follows weather that
        a separate NWP system forecasts.  AI-DUST does the same thing, feeding
        meteorology at ``t`` and ``t+dt`` from AIFS.  Leaving this at 1 while
        predicting 24 h ahead is what reduced our 24 h threat score to 0.025.
        """
        f, cfg = self.f, self.cfg
        nxt = min(idx + int(met_offset), f.n_steps - 1)
        u, v = (f.u_courant, f.v_courant) if cfg.courant else (f.u, f.v)

        cols = [f.pm10[idx], u[idx], v[idx], f.pm25[idx],
                f.t2m[idx], f.rh[idx], f.tp[idx], f.speed[idx]]
        for j in range(1, cfg.window):
            cols += [f.pm10[idx - j], f.pm25[idx - j], u[idx - j], v[idx - j]]
        if cfg.future_met:
            cols += [u[nxt], v[nxt], f.t2m[nxt], f.rh[nxt], f.speed[nxt], f.tp[nxt]]
        if cfg.emission_features:
            cols += [f.ustar[nxt], f.emission[nxt], f.emis_tendency[nxt]]
        if cfg.relief:
            cols += [f.elev]
        if cfg.transport_feature:
            cols += [f.transport[idx]]
        if cfg.physics_features:
            # Mixing depth and threshold excess are taken at `nxt`, the valid
            # time, for the same reason the wind is: they describe the atmosphere
            # when the forecast lands, not when it was issued. Wetness is taken at
            # `idx` because it is a memory of rain that has ALREADY fallen.
            cols += [np.full(f.n_nodes, f.hsin[nxt], "float32"),
                     np.full(f.n_nodes, f.hcos[nxt], "float32"),
                     f.blh[nxt], f.ustar_excess[nxt], f.wetness[idx]]
        if self.has_levels:
            # At `nxt`, not at `idx`: stability decides whether dust reaching the
            # station is mixed to eye level, and that is a property of the
            # atmosphere at the VALID time, not at initialisation. Same reasoning
            # -- and the same `met_offset` -- as the future-meteorology channels.
            cols += [f.levels[nxt, c] for c in range(f.levels.shape[1])]
        cols += [np.full(f.n_nodes, f.dsin[idx], "float32"),
                 np.full(f.n_nodes, f.dcos[idx], "float32")]
        return np.stack(cols, axis=1).astype("float32")

    def key_features(self, idx):
        """The three inputs AI-DUST re-injects at each layer: transport, emission, C_t."""
        f = self.f
        return np.stack([f.transport[idx], f.emis_tendency[idx + 1], f.pm10[idx]],
                        axis=1).astype("float32")

    def anchors(self, idx):
        """``(C_t, C_ref)``, each ``[n_nodes, n_targets]`` in log1p space."""
        f = self.f
        if self.cfg.multitask:
            c_t = np.stack([f.pm10[idx], f.pm25[idx]], axis=1)
            c_ref = np.stack([f.c_ref10[idx], f.c_ref25[idx]], axis=1)
        else:
            c_t = f.pm10[idx][:, None]
            c_ref = f.c_ref10[idx][:, None]
        return c_t.astype("float32"), c_ref.astype("float32")

    def target(self, idx):
        f = self.f
        if self.cfg.multitask:
            return np.stack([f.pm10[idx + 1], f.pm25[idx + 1]], axis=1).astype("float32")
        return f.pm10[idx + 1][:, None].astype("float32")


def build_patch_gather(fields: SaudiFields, patch: int):
    """Index map ``[n_nodes, patch*patch]`` from each centre cell to its patch members.

    Entries are node indices, or ``-1`` where the patch falls outside the domain or
    onto an invalid cell; those positions are zero-filled at assembly time, which is
    AI-DUST's "pad the missing neighbouring values with zeros".
    """
    h, w = fields.grid_shape
    node_of_flat = np.full(h * w, -1, dtype=np.int64)
    node_of_flat[fields.node_flat] = np.arange(fields.n_nodes)

    half = patch // 2
    rows, cols = np.divmod(fields.node_flat, w)
    dr, dc = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1),
                         indexing="ij")
    rr = rows[:, None] + dr.ravel()[None, :]
    cc = cols[:, None] + dc.ravel()[None, :]
    inside = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    flat = np.where(inside, rr * w + cc, 0)
    return np.where(inside, node_of_flat[flat], -1).astype(np.int64)


class FullDomainDataset(torch.utils.data.Dataset):
    """Your topology: one graph per timestep, all 251 cells predicted at once."""

    def __init__(self, fields: SaudiFields, idxs, edge_index, pos, cfg=None):
        self.f = fields
        self.cfg = cfg if cfg is not None else fields.cfg
        self.asm = FeatureAssembler(fields, self.cfg)
        self.idxs = np.asarray(idxs)
        self.edge_index = edge_index
        self.pos = pos

    @property
    def n_channels(self):
        return self.asm.n_channels

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        idx = int(self.idxs[k])
        c_t, c_ref = self.asm.anchors(idx)
        d = Data(
            x=torch.from_numpy(self.asm.assemble(idx)),
            edge_index=self.edge_index,
            pos=self.pos,
            y=torch.from_numpy(self.asm.target(idx)),
        )
        d.key = torch.from_numpy(self.asm.key_features(idx))
        d.c_t = torch.from_numpy(c_t)
        d.c_ref = torch.from_numpy(c_ref)
        d.emis_active = torch.from_numpy(self.f.emis_active[idx][:, None].copy())
        return d


class PatchDataset(torch.utils.data.Dataset):
    """AI-DUST's topology: one P x P patch per (timestep, centre cell).

    Only the centre node is a prediction target.  Samples are ``(timestep, centre)``
    pairs, which is what makes AI-DUST's stratified sampling expressible: the
    training set can be balanced by the centre cell's own emission state,
    concentration bin and terrain relief.

    Consecutive samples from the same timestep reuse one assembled frame, so
    drawing all 251 centres for a step costs one assembly rather than 251.
    """

    def __init__(self, fields: SaudiFields, pairs, edge_index, pos, gather, patch,
                 cfg=None):
        self.f = fields
        self.cfg = cfg if cfg is not None else fields.cfg
        self.asm = FeatureAssembler(fields, self.cfg)
        self.pairs = np.asarray(pairs)          # [S, 2] = (time index, centre node)
        self.edge_index = edge_index
        self.pos = pos
        self.gather = gather
        self.patch = patch
        self.centre = (patch // 2) * patch + (patch // 2)
        self._relief_col = (self.asm.channels.index("rel_elev")
                            if self.cfg.relief else None)
        self._cache_idx = -1
        self._cache = None

    @property
    def n_channels(self):
        return self.asm.n_channels

    def __len__(self):
        return len(self.pairs)

    def _frame(self, idx):
        if self._cache_idx != idx:
            c_t, c_ref = self.asm.anchors(idx)
            self._cache = (self.asm.assemble(idx), self.asm.key_features(idx),
                           c_t, c_ref, self.asm.target(idx))
            self._cache_idx = idx
        return self._cache

    def __getitem__(self, k):
        idx, centre = int(self.pairs[k][0]), int(self.pairs[k][1])
        x_all, key_all, c_t_all, c_ref_all, y_all = self._frame(idx)

        g = self.gather[centre]
        valid = g >= 0
        safe = np.where(valid, g, 0)
        keep = valid[:, None].astype("float32")

        x = x_all[safe] * keep                       # zero-pad outside the domain
        key = key_all[safe] * keep
        if self._relief_col is not None:             # relief is patch-centre relative
            col = self._relief_col
            x[:, col] = (x_all[safe, col] - x_all[centre, col]) * keep[:, 0]

        d = Data(
            x=torch.from_numpy(x),
            edge_index=self.edge_index,
            pos=self.pos,
            y=torch.from_numpy(y_all[centre][None, :].copy()),      # centre node only
        )
        d.key = torch.from_numpy(key)
        d.c_t = torch.from_numpy(c_t_all[centre][None, :].copy())
        d.c_ref = torch.from_numpy(c_ref_all[centre][None, :].copy())
        d.centre_idx = torch.tensor([self.centre], dtype=torch.long)
        d.emis_active = torch.from_numpy(
            np.array([[bool(self.f.emis_active[idx, centre])]]))
        return d


# --------------------------------------------------------------------------- #
# AI-DUST stratified sampling
# --------------------------------------------------------------------------- #
CONC_BINS = (100.0, 500.0, 1000.0)          # ug/m^3 -- AI-DUST's inactive-region bins
RELIEF_MIN_FRACTION = 0.2                   # >= 1/5 of samples above the relief cut
RELIEF_THRESHOLD_M = 500.0


def stratified_pairs(fields: SaudiFields, idxs, n_samples, seed=0):
    """AI-DUST's sampling design (their Training-data section).

    Balances (1) emission-active vs inactive centre cells and (2) concentration
    bins, then tops up until at least a fifth of samples sit on high-relief cells.

    Why it matters here: uniform sampling over ~46k timesteps x 251 cells is
    dominated by the clean, flat, non-emitting majority -- exactly the regime that
    TS does not score.  A model tuned on that majority can post a respectable R^2
    and a near-zero threat score.
    """
    rng = np.random.default_rng(seed)
    idxs = np.asarray(idxs)

    pm_phys = np.expm1(fields.pm10[idxs])                       # [T, N] ug/m^3
    active = fields.emis_active[idxs]
    conc_bin = np.digitize(pm_phys, CONC_BINS)                  # 0..3
    high_relief = np.abs(fields.elev) > RELIEF_THRESHOLD_M      # [N]

    t_grid, n_grid = np.meshgrid(np.arange(len(idxs)), np.arange(fields.n_nodes),
                                 indexing="ij")
    strata = []
    for a in (False, True):
        for b in range(len(CONC_BINS) + 1):
            sel = (active == a) & (conc_bin == b)
            if sel.any():
                strata.append(np.stack([idxs[t_grid[sel]], n_grid[sel]], axis=1))

    per = max(1, n_samples // max(1, len(strata)))
    picked = []
    for cells in strata:
        take = min(per, len(cells))
        picked.append(cells[rng.choice(len(cells), take, replace=False)])
    pairs = np.concatenate(picked)

    if high_relief.any():
        have = high_relief[pairs[:, 1]].mean()
        need = int((RELIEF_MIN_FRACTION - have) * len(pairs))
        if need > 0:
            relief_nodes = np.flatnonzero(high_relief)
            extra = np.stack([rng.choice(idxs, need), rng.choice(relief_nodes, need)],
                             axis=1)
            pairs = np.concatenate([pairs, extra])

    return pairs[rng.permutation(len(pairs))]


def uniform_pairs(fields: SaudiFields, idxs, n_samples, seed=0):
    """Unstratified control, so `+stratified` measures the sampling design alone."""
    rng = np.random.default_rng(seed)
    t = rng.choice(np.asarray(idxs), n_samples)
    n = rng.integers(0, fields.n_nodes, n_samples)
    return np.stack([t, n], axis=1)
