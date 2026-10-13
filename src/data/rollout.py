"""Free-running multi-step rollout -- the evaluation that makes our numbers comparable.

Why this module exists
----------------------
Single-step scores on this dataset barely measure forecast skill.  At the native
3 h cadence only **4.8%** of cells change dust-event state from one step to the
next, so persistence alone reaches TS 0.586 and any reasonable model lands in the
0.6-0.7 band.  AI-DUST's headline TS of 0.33 is a *one-day* forecast and their
0.42 covers 24-48 h; putting our 3 h step beside those numbers would flatter us by
about the amount the task is easier.

A rollout closes that gap.  From one analysis state the model's own prediction is
fed back as the next input and the field is advanced K steps (K = 80 -> 10 days,
matching their Figs. 2-3), while meteorological forcing is read from ERA5 at each
future step -- mirroring AI-DUST's operational setup, where met comes from an AIFS
forecast rather than from the dust model.

What must be recomputed each step
---------------------------------
Feeding back a prediction is not just overwriting one channel.  Everything derived
from concentration goes stale:

  * ``log_pm10`` / ``log_pm25`` and every history channel  -> from the state buffer
  * ``transport`` (``div(VC)`` or ``u.grad C``)            -> re-derived on the grid
  * ``c_t`` and ``c_ref`` anchors                          -> re-derived from state

Met-, emission- and calendar-derived channels stay on their true values, because a
real forecast would also have those.  Leaving ``transport`` or ``c_ref`` on truth
would leak the answer into the rollout and silently inflate long-lead skill.

All S start times advance together: at lead step k, start i sits at absolute index
``t0_i + k``, and the node-level field arrays index straight through
(``fields.u[idxs]`` is ``[S, n_nodes]``), so each step costs S feature assemblies
and **one** batched model call rather than S calls.
"""
from collections import deque
from types import SimpleNamespace

import numpy as np
import torch

from src.data.saudi_dataset import PM25_EMISSION_FRACTION
from src.data.topologies import FeatureAssembler
from src.physics import transport as trans


class Rollout:
    """Advances a model K steps from each of a set of start times."""

    def __init__(self, fields, cfg, model, device, topology,
                 edge_index=None, pos=None, patch_builder=None):
        self.f = fields
        self.cfg = cfg
        self.model = model
        self.device = device
        self.topology = topology
        self.asm = FeatureAssembler(fields, cfg)
        self.edge_index = edge_index.to(device) if edge_index is not None else None
        self.pos = pos.to(device) if pos is not None else None
        self.pb = patch_builder

        ch = self.asm.channels
        self.ix_pm10 = ch.index("log_pm10")
        self.ix_pm25 = ch.index("log_pm25")
        self.ix_transport = ch.index("transport") if "transport" in ch else None
        self.hist_pm10 = [ch.index(f"log_pm10_-{j}") for j in range(1, cfg.window)]
        self.hist_pm25 = [ch.index(f"log_pm25_-{j}") for j in range(1, cfg.window)]
        self.n_targets = 2 if cfg.multitask else 1

    # ------------------------------------------------------------------ #
    def _to_grid(self, node_vals):
        """``[S, n_nodes]`` -> ``[S, H, W]``, zeros outside the valid mask."""
        s = node_vals.shape[0]
        grid = np.zeros((s, int(np.prod(self.f.grid_shape))), dtype="float32")
        grid[:, self.f.node_flat] = node_vals
        return grid.reshape(s, *self.f.grid_shape)

    def _to_nodes(self, grid):
        return grid.reshape(grid.shape[0], -1)[:, self.f.node_flat]

    def _physics(self, state, idxs):
        """Re-derive transport and the reference states from the *predicted* field."""
        f, cfg = self.f, self.cfg
        u = self._to_grid(f.u[idxs])
        v = self._to_grid(f.v[idxs])
        pm10_phys = self._to_grid(np.expm1(np.clip(state[:, :, 0], 0, None)))
        tend10 = self._to_grid(f.emis_tendency[idxs + 1])

        transport = self._to_nodes(trans.transport_tendency(
            pm10_phys, u, v, f.dx, f.dy, divergence=cfg.divergence))
        ref10 = self._to_nodes(trans.build_reference(
            cfg.reference_mode, pm10_phys, u, v, f.dx, f.dy, f.dt,
            emission_tendency=tend10, divergence=cfg.divergence, mask=f.mask))
        refs = [trans.to_log(ref10)]

        if self.n_targets == 2:
            pm25_phys = self._to_grid(np.expm1(np.clip(state[:, :, 1], 0, None)))
            ref25 = self._to_nodes(trans.build_reference(
                cfg.reference_mode, pm25_phys, u, v, f.dx, f.dy, f.dt,
                emission_tendency=tend10 * PM25_EMISSION_FRACTION,
                divergence=cfg.divergence, mask=f.mask))
            refs.append(trans.to_log(ref25))

        return transport, np.stack(refs, axis=-1).astype("float32")

    def _features(self, idxs, state, history):
        """Truth-assembled features with every concentration-derived channel replaced."""
        x = np.stack([self.asm.assemble(int(i)) for i in idxs])        # [S, N, C]

        x[:, :, self.ix_pm10] = state[:, :, 0]
        if self.n_targets == 2:
            x[:, :, self.ix_pm25] = state[:, :, 1]
        for k, col in enumerate(self.hist_pm10):
            if k < len(history):
                x[:, :, col] = history[-(k + 1)][:, :, 0]
        if self.n_targets == 2:
            for k, col in enumerate(self.hist_pm25):
                if k < len(history):
                    x[:, :, col] = history[-(k + 1)][:, :, 1]

        transport, c_ref = self._physics(state, idxs)
        if self.ix_transport is not None:
            x[:, :, self.ix_transport] = transport
        return x, state.copy(), c_ref

    def _batched_full(self, x, c_t, c_ref, idxs):
        """S independent domain graphs concatenated into one batched forward."""
        s, n = x.shape[0], x.shape[1]
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.device)
        offs = (torch.arange(s, device=self.device) * n).view(1, -1, 1)
        key = np.stack([self.asm.key_features(int(i)) for i in idxs])
        data = SimpleNamespace(
            x=t(x.reshape(s * n, -1)),
            key=t(key.reshape(s * n, -1)),
            edge_index=(self.edge_index.unsqueeze(1) + offs).reshape(2, -1),
            pos=self.pos.repeat(s, 1),
            c_t=t(c_t.reshape(s * n, -1)),
            c_ref=t(c_ref.reshape(s * n, -1)),
            ptr=torch.arange(s + 1, device=self.device) * n,
            emis_active=t(self.f.emis_active[idxs].reshape(-1, 1), torch.bool),
        )
        return self.model(data).cpu().numpy().reshape(s, n, -1)

    def _batched_patch(self, x, c_t, c_ref, idxs):
        """Patch topology: one frame of 251 patches per start, stacked."""
        outs = []
        for i, idx in enumerate(idxs):
            data = self.pb.frame_from(x[i], c_t[i], c_ref[i], int(idx))
            outs.append(self.model(data).cpu().numpy())
        return np.stack(outs)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def run(self, start_idxs, n_steps):
        """Advance ``n_steps`` from each start.

        Returns ``(pred, true)``, each ``[K, S, n_nodes, n_targets]``, log1p space.
        """
        self.model.eval()
        f = self.f
        starts = np.asarray(start_idxs, dtype=int)

        stack2 = lambda t: np.stack([f.pm10[t], f.pm25[t]], -1)[:, :, :self.n_targets]
        state = stack2(starts)
        history = deque(maxlen=max(1, self.cfg.window - 1))
        for j in range(self.cfg.window - 1, 0, -1):
            history.append(stack2(np.clip(starts - j, 0, f.n_steps - 1)))

        forward = self._batched_patch if self.topology == "patch" else self._batched_full
        preds, trues = [], []
        for k in range(n_steps):
            idxs = np.clip(starts + k, 0, f.n_steps - 2)
            x, c_t, c_ref = self._features(idxs, state, list(history))
            history.append(state.copy())
            state = forward(x, c_t, c_ref, idxs)
            preds.append(state.copy())
            trues.append(stack2(np.clip(starts + k + 1, 0, f.n_steps - 1)))

        return np.stack(preds), np.stack(trues)
