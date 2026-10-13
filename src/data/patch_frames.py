"""Vectorised patch assembly for evaluation.

Scoring a patch model on the same points as a full-domain model means running it
once per (timestep, centre cell) -- ~733k patches for a 4-year test set at
``--eval-stride 4``.  Going through ``PatchDataset`` + the PyG loader does that as
733k individual ``__getitem__`` calls plus 733k collations, which dominates the
entire benchmark (measured: slower than all training combined).

Every patch in a timestep is drawn from the *same* assembled frame, so the whole
frame can be gathered in one fancy-index and shipped as a single pre-batched
graph.  That turns 733k Python-level assemblies into ~2.9k, with the per-patch
graph structure precomputed once.

The object returned by :meth:`PatchFrameBuilder.frame` carries exactly the
attributes :class:`src.models.unified.UnifiedDustModel` reads, so the model does
not know or care that it came from here rather than from a PyG ``Batch``.
"""
from types import SimpleNamespace

import numpy as np
import torch

from src.data.topologies import FeatureAssembler
from src.utils.graph import build_dust_graph


class PatchFrameBuilder:
    """Builds all patches of one timestep as a single batched graph."""

    def __init__(self, fields, cfg, gather, patch, neighbors=4, device=None):
        self.f = fields
        self.cfg = cfg
        self.asm = FeatureAssembler(fields, cfg)
        self.gather = gather
        self.patch = patch
        self.n_centres = gather.shape[0]
        self.nodes_per_patch = patch * patch
        self.centre = (patch // 2) * patch + (patch // 2)
        self.device = device or torch.device("cpu")
        self._relief_col = (self.asm.channels.index("rel_elev") if cfg.relief else None)

        edge_index, pos = build_dust_graph(patch, patch, neighbors=neighbors)

        # Replicate the patch graph once per centre, offsetting node ids per patch.
        offsets = (torch.arange(self.n_centres, dtype=torch.long)
                   * self.nodes_per_patch).view(1, -1, 1)
        self.edge_index = (edge_index.unsqueeze(1) + offsets).reshape(2, -1).to(self.device)
        self.pos = pos.repeat(self.n_centres, 1).to(self.device)
        self.ptr = (torch.arange(self.n_centres + 1, dtype=torch.long)
                    * self.nodes_per_patch).to(self.device)
        self.centre_idx = torch.full((self.n_centres,), self.centre,
                                     dtype=torch.long, device=self.device)

        # Gather bookkeeping, hoisted out of the per-frame path.
        self._valid = gather >= 0
        self._safe = np.where(self._valid, gather, 0)
        self._keep = self._valid[..., None].astype("float32")

    def _expand(self, node_array):
        """``[n_nodes, C]`` -> ``[n_centres * patch^2, C]``, zero-padded outside."""
        out = node_array[self._safe] * self._keep
        return out.reshape(-1, node_array.shape[-1])

    def frame_from(self, x_all, c_t_all, c_ref_all, idx):
        """Build a patch frame from *supplied* node arrays instead of the truth.

        Used by the rollout, where the concentration-derived channels come from the
        model's own previous prediction rather than from the cached fields.
        """
        x = self._expand(np.ascontiguousarray(x_all))
        if self._relief_col is not None:
            col = self._relief_col
            rel = (x_all[self._safe, col] - x_all[:, col][:, None]) * self._keep[..., 0]
            x[:, col] = rel.reshape(-1)

        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.device)
        return SimpleNamespace(
            x=t(x),
            key=t(self._expand(self.asm.key_features(idx))),
            edge_index=self.edge_index,
            pos=self.pos,
            ptr=self.ptr,
            centre_idx=self.centre_idx,
            c_t=t(c_t_all),
            c_ref=t(c_ref_all),
            emis_active=t(self.f.emis_active[idx][:, None], torch.bool),
        )

    def frame(self, idx):
        """All patches for timestep ``idx`` as one batched graph on the target device."""
        x_all = self.asm.assemble(idx)
        key_all = self.asm.key_features(idx)
        c_t_all, c_ref_all = self.asm.anchors(idx)
        y_all = self.asm.target(idx)

        x = self._expand(x_all)
        if self._relief_col is not None:
            col = self._relief_col
            rel = (x_all[self._safe, col] - x_all[:, col][:, None]) * self._keep[..., 0]
            x[:, col] = rel.reshape(-1)

        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.device)
        return SimpleNamespace(
            x=t(x),
            key=t(self._expand(key_all)),
            edge_index=self.edge_index,
            pos=self.pos,
            ptr=self.ptr,
            centre_idx=self.centre_idx,
            c_t=t(c_t_all),
            c_ref=t(c_ref_all),
            y=t(y_all),
            emis_active=t(self.f.emis_active[idx][:, None], torch.bool),
        )


@torch.no_grad()
def predict_frames(model, builder, idxs):
    """Run a patch model over whole timesteps -> ``([T, N, targets], [T, N, targets])``."""
    model.eval()
    preds, trues = [], []
    for idx in idxs:
        data = builder.frame(int(idx))
        preds.append(model(data).cpu().numpy())
        trues.append(data.y.cpu().numpy())
    return np.stack(preds), np.stack(trues)
