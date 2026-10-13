"""Improvement ablation for the PI-GNN dust model (no new data required).

Compares the current baseline against an "enhanced" configuration that targets
the root causes of the persistence-level performance:
  * temporal history window (lets the model perceive the tendency dC/dt),
  * physically-motivated features (wind speed, day-of-year season encoding),
  * deeper / wider spatial graph (2 flux hops, 8-neighbour connectivity).

Each config is trained with several seeds and evaluated on the held-out TEST
split, reporting skill vs the (deterministic) persistence baseline so we can tell
whether any gain is real or within seed noise.

Run from the repository root:  python experiments/gulf_pm10/ablation.py
"""
import os
import sys
import math

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.layers.flux import PhysicsFluxLayer
from src.training.trainer import Trainer

DATA = os.path.join(REPO_ROOT, "data")
SFC = os.path.join(DATA, "data_sfc.nc")
LAT, LON = 7, 12
CELLS = LAT * LON


# --------------------------------------------------------------------------- #
# Graph (4- or 8-neighbour), geographically-oriented positions.
# --------------------------------------------------------------------------- #
def build_graph(neighbors=4):
    offsets4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    offsets8 = offsets4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    offsets = offsets8 if neighbors == 8 else offsets4
    edges, pos = [], []
    for r in range(LAT):
        for c in range(LON):
            n = r * LON + c
            for dr, dc in offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < LAT and 0 <= nc < LON:
                    edges.append([n, nr * LON + nc])
            pos.append([c, (LAT - 1 - r)])
    return (torch.tensor(edges, dtype=torch.long).t().contiguous(),
            torch.tensor(pos, dtype=torch.float))


# --------------------------------------------------------------------------- #
# Dataset with temporal window + optional derived features.
# Feature layout keeps [pm10_t, u_t, v_t, t2m_t] at indices 0..3 so the physics
# loss / residual update (which read indices 0,1,2) work unchanged.
# --------------------------------------------------------------------------- #
class EnhancedDataset(Dataset):
    def __init__(self, era_files, window=1, resample="12h",
                 add_speed=False, add_doy=False, neighbors=4):
        super().__init__()
        self.window = window
        self.add_speed = add_speed
        self.add_doy = add_doy

        ds_pm10 = xr.open_dataset(SFC)
        eds = [xr.open_dataset(f) for f in era_files if os.path.exists(f)]
        ds_era = xr.concat(eds, dim="valid_time").sortby("valid_time")
        ds = xr.merge([ds_pm10, ds_era], join="inner").resample(valid_time=resample).mean()

        self.pm10 = np.nan_to_num(np.log1p(ds["pm10"].values * 1e9)).reshape(-1, CELLS)
        self.u = np.nan_to_num(ds["u10"].values).reshape(-1, CELLS)
        self.v = np.nan_to_num(ds["v10"].values).reshape(-1, CELLS)
        self.t2m = np.nan_to_num(ds["t2m"].values).reshape(-1, CELLS)
        doy = ds["valid_time"].dt.dayofyear.values.astype(np.float32)
        self.doy_sin = np.sin(2 * math.pi * doy / 365.25)
        self.doy_cos = np.cos(2 * math.pi * doy / 365.25)

        self.T = self.pm10.shape[0]
        self.edge_index, self.pos = build_graph(neighbors)

    def len(self):
        return self.T - self.window  # need history (window-1) and target (+1)

    def _assemble(self, idx):
        """idx = current-frame index; returns [CELLS, C] float32."""
        cols = [self.pm10[idx], self.u[idx], self.v[idx], self.t2m[idx]]
        for j in range(1, self.window):
            cols += [self.pm10[idx - j], self.u[idx - j], self.v[idx - j]]
        if self.add_speed:
            cols += [np.sqrt(self.u[idx] ** 2 + self.v[idx] ** 2)]
        if self.add_doy:
            cols += [np.full(CELLS, self.doy_sin[idx]), np.full(CELLS, self.doy_cos[idx])]
        return np.stack(cols, axis=1).astype(np.float32)

    @property
    def in_channels(self):
        return 4 + 3 * (self.window - 1) + (1 if self.add_speed else 0) + (2 if self.add_doy else 0)

    def get(self, i):
        idx = i + (self.window - 1)
        x = torch.tensor(self._assemble(idx))
        y = torch.tensor(self.pm10[idx + 1], dtype=torch.float32).view(-1, 1)
        return Data(x=x, edge_index=self.edge_index, pos=self.pos, y=y)

    def feature_stats(self):
        rows = [self._assemble(i + (self.window - 1)) for i in range(self.len())]
        feats = np.concatenate(rows, axis=0)
        return feats.mean(0).astype(np.float32), feats.std(0).astype(np.float32)


# --------------------------------------------------------------------------- #
# Multi-hop physics-flux model with train-fit normalisation buffers.
# --------------------------------------------------------------------------- #
class EnhancedModel(nn.Module):
    def __init__(self, in_channels, hidden=64, hops=1):
        super().__init__()
        self.hops = hops
        self.flux1 = PhysicsFluxLayer(in_channels, hidden)
        self.flux2 = PhysicsFluxLayer(hidden, hidden) if hops >= 2 else None
        self.head = nn.Linear(hidden, 1)
        self.register_buffer("feat_mean", torch.zeros(in_channels))
        self.register_buffer("feat_std", torch.ones(in_channels))
        nn.init.constant_(self.head.weight, 0.0)
        nn.init.constant_(self.head.bias, 0.0)

    def set_norm_stats(self, mean, std):
        with torch.no_grad():
            self.feat_mean.copy_(torch.as_tensor(mean))
            self.feat_std.copy_(torch.clamp(torch.as_tensor(std), min=1e-6))

    def forward(self, data):
        x, ei, pos = data.x, data.edge_index, data.pos
        pm10 = x[:, 0:1]
        u, v = x[:, 1:2], x[:, 2:3]
        xn = (x - self.feat_mean) / self.feat_std
        h = torch.relu(self.flux1(xn, ei, pos, u, v, pm10))
        if self.flux2 is not None:
            h = h + torch.relu(self.flux2(h, ei, pos, u, v, pm10))
        delta = self.head(h)
        return pm10 + delta, delta


# --------------------------------------------------------------------------- #
def metrics(pred, true):
    mse = float(np.mean((pred - true) ** 2))
    r2 = 1.0 - float(np.sum((true - pred) ** 2)) / float(np.sum((true - np.mean(true)) ** 2))
    return mse, r2


def evaluate(model, ds):
    model.eval()
    preds, trues, persist = [], [], []
    with torch.no_grad():
        for d in DataLoader(ds, batch_size=64, shuffle=False):
            p, _ = model(d)
            preds.append(p.numpy().ravel())
            trues.append(d.y.numpy().ravel())
            persist.append(d.x[:, 0:1].numpy().ravel())
    yhat, y, pp = map(np.concatenate, (preds, trues, persist))
    return metrics(yhat, y), metrics(pp, y)


def run_config(name, cfg, seeds, epochs=40):
    freq = cfg.get("resample", "12h")
    hops = cfg.get("_hops", 1)
    ds_cfg = {k: v for k, v in cfg.items() if k != "_hops"}
    dt = {"3h": 10800.0, "6h": 21600.0, "12h": 43200.0}[freq]
    train_era = [os.path.join(DATA, f"era5_{y}.nc") for y in (2003, 2004, 2005)]
    val_era = [os.path.join(DATA, "era5_2006.nc")]
    test_era = [os.path.join(DATA, "era5_2007.nc")]

    train_ds = EnhancedDataset(train_era, **ds_cfg)
    val_ds = EnhancedDataset(val_era, **ds_cfg)
    test_ds = EnhancedDataset(test_era, **ds_cfg)
    mean, std = train_ds.feature_stats()
    physics_cfg = {"delta_t": dt, "dx": 74880.0, "dy": 83490.0, "grid_size": (LAT, LON)}

    results = []
    persist_m = None
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = EnhancedModel(train_ds.in_channels, hidden=64, hops=hops)
        model.set_norm_stats(mean, std)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(model, opt, torch.device("cpu"), physics_cfg=physics_cfg)

        best_val, best_state = float("inf"), None
        tl = DataLoader(train_ds, batch_size=32, shuffle=True)
        vl = DataLoader(val_ds, batch_size=64, shuffle=False)
        for _ in range(epochs):
            trainer.train_epoch(tl)
            _, vmse = trainer.validate(vl)
            if vmse < best_val:
                best_val = vmse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        (m_mse, m_r2), (p_mse, p_r2) = evaluate(model, test_ds)
        persist_m = (p_mse, p_r2)
        results.append((m_mse, m_r2))

    mse = np.array([r[0] for r in results])
    r2 = np.array([r[1] for r in results])
    skill = (persist_m[0] - mse) / persist_m[0] * 100
    print(f"\n## {name}  (freq={freq}, in_ch={train_ds.in_channels}, "
          f"hops={cfg.get('_hops',1)}, nbr={cfg.get('neighbors',4)}, seeds={list(seeds)})")
    print(f"   PI-GNN  MSE {mse.mean():.4f} +/- {mse.std():.4f} | R2 {r2.mean():.4f} +/- {r2.std():.4f}")
    print(f"   Persist MSE {persist_m[0]:.4f}              | R2 {persist_m[1]:.4f}")
    print(f"   Skill vs persistence: {skill.mean():+.2f}% +/- {skill.std():.2f}%  (MSE)")
    return name, mse.mean(), r2.mean(), skill.mean(), skill.std(), persist_m


def main():
    seeds = (0, 1, 2)
    rows = []
    rows.append(run_config("A. Baseline (current)",
                           {"window": 1, "resample": "12h", "neighbors": 4, "_hops": 1},
                           seeds))
    rows.append(run_config("B. Enhanced (history+feats+2hop+8nbr)",
                           {"window": 3, "resample": "12h", "add_speed": True, "add_doy": True,
                            "neighbors": 8, "_hops": 2},
                           seeds))
    rows.append(run_config("C. Enhanced @ 6h resolution",
                           {"window": 3, "resample": "6h", "add_speed": True, "add_doy": True,
                            "neighbors": 8, "_hops": 2},
                           (0,)))

    print("\n" + "=" * 64)
    print("SUMMARY (held-out test 2007; skill = MSE reduction vs persistence)")
    print("=" * 64)
    for name, mse, r2, sk, sks, _ in rows:
        print(f"  {name:<42} R2={r2:.4f}  skill={sk:+.2f}%")


if __name__ == "__main__":
    main()
