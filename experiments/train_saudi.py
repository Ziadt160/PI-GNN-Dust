"""Multi-task PI-GNN on the full Saudi domain (PM10 + PM2.5) from saudi_merged.nc.

Self-contained end-to-end pipeline for the new 22-year dataset:
  * masked irregular graph over the ~251 valid land cells,
  * features: current [log_pm10, u, v, log_pm2.5, t2m, rh, log1p(tp), speed] +
    temporal history + day-of-year, all standardised on TRAIN only,
  * two residual heads predicting log-space PM10 and PM2.5 at t+1,
  * temporal split: train 2003-2018 / val 2019-2020 / test 2021-2024.

This first run is data-driven (physics-off) to establish the multi-task baseline
on the new domain; emission/deposition physics is added in a later step.

Run from the repository root:  python experiments/train_saudi.py
"""
import os
import sys
import math
import time

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from src.layers.flux import PhysicsFluxLayer
from src.utils.graph import build_masked_graph

MERGED = r"D:/Quantum Projects/DustProject/data/ALL DATA SAUDI/saudi_merged.nc"
WINDOW = 3
RESAMPLE = os.environ.get("SAUDI_RESAMPLE", "12h")
EPOCHS = int(os.environ.get("SAUDI_EPOCHS", "30"))


def load_fields():
    print(f"Loading {MERGED} (resample {RESAMPLE}) ...", flush=True)
    ds = xr.open_dataset(MERGED)
    mask = ds.valid_mask.values
    lat_dim, lon_dim = mask.shape
    edge_index, pos, node_flat = build_masked_graph(mask, neighbors=8)
    print(f"  grid {lat_dim}x{lon_dim} | nodes {len(node_flat)} | edges {edge_index.shape[1]}", flush=True)

    factor = {"3h": 1, "6h": 2, "12h": 4}[RESAMPLE]
    tv_full = ds.valid_time.values
    T_trim = (len(tv_full) // factor) * factor
    tv = tv_full[:T_trim:factor]  # block-start timestamps

    def field(var, transform=None):
        # Extract valid nodes at native 3h cadence, then fast NumPy block-average.
        a = np.nan_to_num(ds[var].values).reshape(len(tv_full), -1)[:, node_flat]
        a = a[:T_trim].reshape(-1, factor, a.shape[1]).mean(axis=1)
        return (transform(a) if transform else a).astype("float32")

    pm10 = field("pm10", lambda a: np.log1p(a * 1e9))
    pm25 = field("pm2p5", lambda a: np.log1p(a * 1e9))
    u10 = field("u10")
    v10 = field("v10")
    t2m = field("t2m")
    rh = field("rh")
    tp = field("tp", lambda a: np.log1p(np.clip(a, 0, None) * 1e3))
    speed = np.sqrt(u10 ** 2 + v10 ** 2).astype("float32")
    years = tv.astype("datetime64[Y]").astype(int) + 1970
    doy = (tv.astype("datetime64[D]") - tv.astype("datetime64[Y]")).astype(int) + 1
    dsin = np.sin(2 * math.pi * doy / 365.25).astype("float32")
    dcos = np.cos(2 * math.pi * doy / 365.25).astype("float32")
    fields = dict(pm10=pm10, pm25=pm25, u10=u10, v10=v10, t2m=t2m, rh=rh, tp=tp,
                  speed=speed, dsin=dsin, dcos=dcos)
    return fields, years, edge_index, pos, len(node_flat)


class SaudiDataset(torch.utils.data.Dataset):
    def __init__(self, fields, idxs, edge_index, pos, window=WINDOW):
        self.f = fields; self.idxs = idxs; self.edge_index = edge_index
        self.pos = pos; self.window = window; self.N = pos.shape[0]

    def __len__(self):
        return len(self.idxs)

    def assemble(self, idx):
        f = self.f
        cols = [f["pm10"][idx], f["u10"][idx], f["v10"][idx], f["pm25"][idx],
                f["t2m"][idx], f["rh"][idx], f["tp"][idx], f["speed"][idx]]
        for j in range(1, self.window):
            cols += [f["pm10"][idx - j], f["pm25"][idx - j], f["u10"][idx - j], f["v10"][idx - j]]
        cols += [np.full(self.N, f["dsin"][idx], "float32"), np.full(self.N, f["dcos"][idx], "float32")]
        return np.stack(cols, axis=1).astype("float32")

    @property
    def in_channels(self):
        return 8 + 4 * (self.window - 1) + 2

    def __getitem__(self, k):
        idx = self.idxs[k]
        x = torch.from_numpy(self.assemble(idx))
        y = torch.from_numpy(np.stack([self.f["pm10"][idx + 1], self.f["pm25"][idx + 1]], axis=1))
        return Data(x=x, edge_index=self.edge_index, pos=self.pos, y=y)


class MultiTaskModel(nn.Module):
    def __init__(self, in_channels, hidden=64, hops=2):
        super().__init__()
        self.flux_in = PhysicsFluxLayer(in_channels, hidden)
        self.flux_hidden = nn.ModuleList(PhysicsFluxLayer(hidden, hidden) for _ in range(hops - 1))
        self.head = nn.Linear(hidden, 2)  # delta for [pm10, pm2.5]
        self.register_buffer("mean", torch.zeros(in_channels))
        self.register_buffer("std", torch.ones(in_channels))
        nn.init.constant_(self.head.weight, 0.0)
        nn.init.constant_(self.head.bias, 0.0)

    def set_stats(self, m, s):
        self.mean.copy_(torch.as_tensor(m)); self.std.copy_(torch.clamp(torch.as_tensor(s), min=1e-6))

    def forward(self, data):
        x, ei, pos = data.x, data.edge_index, data.pos
        pm10, u, v = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        pm25 = x[:, 3:4]
        xn = (x - self.mean) / self.std
        h = torch.relu(self.flux_in(xn, ei, pos, u, v, pm10))
        for fl in self.flux_hidden:
            h = h + torch.relu(fl(h, ei, pos, u, v, pm10))
        delta = self.head(h)
        cur = torch.cat([pm10, pm25], dim=1)
        return cur + delta


def metrics(pred, true):
    mse = np.mean((pred - true) ** 2, axis=0)
    r2 = 1 - np.sum((true - pred) ** 2, axis=0) / np.sum((true - true.mean(0)) ** 2, axis=0)
    return mse, r2


def main():
    torch.manual_seed(0); np.random.seed(0)
    fields, years, edge_index, pos, N = load_fields()
    T = fields["pm10"].shape[0]

    def block_idxs(y0, y1):
        lo = np.argmax(years >= y0)
        hi = T - 1 - np.argmax(years[::-1] <= y1)
        return [i for i in range(lo + WINDOW - 1, hi) if i + 1 <= hi]

    tr_idx, va_idx, te_idx = block_idxs(2003, 2018), block_idxs(2019, 2020), block_idxs(2021, 2024)
    print(f"  samples: train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}")

    train_ds = SaudiDataset(fields, tr_idx, edge_index, pos)
    val_ds = SaudiDataset(fields, va_idx, edge_index, pos)
    test_ds = SaudiDataset(fields, te_idx, edge_index, pos)

    # Train-only normalisation from a subsample.
    sub = np.stack([train_ds.assemble(tr_idx[k]) for k in range(0, len(tr_idx), 20)]).reshape(-1, train_ds.in_channels)
    mean, std = sub.mean(0), sub.std(0)

    model = MultiTaskModel(train_ds.in_channels)
    model.set_stats(mean, std)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    tl = DataLoader(train_ds, batch_size=32, shuffle=True)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False)

    best, state = float("inf"), None
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for d in tl:
            opt.zero_grad()
            loss = mse(model(d), d.y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vtot = 0; nb = 0
        with torch.no_grad():
            for d in vl:
                vtot += mse(model(d), d.y).item(); nb += 1
        vmse = vtot / nb
        if vmse < best:
            best, state = vmse, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep+1:3d}/{EPOCHS}  val_mse={vmse:.4f}")
    print(f"  trained in {time.time()-t0:.0f}s")

    model.load_state_dict(state); model.eval()
    preds, trues, cur = [], [], []
    with torch.no_grad():
        for d in DataLoader(test_ds, batch_size=64, shuffle=False):
            preds.append(model(d).numpy()); trues.append(d.y.numpy())
            cur.append(torch.cat([d.x[:, 0:1], d.x[:, 3:4]], 1).numpy())
    P, Y, C = map(lambda a: np.concatenate(a), (preds, trues, cur))

    m_mse, m_r2 = metrics(P, Y)
    p_mse, p_r2 = metrics(C, Y)
    print("\n=== Saudi multi-task test (2021-2024), log-space ===")
    for k, name in enumerate(["PM10", "PM2.5"]):
        skill = (p_mse[k] - m_mse[k]) / p_mse[k] * 100
        print(f"  {name:<6} model R2={m_r2[k]:.4f} MSE={m_mse[k]:.4f} | "
              f"persist R2={p_r2[k]:.4f} | skill {skill:+.2f}%")


if __name__ == "__main__":
    main()
