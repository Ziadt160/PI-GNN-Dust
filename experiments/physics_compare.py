"""Does COMPLETING the physics help, on the new Saudi domain?

Compares three physics modes for the multi-task PI-GNN, all else equal:
  * none       - pure data-driven (MSE only)            [the strong baseline]
  * advection  - residual of dC/dt + u.grad C = 0       [the 'broken' law: ignores sources/sinks]
  * complete   - residual of dC/dt + u.grad C = E - D   [emission + deposition source/sink]

Emission/deposition are learnable, physically-structured per-step contributions:
  E = softplus(a) * (wind_speed / mean)^3            (dust uplift, ~speed^3)
  D = softplus(d) + softplus(w) * precip             (dry settling + wet scavenging)

If completing the law lets physics stop hurting (or start helping), that validates
the whole 'physics-informed' premise once the right physics + data are present.

Run from the repo root:  python -u experiments/physics_compare.py
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.loader import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
from src.layers.flux import PhysicsFluxLayer
from src.utils.graph import build_masked_graph
from experiments.train_saudi import load_fields, SaudiDataset, metrics, WINDOW, MERGED

DT, DY, DX = 43200.0, 83490.0, 76000.0   # 12 h; 0.75 deg spacing near ~24 N
EPOCHS = int(os.environ.get("PHYS_EPOCHS", "20"))


class PhysModel(nn.Module):
    def __init__(self, in_channels, hidden=64, hops=2):
        super().__init__()
        self.flux_in = PhysicsFluxLayer(in_channels, hidden)
        self.flux_hidden = nn.ModuleList(PhysicsFluxLayer(hidden, hidden) for _ in range(hops - 1))
        self.head = nn.Linear(hidden, 2)
        self.register_buffer("mean", torch.zeros(in_channels))
        self.register_buffer("std", torch.ones(in_channels))
        # learnable, per-pollutant emission/deposition coefficients
        self.emit = nn.Parameter(torch.zeros(2))
        self.dry = nn.Parameter(torch.zeros(2))
        self.wet = nn.Parameter(torch.zeros(2))
        nn.init.constant_(self.head.weight, 0.0)
        nn.init.constant_(self.head.bias, 0.0)

    def set_stats(self, m, s):
        self.mean.copy_(torch.as_tensor(m)); self.std.copy_(torch.clamp(torch.as_tensor(s), min=1e-6))

    def forward(self, data):
        x, ei, pos = data.x, data.edge_index, data.pos
        pm10, u, v, pm25 = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]
        xn = (x - self.mean) / self.std
        h = torch.relu(self.flux_in(xn, ei, pos, u, v, pm10))
        for fl in self.flux_hidden:
            h = h + torch.relu(fl(h, ei, pos, u, v, pm10))
        return torch.cat([pm10, pm25], dim=1) + self.head(h)


def grad_nodes(Cnode, rows, cols, B, H, W):
    """Reconstruct the (masked) grid, central-difference, gather back. NaN at boundaries."""
    N = rows.shape[0]
    grid = torch.full((B, H, W), float("nan"), device=Cnode.device)
    grid[:, rows, cols] = Cnode.view(B, N)
    gy, gx = torch.gradient(grid, spacing=(DY, DX), dim=(1, 2))
    gn = (-gy)[:, rows, cols].reshape(B * N, 1)   # north = -d/drow (row increases south)
    ge = gx[:, rows, cols].reshape(B * N, 1)      # east  = d/dcol
    return gn, ge


def physics_residual(x, pred, model, mode, rows, cols, H, W, mean_speed):
    N = rows.shape[0]; B = x.shape[0] // N
    u, v, speed, tp = x[:, 1:2], x[:, 2:3], x[:, 7:8], x[:, 6:7]
    sq, cnt = 0.0, 0
    for k, (Ccur, Cnext) in enumerate([(x[:, 0:1], pred[:, 0:1]), (x[:, 3:4], pred[:, 1:2])]):
        gn, ge = grad_nodes(Ccur, rows, cols, B, H, W)
        adv = u * ge + v * gn
        if mode == "complete":
            E = F.softplus(model.emit[k]) * (speed / mean_speed) ** 3
            D = F.softplus(model.dry[k]) + F.softplus(model.wet[k]) * tp
            src = E - D
        else:
            src = 0.0
        R = (Cnext - Ccur) + adv * DT - src
        fin = torch.isfinite(R)
        sq = sq + (R[fin] ** 2).sum(); cnt += int(fin.sum())
    return sq / max(cnt, 1)


def train_eval(mode, fields, splits, edge_index, pos, N, rows, cols, H, W, mean_speed, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_idx, va_idx, te_idx = splits
    train_ds = SaudiDataset(fields, tr_idx, edge_index, pos)
    val_ds = SaudiDataset(fields, va_idx, edge_index, pos)
    test_ds = SaudiDataset(fields, te_idx, edge_index, pos)
    sub = np.stack([train_ds.assemble(tr_idx[k]) for k in range(0, len(tr_idx), 20)]).reshape(-1, train_ds.in_channels)

    model = PhysModel(train_ds.in_channels)
    model.set_stats(sub.mean(0), sub.std(0))
    use_phys = mode != "none"
    log_vars = nn.Parameter(torch.zeros(2)) if use_phys else None
    params = list(model.parameters()) + ([log_vars] if use_phys else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    mse = nn.MSELoss()
    tl = DataLoader(train_ds, batch_size=32, shuffle=True)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False)

    best, state = float("inf"), None
    for ep in range(EPOCHS):
        model.train()
        for d in tl:
            opt.zero_grad()
            pred = model(d)
            l_data = mse(pred, d.y)
            if use_phys:
                l_phys = physics_residual(d.x, pred, model, mode, rows, cols, H, W, mean_speed)
                loss = (0.5 * torch.exp(-log_vars[0]) * l_data + 0.5 * log_vars[0]
                        + 0.5 * torch.exp(-log_vars[1]) * l_phys + 0.5 * log_vars[1])
            else:
                loss = l_data
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        model.eval(); vtot = nb = 0
        with torch.no_grad():
            for d in vl:
                vtot += mse(model(d), d.y).item(); nb += 1
        if vtot / nb < best:
            best, state = vtot / nb, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval()

    preds, trues, cur = [], [], []
    with torch.no_grad():
        for d in DataLoader(test_ds, batch_size=64, shuffle=False):
            preds.append(model(d).numpy()); trues.append(d.y.numpy())
            cur.append(torch.cat([d.x[:, 0:1], d.x[:, 3:4]], 1).numpy())
    P, Y, C = map(np.concatenate, (preds, trues, cur))
    m_mse, m_r2 = metrics(P, Y); p_mse, _ = metrics(C, Y)
    extra = ""
    if mode == "complete":
        extra = (f" | E={F.softplus(model.emit).detach().numpy().round(3)} "
                 f"Ddry={F.softplus(model.dry).detach().numpy().round(3)} "
                 f"Dwet={F.softplus(model.wet).detach().numpy().round(3)}")
    return m_r2, (p_mse - m_mse) / p_mse * 100, extra


def main():
    fields, years, edge_index, pos, N = load_fields()
    T = fields["pm10"].shape[0]
    mean_speed = float(fields["speed"].mean())

    # node -> (row, col) on the 25x31 grid, for gradient reconstruction
    ds = xr.open_dataset(MERGED); mask = ds.valid_mask.values; H, W = mask.shape
    _, _, node_flat = build_masked_graph(mask, neighbors=8)
    rows = torch.as_tensor(node_flat // W, dtype=torch.long)
    cols = torch.as_tensor(node_flat % W, dtype=torch.long)

    def block(y0, y1):
        lo = np.argmax(years >= y0); hi = T - 1 - np.argmax(years[::-1] <= y1)
        return [i for i in range(lo + WINDOW - 1, hi) if i + 1 <= hi]
    splits = (block(2003, 2018), block(2019, 2020), block(2021, 2024))
    print(f"  samples train/val/test: {[len(s) for s in splits]} | mean_speed={mean_speed:.2f}", flush=True)

    rows_list = []
    for mode in ["none", "advection", "complete"]:
        t0 = time.time()
        r2, skill, extra = train_eval(mode, fields, splits, edge_index, pos, N, rows, cols, H, W, mean_speed)
        print(f"  [{mode:<9}] PM10 R2={r2[0]:.4f} skill={skill[0]:+.2f}% | "
              f"PM2.5 R2={r2[1]:.4f} skill={skill[1]:+.2f}% | {time.time()-t0:.0f}s{extra}", flush=True)
        rows_list.append((mode, r2, skill))

    print("\n==== PHYSICS MODE COMPARISON (Saudi test 2021-24) ====")
    for mode, r2, skill in rows_list:
        print(f"  {mode:<10} PM10 R2={r2[0]:.4f} (+{skill[0]:.1f}%)   PM2.5 R2={r2[1]:.4f} (+{skill[1]:.1f}%)")


if __name__ == "__main__":
    main()
