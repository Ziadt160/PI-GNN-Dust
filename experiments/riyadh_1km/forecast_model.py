"""Physics-informed 1 km PM forecaster for a regional box (e.g. Riyadh).

Reuses the validated PhysicsFluxLayer GNN, but for MULTI-STEP FORECASTING:
given an initial 1 km PM field + future wind forcing, it rolls the field forward
K steps. A light advection-consistency residual keeps the rollout sharp and
transport-consistent (our experiments showed pure data-driven rollouts blur/drift,
which is exactly what physics fixes for forecasting).

This module is data-agnostic: it trains on any sequence of (PM, u, v) fields on a
regular grid. `synthetic_demo()` validates the whole pipeline on a synthetic
advection sequence (where transport IS the true dynamics, so physics-on should beat
physics-off on rollout) before real MAIAC/monitor data is wired in via
build_riyadh_dataset.py.

Run:  python -u experiments/riyadh_1km/forecast_model.py
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import map_coordinates

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from src.layers.flux import PhysicsFluxLayer
from src.utils.graph import build_dust_graph

DT = 1.0  # one forecast step (synthetic units; real = e.g. 3600s, set in config)


class ForecastModel(nn.Module):
    """Residual flux-GNN: features [C, u, v] -> delta C; next = C + delta."""
    def __init__(self, hidden=64, hops=2):
        super().__init__()
        self.flux_in = PhysicsFluxLayer(3, hidden)
        self.flux_hidden = nn.ModuleList(PhysicsFluxLayer(hidden, hidden) for _ in range(hops - 1))
        self.head = nn.Linear(hidden, 1)
        self.register_buffer("mean", torch.zeros(3))
        self.register_buffer("std", torch.ones(3))
        nn.init.constant_(self.head.weight, 0.0); nn.init.constant_(self.head.bias, 0.0)

    def set_stats(self, m, s):
        self.mean.copy_(torch.as_tensor(m)); self.std.copy_(torch.clamp(torch.as_tensor(s), min=1e-6))

    def step(self, C, u, v, edge_index, pos):
        """One forecast step: predict C_{t+1} from (C, u, v). Reusable for rollout."""
        x = torch.cat([C, u, v], dim=1)
        xn = (x - self.mean) / self.std
        h = torch.relu(self.flux_in(xn, edge_index, pos, u, v, C))
        for fl in self.flux_hidden:
            h = h + torch.relu(fl(h, edge_index, pos, u, v, C))
        return C + self.head(h)

    def forward(self, data):
        return self.step(data.x[:, 0:1], data.x[:, 1:2], data.x[:, 2:3], data.edge_index, data.pos)


def advection_residual(C_cur, C_next, u, v, B, H, W, dx, dy):
    """Explicit-Euler residual of dC/dt + u.grad C = 0. ONLY valid at CFL < 1."""
    g = lambda a: a.view(B, H, W)
    gy, gx = torch.gradient(g(C_cur), spacing=(dy, dx), dim=(1, 2))
    grad_n = (-gy).reshape(-1, 1); grad_e = gx.reshape(-1, 1)
    adv = u * grad_e + v * grad_n
    R = (C_next - C_cur) + adv * DT
    return torch.mean(R ** 2)


def semilag_residual(C_cur, C_next, disp_row, disp_col, B, H, W):
    """CFL-STABLE semi-Lagrangian transport residual (valid at any Courant number).

    Enforces C_next(x) ~= C_cur(x - velocity*dt): back-trace every cell to its
    departure point and bilinearly sample C_cur there, then penalise the mismatch.
    `disp_row`/`disp_col` are per-cell displacements in GRID CELLS over one step
    (= velocity * dt), [B*N, 1]. Uses grid_sample (differentiable, any displacement).
    """
    N = H * W
    grid_C = C_cur.view(B, 1, H, W)
    rr, cc = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    dep_row = rr.unsqueeze(0) - disp_row.view(B, H, W)   # departure points (back-trace)
    dep_col = cc.unsqueeze(0) - disp_col.view(B, H, W)
    gx = 2 * dep_col / (W - 1) - 1                        # normalise to [-1, 1]
    gy = 2 * dep_row / (H - 1) - 1
    samp_grid = torch.stack([gx, gy], dim=-1)            # [B, H, W, 2] (x=col, y=row)
    advected = F.grid_sample(grid_C, samp_grid, mode="bilinear",
                             padding_mode="border", align_corners=True).reshape(B * N, 1)
    return torch.mean((C_next - advected) ** 2)


# --------------------------------------------------------------------------- #
# Synthetic advection sequences (validation only; replace with real data).
# --------------------------------------------------------------------------- #
def make_sequence(H, W, T, rng, wind_scale=1.2):
    # initial field: a few Gaussian "plumes"
    yy, xx = np.mgrid[0:H, 0:W]
    C = np.zeros((H, W), np.float32)
    for _ in range(rng.integers(2, 5)):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        s = rng.uniform(3, 7)
        C += rng.uniform(0.5, 1.5) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * s ** 2))
    # smooth, near-constant wind for this sequence (cells/step = Courant number)
    u = rng.uniform(-wind_scale, wind_scale); v = rng.uniform(-wind_scale, wind_scale)
    seq = [C]
    for _ in range(T - 1):
        # semi-Lagrangian backtrace (unconditionally stable)
        coords = np.array([yy - v, xx - u])
        Cn = map_coordinates(seq[-1], coords, order=1, mode="nearest")
        Cn = Cn * 0.98 + 0.01 * rng.standard_normal((H, W)).astype(np.float32)  # mild decay + noise
        seq.append(Cn.astype(np.float32))
    U = np.full((H, W), u, np.float32); V = np.full((H, W), v, np.float32)
    return np.stack(seq), U, V  # [T,H,W], [H,W], [H,W]


def build_pairs(seqs, edge_index, pos, H, W):
    data = []
    for C, U, V in seqs:
        for t in range(len(C) - 1):
            x = torch.tensor(np.stack([C[t].ravel(), U.ravel(), V.ravel()], 1), dtype=torch.float32)
            y = torch.tensor(C[t + 1].ravel(), dtype=torch.float32).view(-1, 1)
            data.append(Data(x=x, edge_index=edge_index, pos=pos, y=y))
    return data


def build_rollout_windows(seqs, edge_index, pos, H, W, K):
    """Each sample = (C_t, winds) + the next K true fields, for seq2seq rollout training."""
    data = []
    for C, U, V in seqs:
        u = torch.tensor(U.ravel(), dtype=torch.float32).view(-1, 1)
        v = torch.tensor(V.ravel(), dtype=torch.float32).view(-1, 1)
        for t in range(len(C) - K):
            c0 = torch.tensor(C[t].ravel(), dtype=torch.float32).view(-1, 1)
            yseq = torch.stack([torch.tensor(C[t + k + 1].ravel(), dtype=torch.float32)
                                for k in range(K)], dim=1)  # [N, K]
            d = Data(x=torch.cat([c0, u, v], 1), edge_index=edge_index, pos=pos)
            d.yseq = yseq
            data.append(d)
    return data


def _roughness(field_HW):
    gy, gx = np.gradient(field_HW)
    return float(np.mean(gx ** 2 + gy ** 2))


@torch.no_grad()
def rollout_metrics(model, seqs, edge_index, pos, H, W, K):
    """Returns (rollout RMSE, fidelity = roughness(pred)/roughness(truth) at step K)."""
    model.eval(); errs, rr = [], []
    for C, U, V in seqs:
        cur = torch.tensor(C[0].ravel(), dtype=torch.float32).view(-1, 1)
        u = torch.tensor(U.ravel(), dtype=torch.float32).view(-1, 1)
        v = torch.tensor(V.ravel(), dtype=torch.float32).view(-1, 1)
        for k in range(1, K + 1):
            d = Data(x=torch.cat([cur, u, v], 1), edge_index=edge_index, pos=pos)
            cur = model(d)
        pred = cur.view(-1).numpy().reshape(H, W)
        errs.append(float(np.sqrt(np.mean((pred - C[K]) ** 2))))
        rr.append(_roughness(pred) / max(_roughness(C[K]), 1e-9))
    return float(np.mean(errs)), float(np.mean(rr))


def train_eval(mode, H, W, edge_index, pos, train_seqs, test_seqs, epochs=15, K=6, k_train=1):
    """k_train=1 -> single-step (teacher-forced); k_train>1 -> seq2seq rollout training
    (predictions fed back, loss + physics applied at every one of k_train steps)."""
    torch.manual_seed(0); np.random.seed(0)
    N = H * W
    multistep = k_train > 1
    train_data = (build_rollout_windows(train_seqs, edge_index, pos, H, W, k_train) if multistep
                  else build_pairs(train_seqs, edge_index, pos, H, W))
    loader = DataLoader(train_data, batch_size=8, shuffle=True)
    allx = torch.cat([d.x for d in train_data[:200]])
    model = ForecastModel(); model.set_stats(allx.mean(0), allx.std(0))
    use_phys = mode != "none"
    log_vars = nn.Parameter(torch.zeros(2)) if use_phys else None
    params = list(model.parameters()) + ([log_vars] if use_phys else [])
    opt = torch.optim.Adam(params, lr=1e-3); mse = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        for d in loader:
            opt.zero_grad()
            B = d.x.shape[0] // N
            u, v = d.x[:, 1:2], d.x[:, 2:3]
            cur = d.x[:, 0:1]
            l_data = torch.zeros(()); l_phys = torch.zeros(())
            steps = k_train if multistep else 1
            for k in range(steps):
                pred = model.step(cur, u, v, d.edge_index, d.pos)
                target = d.yseq[:, k:k + 1] if multistep else d.y
                l_data = l_data + mse(pred, target)
                if use_phys:
                    if mode == "semilag":
                        l_phys = l_phys + semilag_residual(cur, pred, v * DT, u * DT, B, H, W)
                    else:
                        l_phys = l_phys + advection_residual(cur, pred, u, v, B, H, W, 1.0, 1.0)
                cur = pred  # free-running feedback (exposure to own errors)
            if use_phys:
                loss = (0.5 * torch.exp(-log_vars[0]) * l_data + 0.5 * log_vars[0]
                        + 0.5 * torch.exp(-log_vars[1]) * l_phys + 0.5 * log_vars[1])
            else:
                loss = l_data
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    return rollout_metrics(model, test_seqs, edge_index, pos, H, W, K)


def synthetic_demo():
    H = W = 40  # small grid for fast validation; real Riyadh ~89x81
    edge_index, pos = build_dust_graph(H, W, neighbors=8)
    cfl = 1.2  # high-Courant regime (the one 1 km/hourly forecasting lives in)
    rng = np.random.default_rng(0)
    train = [make_sequence(H, W, 12, rng, cfl) for _ in range(40)]
    test = [make_sequence(H, W, 12, rng, cfl) for _ in range(10)]
    print(f"Synthetic validation: grid {H}x{W} ({H*W} nodes, {edge_index.shape[1]} edges), CFL~{cfl}\n"
          f"Comparing single-step vs multi-step (seq2seq) training; 6-step rollout:\n", flush=True)

    configs = [
        ("single-step  none   ", "none",    1),
        ("single-step  semilag", "semilag", 1),
        ("multi-step   none   ", "none",    4),
        ("multi-step   semilag", "semilag", 4),
    ]
    for label, mode, k_train in configs:
        rmse, fid = train_eval(mode, H, W, edge_index, pos, train, test, k_train=k_train)
        print(f"  {label}  rollout RMSE={rmse:.4f}  fidelity(rough/GT, ~1 ideal)={fid:.2f}", flush=True)

    print("\nExpected: multi-step (seq2seq) training lowers rollout error vs single-step by")
    print("exposing the model to its own predictions; semi-Lagrangian physics keeps it stable")
    print("at CFL>>1 (the 1 km regime, where the explicit advection residual diverges).")


if __name__ == "__main__":
    synthetic_demo()
