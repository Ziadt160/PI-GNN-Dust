"""Are the models' predictions PHYSICALLY consistent (not just accurate)?

A physics-informed model is meant to trade some accuracy for physical realism.
This script evaluates every model on the held-out test set with both accuracy
metrics AND physics diagnostics, so we can see whether the PI-GNN's physics loss
buys physical consistency that the (more accurate) classical / data-driven models
lack.

Diagnostics (test 2007, log-space, same features):
  * R2 / MSE             - skill
  * adv-residual (RMS)   - |(C_{t+1}-C_t) + (u.grad C_t) dt|  : advection consistency
  * mass-err %           - domain-mean drift vs truth
  * neg %                - fraction predicting negative concentration (unphysical)
  * roughness ratio      - spatial gradient energy of prediction / of ground truth
                           (>>1 = noisy/incoherent fields; ~1 = realistic structure)

Run from the repository root:  python experiments/gulf_pm10/physical_consistency.py
"""
import os
import sys

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.pi_gnn import PIDustModel
from src.training.dataset import GraphDustDataset
from src.training.trainer import Trainer

DATA = os.path.join(REPO_ROOT, "data")
SFC = os.path.join(DATA, "data_sfc.nc")
FEAT = dict(window=3, add_speed=True, add_doy=True, neighbors=8)
LAT, LON = 7, 12
DT, DX, DY = 43200.0, 74880.0, 83490.0   # 12 h


def make_ds(years):
    era = [os.path.join(DATA, f"era5_{y}.nc") for y in years]
    return GraphDustDataset(SFC, era, LAT, LON, resample="12h", **FEAT)


def assemble(ds):
    X, y, cur = [], [], []
    for i in range(ds.len()):
        idx = i + ds.window - 1
        X.append(ds._assemble(idx)); y.append(ds.pm10_log[idx + 1]); cur.append(ds.pm10_log[idx])
    return np.concatenate(X), np.concatenate(y), np.concatenate(cur)


def grid(a):
    return a.reshape(-1, LAT, LON)


def acc(pred, true):
    mse = np.mean((pred - true) ** 2)
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
    return mse, r2


def adv_residual(pred, cur, u, v):
    """RMS of the advection-equation residual of the prediction."""
    P, C = grid(pred), grid(cur)
    U, V = grid(u), grid(v)
    g = np.gradient(C, DY, DX, axis=(1, 2))
    gn, ge = -g[0], g[1]
    advection = U * ge + V * gn
    res = (P - C) + advection * DT
    return float(np.sqrt(np.mean(res ** 2)))


def roughness(field):
    G = grid(field)
    gx = np.gradient(G, axis=2)
    gy = np.gradient(G, axis=1)
    return float(np.mean(gx ** 2 + gy ** 2))


def train_gnn(train_ds, val_ds, test_ds, use_physics, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = PIDustModel(in_channels=train_ds.in_channels, hidden_channels=64, hops=2)
    m.set_norm_stats(*train_ds.compute_feature_stats())
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    tr = Trainer(m, opt, torch.device("cpu"),
                 physics_cfg={"delta_t": DT, "dx": DX, "dy": DY, "grid_size": (LAT, LON)},
                 use_physics=use_physics)
    tl = DataLoader(train_ds, batch_size=32, shuffle=True)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False)
    best, state = float("inf"), None
    for _ in range(40):
        tr.train_epoch(tl)
        _, vmse = tr.validate(vl)
        if vmse < best:
            best, state = vmse, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    preds = []
    with torch.no_grad():
        for d in DataLoader(test_ds, batch_size=64, shuffle=False):
            p, _ = m(d); preds.append(p.numpy().ravel())
    return np.concatenate(preds)


def main():
    print("Building datasets + features ...")
    train_ds, val_ds, test_ds = make_ds((2003, 2004, 2005)), make_ds((2006,)), make_ds((2007,))
    Xtr, ytr, _ = assemble(train_ds)
    Xte, yte, cur = assemble(test_ds)
    u_te, v_te = Xte[:, 1], Xte[:, 2]

    preds = {}
    preds["Ground truth"] = yte
    preds["Persistence"] = cur
    print("Fitting classical models ...")
    preds["Ridge"] = Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)
    preds["Random Forest"] = RandomForestRegressor(
        n_estimators=150, max_depth=12, n_jobs=-1, random_state=0).fit(Xtr, ytr).predict(Xte)
    preds["Gradient Boosting"] = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr).predict(Xte)
    print("Training PI-GNN physics-OFF ...")
    preds["PI-GNN physics-OFF"] = train_gnn(train_ds, val_ds, test_ds, use_physics=False)
    print("Training PI-GNN physics-ON ...")
    preds["PI-GNN physics-ON"] = train_gnn(train_ds, val_ds, test_ds, use_physics=True)

    gt_rough = roughness(yte)
    true_mean_phys = np.expm1(yte).mean()

    print("\n" + "=" * 92)
    print(f"{'Model':<22}{'R2':>8}{'MSE':>9}{'adv-resid':>11}{'mass-err%':>11}{'neg%':>8}{'rough/GT':>10}")
    print("-" * 92)
    for name, p in preds.items():
        mse, r2 = acc(p, yte)
        res = adv_residual(p, cur, u_te, v_te)
        mass = abs(np.expm1(np.clip(p, 0, None)).mean() - true_mean_phys) / true_mean_phys * 100
        neg = np.mean(p < 0) * 100
        rr = roughness(p) / gt_rough
        tag = "  (reference)" if name == "Ground truth" else ""
        print(f"{name:<22}{r2:>8.3f}{mse:>9.4f}{res:>11.4f}{mass:>11.2f}{neg:>8.2f}{rr:>10.2f}{tag}")
    print("=" * 92)
    print("adv-resid: lower = more consistent with the advection equation (GT = irreducible).")
    print("rough/GT : ~1 = realistic spatial structure; >>1 = noisy/incoherent fields.")


if __name__ == "__main__":
    main()
