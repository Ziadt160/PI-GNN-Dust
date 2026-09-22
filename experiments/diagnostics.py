"""Diagnose the PI-GNN: is the physics prior helping, and does the model win at the
resolution where the advection physics is numerically valid (6 h vs 12 h)?

For each resolution it trains the enhanced GNN with physics ON and OFF (multi-seed)
and fits the classical baselines on the same features, then ranks everything on the
held-out 2007 test set.

Run from the repository root:  python experiments/diagnostics.py
"""
import os
import sys

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.pi_gnn import PIDustModel
from src.training.dataset import GraphDustDataset
from src.training.trainer import Trainer

DATA = os.path.join(REPO_ROOT, "data")
SFC = os.path.join(DATA, "data_sfc.nc")
FEAT = dict(window=3, add_speed=True, add_doy=True, neighbors=8)
HIDDEN, HOPS, EPOCHS = 64, 2, 40


def metrics(pred, true):
    mse = float(np.mean((pred - true) ** 2))
    r2 = 1.0 - float(np.sum((true - pred) ** 2)) / float(np.sum((true - np.mean(true)) ** 2))
    return mse, r2


def make_ds(years, resample):
    era = [os.path.join(DATA, f"era5_{y}.nc") for y in years]
    return GraphDustDataset(SFC, era, 7, 12, resample=resample, **FEAT)


def assemble_xy(ds):
    X, y, cur = [], [], []
    for i in range(ds.len()):
        idx = i + (ds.window - 1)
        X.append(ds._assemble(idx)); y.append(ds.pm10_log[idx + 1]); cur.append(ds.pm10_log[idx])
    return np.concatenate(X), np.concatenate(y), np.concatenate(cur)


def train_gnn(train_ds, val_ds, test_ds, dt, use_physics, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = PIDustModel(in_channels=train_ds.in_channels, hidden_channels=HIDDEN, hops=HOPS)
    model.set_norm_stats(*train_ds.compute_feature_stats())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    physics_cfg = {"delta_t": dt, "dx": 74880.0, "dy": 83490.0, "grid_size": (7, 12)}
    tr = Trainer(model, opt, torch.device("cpu"), physics_cfg=physics_cfg, use_physics=use_physics)
    tl = DataLoader(train_ds, batch_size=32, shuffle=True)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False)
    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        tr.train_epoch(tl)
        _, vmse = tr.validate(vl)
        if vmse < best:
            best, best_state = vmse, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    preds = []
    with torch.no_grad():
        for d in DataLoader(test_ds, batch_size=64, shuffle=False):
            p, _ = model(d); preds.append(p.numpy().ravel())
    return np.concatenate(preds)


def run_resolution(resample, dt, seeds):
    print(f"\n########## RESOLUTION = {resample} ##########")
    train_ds = make_ds((2003, 2004, 2005), resample)
    val_ds = make_ds((2006,), resample)
    test_ds = make_ds((2007,), resample)
    Xtr, ytr, _ = assemble_xy(train_ds)
    Xte, yte, cur_te = assemble_xy(test_ds)

    rows = {}
    rows["Persistence"] = metrics(cur_te, yte)
    rows["Ridge linear"] = metrics(Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte), yte)
    rows["Random Forest"] = metrics(
        RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=0).fit(Xtr, ytr).predict(Xte), yte)
    rows["Gradient Boosting"] = metrics(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr).predict(Xte), yte)

    for phys in (True, False):
        mses, r2s = [], []
        for s in seeds:
            pred = train_gnn(train_ds, val_ds, test_ds, dt, phys, s)
            m, r = metrics(pred, yte); mses.append(m); r2s.append(r)
        label = f"PI-GNN physics-{'ON' if phys else 'OFF'}"
        rows[label] = (float(np.mean(mses)), float(np.mean(r2s)), float(np.std(r2s)))

    persist = rows["Persistence"][0]
    print(f"\n{resample} results (test 2007, {len(seeds)} seeds for GNN):")
    print(f"  {'Method':<26}{'MSE':>9}{'R2':>9}{'skill%':>10}")
    for name, vals in sorted(rows.items(), key=lambda kv: kv[1][0]):
        mse, r2 = vals[0], vals[1]
        sd = f" +/-{vals[2]:.3f}" if len(vals) > 2 else ""
        print(f"  {name:<26}{mse:>9.4f}{r2:>9.4f}{(persist - mse) / persist * 100:>+9.2f}%{sd}")
    return rows


def main():
    run_resolution("12h", 43200.0, (0, 1, 2))
    run_resolution("6h", 21600.0, (0, 1))
    print("\nKey questions:")
    print("  Q1: physics-ON vs physics-OFF at 12h -> is the advection prior helping?")
    print("  Q2: PI-GNN vs Gradient Boosting at 6h -> does the GNN win where physics is valid?")


if __name__ == "__main__":
    main()
