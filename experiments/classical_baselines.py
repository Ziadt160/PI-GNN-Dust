"""Compare the PI-GNN against classical baselines on the held-out test set.

Answers "is the PI-GNN prediction actually right / worth the complexity?" by
evaluating every method on the SAME 3-way split, SAME target (log-space PM10 at
t+1), and SAME node features (where applicable), pooled over all grid cells:

  * Climatology            - training monthly mean per cell (seasonal no-skill ref)
  * Persistence            - C_{t+1} = C_t
  * Persistence-of-tendency- C_{t+1} = C_t + (C_t - C_{t-1})
  * Ridge linear           - linear map on the same features (no graph, no physics)
  * Random Forest          - nonlinear tabular model, same features, no graph
  * Gradient Boosting      - nonlinear tabular model, same features, no graph
  * PI-GNN                 - the trained graph + physics model (results/best_pi_gnn.pth)

Learning baselines are fit on the TRAIN split (2003-05), the same parameter-fitting
data the PI-GNN used. If the GNN's graph/physics structure helps, it should beat the
structure-free tabular models that see identical features.

Run from the repository root:  python experiments/classical_baselines.py
"""
import os
import sys
import time

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


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def metrics(pred, true):
    mse = float(np.mean((pred - true) ** 2))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return mse, float(np.sqrt(mse)), float(np.mean(np.abs(pred - true))), 1.0 - ss_res / ss_tot


def make_ds(cfg, years):
    data_dir = resolve(cfg["data"]["dir"])
    feat = cfg.get("features", {})
    era = [os.path.join(data_dir, f"era5_{y}.nc") for y in years]
    return GraphDustDataset(
        os.path.join(data_dir, cfg["data"]["sfc_file"]),
        era, cfg["physics"]["lat_dim"], cfg["physics"]["lon_dim"],
        window=feat.get("window", 1), add_speed=feat.get("add_speed", False),
        add_doy=feat.get("add_doy", False), neighbors=feat.get("neighbors", 4),
        resample=feat.get("resample", "12h"),
    )


def assemble_xy(ds):
    """Flatten to node-level samples: X [n*cells, F], y [n*cells], plus current/prev pm10."""
    X, y, cur, prev = [], [], [], []
    win = ds.window
    for i in range(ds.len()):
        idx = i + (win - 1)
        feats = ds._assemble(idx)              # [cells, F]
        X.append(feats)
        y.append(ds.pm10_log[idx + 1])         # target [cells]
        cur.append(ds.pm10_log[idx])           # current [cells]
        prev.append(ds.pm10_log[idx - 1] if win > 1 else ds.pm10_log[idx])
    return (np.concatenate(X), np.concatenate(y),
            np.concatenate(cur), np.concatenate(prev))


def climatology(train_ds, test_ds):
    """Per-(month, cell) mean of log PM10 from TRAIN; looked up for each test target."""
    clim = {}
    for m in range(1, 13):
        mask = train_ds.month == m
        if mask.any():
            clim[m] = train_ds.pm10_log[mask].mean(axis=0)   # [cells]
    fallback = train_ds.pm10_log.mean(axis=0)
    preds = []
    win = test_ds.window
    for i in range(test_ds.len()):
        tgt_idx = i + (win - 1) + 1
        m = int(test_ds.month[tgt_idx])
        preds.append(clim.get(m, fallback))
    return np.concatenate(preds)


def main():
    with open(resolve("configs/default_config.yaml")) as f:
        cfg = yaml.safe_load(f)

    print("Building datasets (train 2003-05, test 2007) ...")
    train_ds = make_ds(cfg, cfg["training"]["years"])
    test_ds = make_ds(cfg, cfg["test"]["years"])

    Xtr, ytr, _, _ = assemble_xy(train_ds)
    Xte, yte, cur_te, prev_te = assemble_xy(test_ds)
    print(f"  train rows {Xtr.shape}, test rows {Xte.shape}")

    results = {}

    # --- Trivial references ---
    results["Climatology (monthly)"] = metrics(climatology(train_ds, test_ds), yte)
    results["Persistence"] = metrics(cur_te, yte)
    results["Persistence-of-tendency"] = metrics(2 * cur_te - prev_te, yte)

    # --- Classical ML on the same features (no graph / no physics) ---
    print("Fitting Ridge ...")
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    results["Ridge linear"] = metrics(ridge.predict(Xte), yte)

    print("Fitting Random Forest ...")
    rf = RandomForestRegressor(n_estimators=150, max_depth=12, n_jobs=-1, random_state=0).fit(Xtr, ytr)
    results["Random Forest"] = metrics(rf.predict(Xte), yte)

    print("Fitting Gradient Boosting ...")
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
    results["Gradient Boosting"] = metrics(hgb.predict(Xte), yte)

    # --- PI-GNN (trained checkpoint) ---
    model = PIDustModel(in_channels=test_ds.in_channels,
                        hidden_channels=cfg["model"]["hidden_channels"],
                        hops=cfg["model"]["hops"])
    ckpt = resolve(cfg["paths"]["model_out"])
    gnn_preds = None
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()
        preds = []
        with torch.no_grad():
            for d in DataLoader(test_ds, batch_size=64, shuffle=False):
                p, _ = model(d)
                preds.append(p.numpy().ravel())
        gnn_preds = np.concatenate(preds)
        results["PI-GNN (graph+physics)"] = metrics(gnn_preds, yte)
    else:
        print(f"  (PI-GNN checkpoint not found at {ckpt}; run train_model.py)")

    # --- Report ---
    persist_mse = results["Persistence"][0]
    order = sorted(results.items(), key=lambda kv: kv[1][0])  # by MSE ascending
    print("\n" + "=" * 76)
    print(f"{'Method':<28}{'MSE':>9}{'RMSE':>9}{'MAE':>9}{'R2':>9}{'skill%':>10}")
    print("-" * 76)
    for name, (mse, rmse, mae, r2) in order:
        skill = (persist_mse - mse) / persist_mse * 100
        star = "  <-- PI-GNN" if name.startswith("PI-GNN") else ""
        print(f"{name:<28}{mse:>9.4f}{rmse:>9.4f}{mae:>9.4f}{r2:>9.4f}{skill:>+9.2f}%{star}")
    print("=" * 76)
    print("skill% = MSE reduction vs persistence (test 2007, log-space, pooled over cells)")


if __name__ == "__main__":
    main()
