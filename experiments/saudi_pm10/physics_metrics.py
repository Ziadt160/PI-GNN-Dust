"""Accuracy is NOT the right yardstick for a physics-informed model.

This evaluates every model on PHYSICS metrics, not just R2, on the Saudi test set:
  * R2                - accuracy (for reference)
  * adv-resid (RMS)   - |(C_{t+1}-C_t) + (u.grad C_t) dt| : transport consistency
                        (model-independent; ground truth gives the irreducible value)
  * mass-err %        - domain-mean drift vs truth (physical-space)
  * rough/GT          - spatial gradient energy of prediction / ground truth
                        (~1 = realistic structure; <1 = blurry/over-smoothed)

Compares the 3 physics modes (none / advection / complete) and the structure-free
classical baselines (Ridge / RF / GBM). Hypothesis (the whole point): the data-driven
/ classical models win on R2 but are physically WORSE -- higher transport residual,
worse mass conservation, blurrier fields -- while the physics-trained models are more
physically faithful despite lower accuracy.

Run from the repo root:  python -u experiments/saudi_pm10/physics_metrics.py
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import warnings
warnings.filterwarnings("ignore")
from torch_geometric.loader import DataLoader
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)
from src.utils.graph import build_masked_graph
from experiments.saudi_pm10.train_saudi import load_fields, SaudiDataset, metrics, WINDOW, MERGED
from experiments.saudi_pm10.physics_compare import PhysModel, physics_residual, DT, DY, DX

EPOCHS = int(os.environ.get("PHYS_EPOCHS", "18"))


# ---------- physics metrics (numpy, on test predictions) ----------
def grad_at_nodes(vals_SN, rows, cols, H, W):
    S = vals_SN.shape[0]
    grid = np.full((S, H, W), np.nan, np.float32)
    grid[:, rows, cols] = vals_SN
    gy = np.gradient(grid, DY, axis=1)
    gx = np.gradient(grid, DX, axis=2)
    return -gy[:, rows, cols], gx[:, rows, cols]   # north, east  [S,N]


def adv_resid(next_SN, cur_SN, u_SN, v_SN, rows, cols, H, W):
    gn, ge = grad_at_nodes(cur_SN, rows, cols, H, W)
    R = (next_SN - cur_SN) + (u_SN * ge + v_SN * gn) * DT
    fin = np.isfinite(R)
    return float(np.sqrt(np.mean(R[fin] ** 2)))


def roughness(field_SN, rows, cols, H, W):
    S = field_SN.shape[0]
    grid = np.full((S, H, W), np.nan, np.float32)
    grid[:, rows, cols] = field_SN
    gy = np.gradient(grid, axis=1)[:, rows, cols]
    gx = np.gradient(grid, axis=2)[:, rows, cols]
    e = gx ** 2 + gy ** 2
    return float(np.mean(e[np.isfinite(e)]))


def mass_err(pred_SN, true_SN):
    p = np.expm1(np.clip(pred_SN, 0, None)).mean()
    t = np.expm1(true_SN).mean()
    return abs(p - t) / t * 100


# ---------- training ----------
def train_gnn(mode, train_ds, val_ds, mean, std, rows_t, cols_t, H, W, mean_speed):
    torch.manual_seed(0); np.random.seed(0)
    model = PhysModel(train_ds.in_channels); model.set_stats(mean, std)
    use_phys = mode != "none"
    log_vars = nn.Parameter(torch.zeros(2)) if use_phys else None
    params = list(model.parameters()) + ([log_vars] if use_phys else [])
    opt = torch.optim.Adam(params, lr=1e-3); mse = nn.MSELoss()
    tl = DataLoader(train_ds, batch_size=32, shuffle=True)
    vl = DataLoader(val_ds, batch_size=64, shuffle=False)
    best, state = float("inf"), None
    for ep in range(EPOCHS):
        model.train()
        for d in tl:
            opt.zero_grad(); pred = model(d); l = mse(pred, d.y)
            if use_phys:
                lp = physics_residual(d.x, pred, model, mode, rows_t, cols_t, H, W, mean_speed)
                l = (0.5 * torch.exp(-log_vars[0]) * l + 0.5 * log_vars[0]
                     + 0.5 * torch.exp(-log_vars[1]) * lp + 0.5 * log_vars[1])
            l.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        model.eval(); vt = nb = 0
        with torch.no_grad():
            for d in vl:
                vt += mse(model(d), d.y).item(); nb += 1
        if vt / nb < best:
            best, state = vt / nb, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state); model.eval()
    return model


def gnn_predict(model, test_ds, N):
    out = []
    with torch.no_grad():
        for d in DataLoader(test_ds, batch_size=64, shuffle=False):
            out.append(model(d).numpy())
    return np.concatenate(out).reshape(-1, N, 2)   # [S, N, 2]


def main():
    fields, years, edge_index, pos, N = load_fields()
    T = fields["pm10"].shape[0]; mean_speed = float(fields["speed"].mean())
    ds = xr.open_dataset(MERGED); mask = ds.valid_mask.values; H, W = mask.shape
    _, _, node_flat = build_masked_graph(mask, neighbors=8)
    rows, cols = node_flat // W, node_flat % W
    rows_t, cols_t = torch.as_tensor(rows), torch.as_tensor(cols)

    def block(y0, y1):
        lo = np.argmax(years >= y0); hi = T - 1 - np.argmax(years[::-1] <= y1)
        return [i for i in range(lo + WINDOW - 1, hi) if i + 1 <= hi]
    tr_idx, va_idx, te_idx = block(2003, 2018), block(2019, 2020), block(2021, 2024)
    train_ds = SaudiDataset(fields, tr_idx, edge_index, pos)
    val_ds = SaudiDataset(fields, va_idx, edge_index, pos)
    test_ds = SaudiDataset(fields, te_idx, edge_index, pos)
    mean = std = None

    # Test tensors [S, N, *]
    S = len(te_idx)
    Xte = np.stack([test_ds.assemble(i) for i in te_idx])              # [S,N,F]
    Yte = np.stack([np.stack([fields["pm10"][i + 1], fields["pm25"][i + 1]], 1) for i in te_idx])  # [S,N,2]
    cur = Xte[:, :, [0, 3]]                                            # current pm10,pm2.5 [S,N,2]
    u_te, v_te = Xte[:, :, 1], Xte[:, :, 2]
    sub = np.stack([train_ds.assemble(tr_idx[k]) for k in range(0, len(tr_idx), 20)]).reshape(-1, train_ds.in_channels)
    mean, std = sub.mean(0), sub.std(0)

    preds = {}
    preds["GroundTruth"] = Yte
    preds["Persistence"] = cur

    # GNN physics modes
    for mode in ["none", "advection", "complete"]:
        t0 = time.time()
        m = train_gnn(mode, train_ds, val_ds, mean, std, rows_t, cols_t, H, W, mean_speed)
        preds[f"GNN-{mode}"] = gnn_predict(m, test_ds, N)
        print(f"  trained GNN-{mode} ({time.time()-t0:.0f}s)", flush=True)

    # Classical baselines on flat node features (subsample train rows: 2.9M is overkill).
    Xtr = np.stack([train_ds.assemble(i) for i in tr_idx]).reshape(-1, train_ds.in_channels)
    Ytr = np.stack([np.stack([fields["pm10"][i + 1], fields["pm25"][i + 1]], 1) for i in tr_idx]).reshape(-1, 2)
    rng = np.random.default_rng(0)
    sel = rng.choice(Xtr.shape[0], size=min(150_000, Xtr.shape[0]), replace=False)
    Xtr, Ytr = Xtr[sel], Ytr[sel]
    Xflat = Xte.reshape(-1, train_ds.in_channels)
    print(f"  fitting classical baselines on {Xtr.shape[0]} rows ...", flush=True)
    preds["Ridge"] = Ridge(alpha=1.0).fit(Xtr, Ytr).predict(Xflat).reshape(S, N, 2)
    preds["RandomForest"] = RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1,
                                                  random_state=0).fit(Xtr, Ytr).predict(Xflat).reshape(S, N, 2)
    gb = np.stack([HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, random_state=0)
                   .fit(Xtr, Ytr[:, k]).predict(Xflat) for k in range(2)], axis=1)
    preds["GradBoost"] = gb.reshape(S, N, 2)

    # ---- metrics table (PM10 channel) ----
    gt_rough = roughness(Yte[:, :, 0], rows, cols, H, W)
    print("\n" + "=" * 86)
    print(f"{'Model':<16}{'R2':>8}{'adv-resid':>11}{'mass-err%':>11}{'rough/GT':>10}   (PM10)")
    print("-" * 86)
    for name, P in preds.items():
        p10 = P[:, :, 0]
        _, r2 = metrics(P.reshape(-1, 2), Yte.reshape(-1, 2))
        ar = adv_resid(p10, cur[:, :, 0], u_te, v_te, rows, cols, H, W)
        me = mass_err(p10, Yte[:, :, 0])
        rr = roughness(p10, rows, cols, H, W) / gt_rough
        tag = "  <- reference" if name == "GroundTruth" else ""
        print(f"{name:<16}{r2[0]:>8.3f}{ar:>11.4f}{me:>11.2f}{rr:>10.2f}{tag}")
    print("=" * 86)
    print("adv-resid lower = more transport-consistent (GT = irreducible). rough/GT ~1 = realistic.")


if __name__ == "__main__":
    main()
