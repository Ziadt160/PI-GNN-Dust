"""Held-out evaluation suite for the PI-GNN dust model.

Evaluates the trained checkpoint on the TEST split (default: 2007), which is
never used for model selection. Reports skill metrics against the persistence
baseline plus stability, wind-sensitivity, mass-conservation and latency checks.

Run from the repository root:  python experiments/benchmark.py
"""
import os
import sys
import json
import time

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.pi_gnn import PIDustModel
from src.training.dataset import GraphDustDataset
from src.physics.loss import mass_conservation_error


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def metrics(pred, true):
    mse = float(np.mean((pred - true) ** 2))
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(np.mean(np.abs(pred - true))),
        "R2": 1.0 - ss_res / ss_tot,
    }


def load_config():
    with open(resolve("configs/default_config.yaml")) as f:
        return yaml.safe_load(f)


def main():
    print("=== PI-GNN Held-out Benchmark (TEST split) ===")
    cfg = load_config()
    device = torch.device("cpu")
    data_dir = resolve(cfg["data"]["dir"])
    lat_dim, lon_dim = cfg["physics"]["lat_dim"], cfg["physics"]["lon_dim"]
    test_years = cfg["test"]["years"]
    feat = cfg.get("features", {})

    era = [os.path.join(data_dir, f"era5_{y}.nc") for y in test_years]
    ds = GraphDustDataset(
        os.path.join(data_dir, cfg["data"]["sfc_file"]), era, lat_dim, lon_dim,
        window=feat.get("window", 1), add_speed=feat.get("add_speed", False),
        add_doy=feat.get("add_doy", False), neighbors=feat.get("neighbors", 4),
        resample=feat.get("resample", "12h"),
    )
    print(f"Test years {test_years}: {len(ds)} samples | in_channels: {ds.in_channels}")

    model = PIDustModel(
        in_channels=ds.in_channels,
        hidden_channels=cfg["model"]["hidden_channels"],
        hops=cfg["model"]["hops"],
    ).to(device)
    model_path = resolve(cfg["paths"]["model_out"])
    if not os.path.exists(model_path):
        print(f"Error: checkpoint not found at {model_path}. Run train_model.py first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {model_path}")

    # --- Test 1: Skill vs persistence (1-step ahead, pooled over all cells) ---
    preds, trues, persist = [], [], []
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            p, _ = model(data)
            preds.append(p.cpu().numpy().ravel())
            trues.append(data.y.cpu().numpy().ravel())
            persist.append(data.x[:, 0:1].cpu().numpy().ravel())
    yhat, y, pp = map(np.concatenate, (preds, trues, persist))

    m_model = metrics(yhat, y)
    m_persist = metrics(pp, y)
    print("\n[Test 1] Skill vs persistence (log-space, 1-step)")
    print(f"  {'metric':<6} {'PI-GNN':>10} {'persistence':>12}")
    for k in ("MSE", "RMSE", "MAE", "R2"):
        print(f"  {k:<6} {m_model[k]:>10.4f} {m_persist[k]:>12.4f}")
    impr = (m_persist["MSE"] - m_model["MSE"]) / m_persist["MSE"] * 100
    verdict = "beats" if impr > 0 else "trails"
    print(f"  -> PI-GNN {verdict} persistence by {impr:+.2f}% MSE")

    # --- Test 2: Mass-conservation diagnostic (not an optimisation target) ---
    mc_log = mass_conservation_error(torch.tensor(yhat), torch.tensor(y)) * 100
    yhat_phys = np.expm1(np.clip(yhat, 0, None))
    y_phys = np.expm1(y)
    mc_phys = abs(yhat_phys.mean() - y_phys.mean()) / y_phys.mean() * 100
    print("\n[Test 2] Mass-conservation diagnostic (domain-mean error)")
    print(f"  log-space:      {mc_log:.2f}%")
    print(f"  physical-space: {mc_phys:.2f}%")

    # --- Test 3: Autoregressive rollout stability (uses true future forcing) ---
    steps = 6
    data = ds.get(0)
    g = data.clone()
    g.x = g.x.to(device); g.edge_index = g.edge_index.to(device); g.pos = g.pos.to(device)
    traj = []
    with torch.no_grad():
        for s in range(steps):
            pm10_next, _ = model(g)
            traj.append(float(pm10_next.mean()))
            future = ds.get(min(s + 1, len(ds) - 1)).x[:, 1:].to(device)  # true future forcing
            g.x = torch.cat([pm10_next, future], dim=1)
    stable = not (any(np.isnan(traj)) or max(traj) > 20)
    print(f"\n[Test 3] Autoregressive rollout ({steps} steps / 72 h, true forcing)")
    print(f"  mean log-PM10 trajectory: {[round(t, 3) for t in traj]}")
    print(f"  -> {'PASSED (stable)' if stable else 'FAILED (explosion)'}")

    # --- Test 4: Wind sensitivity (flip winds, expect delta to change) ---
    sample = next(iter(DataLoader([ds.get(min(10, len(ds) - 1))], batch_size=1))).to(device)
    with torch.no_grad():
        _, d_orig = model(sample)
        flip = sample.clone()
        flip.x[:, 1] = -flip.x[:, 1]
        flip.x[:, 2] = -flip.x[:, 2]
        _, d_flip = model(flip)
        corr = torch.corrcoef(torch.stack([d_orig.flatten(), d_flip.flatten()]))[0, 1].item()
    print("\n[Test 4] Wind sensitivity")
    print(f"  corr(delta_orig, delta_flipped) = {corr:.4f}")
    print(f"  -> {'sensitive to wind' if corr < 0.9 else 'WARNING: may ignore wind'}")

    # --- Test 5: Latency ---
    g1 = next(iter(DataLoader([ds.get(0)], batch_size=1))).to(device)
    for _ in range(10):
        with torch.no_grad():
            model(g1)
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            model(g1)
    ms = (time.time() - t0) / 100 * 1000
    print(f"\n[Test 5] Latency: {ms:.2f} ms/step ({'production-ready' if ms < 50 else 'slow'})")

    # Persist a machine-readable summary.
    out = {
        "test_years": test_years,
        "n_samples": len(ds),
        "model": m_model,
        "persistence": m_persist,
        "improvement_pct_mse": impr,
        "mass_cons_pct_log": mc_log,
        "mass_cons_pct_phys": float(mc_phys),
        "rollout_stable": bool(stable),
        "wind_flip_corr": corr,
        "latency_ms": ms,
    }
    summary_path = resolve("results/benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
