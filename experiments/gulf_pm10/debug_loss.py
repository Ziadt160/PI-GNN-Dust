"""Inspect the magnitude of the physics (advection) residual on real data.

Useful for sanity-checking the scale of the data vs. advection terms that feed
the homoscedastic loss balancing. Uses the ground-truth next state as the
"prediction" so the printed residual reflects how well the discrete advection
equation holds for the observations themselves.

Run from the repository root:  python experiments/gulf_pm10/debug_loss.py
"""
import os
import sys

import torch
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from src.training.dataset import GraphDustDataset


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def main():
    with open(resolve("configs/default_config.yaml")) as f:
        cfg = yaml.safe_load(f)

    data_dir = resolve(cfg["data"]["dir"])
    lat_dim, lon_dim = cfg["physics"]["lat_dim"], cfg["physics"]["lon_dim"]
    dt, dx, dy = cfg["physics"]["delta_t"], cfg["physics"]["dx"], cfg["physics"]["dy"]

    era = [os.path.join(data_dir, f"era5_{cfg['training']['years'][0]}.nc")]
    ds = GraphDustDataset(os.path.join(data_dir, cfg["data"]["sfc_file"]), era, lat_dim, lon_dim)

    data = ds.get(0)
    pm10_old = data.x[:, 0:1]
    pm10_new = data.y               # ground-truth next state
    u10, v10 = data.x[:, 1], data.x[:, 2]

    pm10_grid = pm10_old.view(1, lat_dim, lon_dim)
    grad = torch.gradient(pm10_grid, spacing=(dy, dx), dim=(1, 2))
    grad_north = (-grad[0]).reshape(-1, 1)   # row+ is south -> negate
    grad_east = grad[1].reshape(-1, 1)
    advection = (u10.view(-1, 1) * grad_east + v10.view(-1, 1) * grad_north)

    change = pm10_new - pm10_old
    residual = change + advection * dt

    print("-" * 40)
    print(f"d(log PM10)  : mean={change.abs().mean():.6f}  max={change.abs().max():.6f}")
    print(f"advection*dt : mean={(advection * dt).abs().mean():.6f}  max={(advection * dt).abs().max():.6f}")
    print(f"residual     : mean={residual.abs().mean():.6f}  max={residual.abs().max():.6f}")
    print(f"physics loss : {torch.mean(residual ** 2).item():.8f}")
    print("-" * 40)


if __name__ == "__main__":
    main()
