"""Train the Physics-Informed GNN (PI-GNN) for dust forecasting.

3-way temporal split (configurable): train -> validation (model selection) ->
held-out test (evaluated separately by ``benchmark.py``). Input features are
standardised using statistics fitted on the **training split only**.

Run from the repository root:  python experiments/train_model.py
"""
import os
import sys
import json
import random

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from src.models.pi_gnn import PIDustModel
from src.training.dataset import GraphDustDataset
from src.training.trainer import Trainer


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_dataset(data_dir, sfc_file, years, lat_dim, lon_dim, feat):
    era = [os.path.join(data_dir, f"era5_{y}.nc") for y in years]
    return GraphDustDataset(
        os.path.join(data_dir, sfc_file), era, lat_dim, lon_dim,
        window=feat.get("window", 1),
        add_speed=feat.get("add_speed", False),
        add_doy=feat.get("add_doy", False),
        neighbors=feat.get("neighbors", 4),
        resample=feat.get("resample", "12h"),
    )


def main(config_path="configs/default_config.yaml"):
    with open(resolve(config_path), "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = resolve(config["data"]["dir"])
    sfc = config["data"]["sfc_file"]
    lat_dim, lon_dim = config["physics"]["lat_dim"], config["physics"]["lon_dim"]
    feat = config.get("features", {})

    print("Creating datasets (train / val) ...")
    train_ds = make_dataset(data_dir, sfc, config["training"]["years"], lat_dim, lon_dim, feat)
    val_ds = make_dataset(data_dir, sfc, config["validation"]["years"], lat_dim, lon_dim, feat)
    print(f"  train samples: {len(train_ds)} | val samples: {len(val_ds)} | in_channels: {train_ds.in_channels}")

    bs = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)

    # Fit normalisation on the TRAIN split only (no leakage) and bake into the model.
    mean, std = train_ds.compute_feature_stats()
    print(f"  feature mean={np.round(mean, 3)}  std={np.round(std, 3)}")

    model = PIDustModel(
        in_channels=train_ds.in_channels,
        hidden_channels=config["model"]["hidden_channels"],
        hops=config["model"]["hops"],
    ).to(device)
    model.set_norm_stats(mean, std)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    physics_cfg = {
        "delta_t": config["physics"]["delta_t"],
        "dx": config["physics"]["dx"],
        "dy": config["physics"]["dy"],
        "grid_size": (lat_dim, lon_dim),
    }
    trainer = Trainer(model, optimizer, device, physics_cfg=physics_cfg)

    model_out = resolve(config["paths"]["model_out"])
    os.makedirs(os.path.dirname(model_out), exist_ok=True)

    best_val_mse = float("inf")
    epochs = config["training"]["epochs"]
    history = {"train_loss": [], "val_loss": [], "val_mse": []}

    print("Starting training ...")
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_mse = trainer.validate(val_loader)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mse"].append(val_mse)

        marker = ""
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), model_out)
            marker = " -> saved best"
        print(f"Epoch {epoch + 1:3d}/{epochs} | train_wloss {train_loss:7.4f} "
              f"| val_wloss {val_loss:7.4f} | val_mse {val_mse:.4f}{marker}")

    # Persist normalisation stats alongside the checkpoint.
    stats_path = resolve(config["paths"]["norm_stats"])
    with open(stats_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    print(f"\nBest validation MSE: {best_val_mse:.4f}")
    print(f"Saved model -> {model_out}")
    print(f"Saved feature stats -> {stats_path}")

    _save_loss_curve(history, resolve(config["paths"]["loss_curve"]))


def _save_loss_curve(history, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipped loss curve: {exc})")
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, history["train_loss"], label="train (weighted)", color="tab:blue")
    ax1.plot(epochs, history["val_loss"], label="val (weighted)", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weighted loss")
    ax2 = ax1.twinx()
    ax2.plot(epochs, history["val_mse"], label="val MSE", color="tab:green", linestyle="--")
    ax2.set_ylabel("Validation MSE (log-space)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title("PI-GNN training convergence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Saved loss curve -> {out_path}")


if __name__ == "__main__":
    main()
