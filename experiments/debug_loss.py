import torch
from train_pi_dgn import GraphDustDataset, DATA_DIR
import os
import numpy as np
from pi_dgn import physics_loss

def debug():
    sfc_file = os.path.join(DATA_DIR, 'data_sfc.nc')
    # Just need one ERA file for debug
    era_files = [os.path.join(DATA_DIR, 'era5_2003.nc')]
    
    print("Loading dataset...")
    ds = GraphDustDataset(sfc_file, era_files)
    
    # Get one sample
    data = ds.get(0)
    
    # Fake a "next" prediction (just use target + noise to simulate training state)
    # or use actual target to see what "perfect" physics looks like (if physics held true)
    # Let's use `data.y` (target) as `pm10_new` to check the magnitude of the *ground truth* physics residual.
    pm10_old = data.x[:, 0:1]
    pm10_new = data.y
    u10 = data.x[:, 1]
    v10 = data.x[:, 2]
    
    print(f"Shapes: Old {pm10_old.shape}, New {pm10_new.shape}, U {u10.shape}")
    
    # Re-implement logic inside to inspect
    delta_t = 3600*12
    dt_pm10 = (pm10_new - pm10_old) / delta_t
    
    # Reshape
    pm10_flat = pm10_old.view(-1)
    batch_size = pm10_flat.size(0) // 84
    pm10_grid = pm10_flat.view(batch_size, 7, 12)
    
    grad = torch.gradient(pm10_grid, dim=(1, 2))
    grad_y = grad[0].reshape(-1, 1)
    grad_x = grad[1].reshape(-1, 1)
    
    u = u10.view(-1, 1)
    v = v10.view(-1, 1)
    
    advection = u * grad_x + v * grad_y
    residual = dt_pm10 + advection
    
    loss = torch.mean(residual**2)
    
    print("-" * 30)
    print(f"dt_pm10 stats:   Mean={dt_pm10.abs().mean():.8f}, Max={dt_pm10.abs().max():.8f}")
    print(f"Advection stats: Mean={advection.abs().mean():.8f}, Max={advection.abs().max():.8f}")
    print(f"Residual stats:  Mean={residual.abs().mean():.8f}, Max={residual.abs().max():.8f}")
    print(f"Loss value:      {loss.item():.10f}")
    print("-" * 30)

if __name__ == "__main__":
    debug()
