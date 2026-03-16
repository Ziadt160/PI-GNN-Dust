import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import numpy as np
import os
import time
import pandas as pd
from tqdm import tqdm

from pi_dgn import PIDustModel, build_dust_graph
from train_pi_dgn import GraphDustDataset, DATA_DIR, DEVICE

def load_val_data():
    sfc_file = os.path.join(DATA_DIR, 'data_sfc.nc')
    # Use validation year for benchmarking
    val_files = [os.path.join(DATA_DIR, 'era5_2007.nc')]
    print("Loading Validation Data...")
    return GraphDustDataset(sfc_file, val_files)

def benchmark():
    print("\n=== PI-DGN Production Benchmark Suite ===")
    
    # 1. Setup
    dataset = load_val_data()
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # Batch 1 for rollout clarity
    
    model = PIDustModel().to(DEVICE)
    try:
        model.load_state_dict(torch.load('best_pi_dgn.pth', map_location=DEVICE, weights_only=True))
        print("Loaded 'best_pi_dgn.pth' successfully.")
    except FileNotFoundError:
        print("Error: 'best_pi_dgn.pth' not found. Train the model first.")
        return

    model.eval()
    mse_crit = nn.MSELoss()
    
    # --- Test 1: Baseline Comparison (Model vs Persistence) ---
    print("\n[Test 1] Baseline Comparison (1-Step Ahead)")
    total_mse_model = 0
    total_mse_persist = 0
    count = 0
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Benchmarking"):
            data = data.to(DEVICE)
            
            # Model Pred
            pred, _ = model(data)
            mse_model = mse_crit(pred, data.y).item()
            
            # Persistence Pred (Next = Current)
            # data.x[:, 0] is PM10 current (Ground Truth at t)
            # data.y is PM10 next (Ground Truth at t+1)
            # Fix 2: Calculate Persistence MSE (comparing t vs t+1)
            gt_t = data.x[:, 0:1]
            gt_next = data.y
            mse_persist = mse_crit(gt_t, gt_next).item()
            
            total_mse_model += mse_model
            total_mse_persist += mse_persist
            count += 1
            
    avg_mse_model = total_mse_model / count
    avg_mse_persist = total_mse_persist / count
    
    print(f"  Persistence MSE: {avg_mse_persist:.5f}")
    print(f"  PI-DGN MSE:      {avg_mse_model:.5f}")
    print(f"  Improvement:     {((avg_mse_persist - avg_mse_model)/avg_mse_persist)*100:.2f}%")
    
    # --- Test 2: Autoregressive Rollout (Stability) ---
    print("\n[Test 2] Autoregressive Rollout (24h / 2 steps @ 12H resolution)")
    # Since our data is 12H, 24h is just 2 steps. Let's try 6 steps (3 days) for stress test.
    steps = 6 
    
    # Get a starting sample
    start_idx = 0
    data = dataset.get(start_idx)
    
    current_x = data.x.clone().to(DEVICE) # [Nodes, 4]
    
    # We need the FUTURE ERA5 data (forcing) to rollout truly, 
    # but for simple stability check, we can just use the ERA5 from the dataset at t+k
    # Or assuming "forecast" ERA5 is available.
    # Here we simulate the rollout loop.
    
    print(f"  Starting Rollout for {steps} steps (72 hours)...")
    rollout_preds = []
    
    with torch.no_grad():
        curr_graph = data
        curr_graph.x = curr_graph.x.to(DEVICE)
        curr_graph.edge_index = curr_graph.edge_index.to(DEVICE)
        curr_graph.pos = curr_graph.pos.to(DEVICE)
        
        for s in range(steps):
            # Predict
            pm10_next, _ = model(curr_graph)
            rollout_preds.append(pm10_next.mean().item())
            
            # Update state for next step
            # We need valid ERA5 for next step. Let's cheat and grab it from dataset
            next_real_data = dataset.get(start_idx + s + 1)
            
            # Construct next input: Predicted PM10 + Future Wind/Temp
            next_x = torch.cat([
                pm10_next, # Prediction
                torch.tensor(next_real_data.x[:, 1:], device=DEVICE) # Future Forcing
            ], dim=1)
            
            curr_graph.x = next_x
            
    print(f"  Mean PM10 Trajectory: {['{:.3f}'.format(x) for x in rollout_preds]}")
    # Check for explosion (NaN or > 100 in log space would be huge)
    if any(np.isnan(rollout_preds)) or max(rollout_preds) > 20:
        print("  RESULT: FAILED (Explosion detected)")
    else:
        print("  RESULT: PASSED (Stable)")
        
        
    # --- Test 3: Physical Sanity (Wind Flip) ---
    print("\n[Test 3] Wind Sensitivity Check")
    data = dataset.get(10) # Random sample
    data = data.to(DEVICE)
    # Batch it (unsqueeze)
    # Actually model handles [N, F] fine without batch dim for single graph if edge_index corrects
    # But usually need batch vector. 
    # Geometric DataLoader handles this. Let's use loader again for a single batch.
    sample_loader = DataLoader([dataset.get(10)], batch_size=1)
    data = next(iter(sample_loader)).to(DEVICE)
    
    with torch.no_grad():
        # Original
        _, delta_orig = model(data)
        
        # Flipped Wind
        data_flip = data.clone()
        # Flip U and V (indices 1 and 2)
        data_flip.x[:, 1] = -data_flip.x[:, 1]
        data_flip.x[:, 2] = -data_flip.x[:, 2]
        
        _, delta_flip = model(data_flip)
        
        # Compare deltas
        # Correlation
        corr = torch.corrcoef(torch.stack([delta_orig.flatten(), delta_flip.flatten()]))[0, 1].item()
        print(f"  Correlation between Original and Flipped Wind Delta: {corr:.4f}")
        
        if corr < 0.9: # If wind matters, changing it drastically should change the pattern (lower corr)
            print("  RESULT: PASSED (Sensitive to Wind)")
        else:
            print("  RESULT: WARNING (Model may be ignoring wind)")
            

    # --- Test 4: Robustness (Masking) ---
    print("\n[Test 4] Robustness (20% Masking)")
    mask_mse = 0
    clean_mse = 0
    
    with torch.no_grad():
        # Single pass on subset
        data = next(iter(loader)).to(DEVICE)
        
        # Clean
        pred_clean, _ = model(data)
        clean_mse = mse_crit(pred_clean, data.y).item()
        
        # Masked
        mask = torch.rand_like(data.x[:, 0]) > 0.2 # Keep 80%
        data_masked = data.clone()
        # Zero out masked nodes (PM10 feature only or all?)
        # Let's zero out PM10 input for 20% of nodes
        data_masked.x[~mask, 0] = 0 
        
        pred_masked, _ = model(data_masked)
        mask_mse = mse_crit(pred_masked, data.y).item()
        
    print(f"  Clean MSE:  {clean_mse:.5f}")
    print(f"  Masked MSE: {mask_mse:.5f}")
    print(f"  degradation: {mask_mse - clean_mse:.5f}")
    
    if mask_mse < clean_mse * 2.0:
        print("  RESULT: PASSED (Robustness acceptable)")
    else:
        print("  RESULT: WARNING (High Sensitivity to defects)")
        
    # --- Test 5: Inference Speed ---
    print("\n[Test 5] Latency Profiling")
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(data)
    end = time.time()
    avg_time = (end - start) / 100
    print(f"  Avg Inference Time: {avg_time*1000:.2f} ms")
    if avg_time < 0.05:
         print("  RESULT: PASSED (Production Ready Speed)")
    else:
         print("  RESULT: WARNING (Slow)")

if __name__ == "__main__":
    benchmark()
