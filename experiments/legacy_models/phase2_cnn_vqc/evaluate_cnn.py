import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import os

# Import from pipeline
from cnn_vqc_dust_pipeline import CNNQCModel, DustDataset, GRID_H, GRID_W, INPUT_STEPS, OUTPUT_STEPS, USE_QUANTUM, DEVICE

def evaluate():
    print(f"Evaluating on Device: {DEVICE}")
    
    # 1. Load Data
    DATA_DIR = "../data"
    files_val = [os.path.join(DATA_DIR, 'era5_2007.nc')]
    val_ds = DustDataset(os.path.join(DATA_DIR, 'data_sfc.nc'), files_val, INPUT_STEPS, OUTPUT_STEPS)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # 2. Load Model
    model = CNNQCModel(GRID_H, GRID_W, INPUT_STEPS, OUTPUT_STEPS, use_quantum=USE_QUANTUM).to(DEVICE)
    MODEL_PATH = "../models/best_cnn_vqc.pth"
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 3. Predict
    all_preds = []
    all_targets = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in val_loader:
            x = batch['x_in'].to(DEVICE)
            y = batch['y_out'].to(DEVICE)
            s = batch['season'].to(DEVICE)
            d = batch['dow'].to(DEVICE)
            sh = batch['sin_hour'].to(DEVICE)
            ch = batch['cos_hour'].to(DEVICE)
            sm = batch['sin_month'].to(DEVICE) if 'sin_month' in batch else batch.get('sin_month', torch.zeros(x.shape[0])).to(DEVICE)
            
            # Note: Pipeline used 'sin_month' in train_model loop but dataset returns it.
            # Let's check dataset keys in pipeline...
            # The dataset returns 'sin_month' etc.
            # But the batch keys might be standard.
            
            sm = batch['sin_month'].to(DEVICE)
            cm = batch['cos_month'].to(DEVICE)
            
            preds = model(x, s, d, sh, ch, sm, cm)
            
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 4. Metrics
    # Flatten
    y_true = all_targets.flatten()
    y_pred = all_preds.flatten()
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print("-" * 30)
    print(f"Evaluation Results (2007 Validation Set)")
    print("-" * 30)
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")
    print("-" * 30)
    
    # 5. Interpretation
    print("\nInterpretation:")
    print(f"An R2 score of {r2:.2f} means the model explains {r2*100:.1f}% of the variance in the dust data.")
    print(f"RMSE of {rmse:.2f} means predictions are typically off by {rmse:.2f} in log-space.")
    
    # 6. Plot Sample
    # Pick first sample, first time step
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title("Actual (Sample 0, t=0)")
    plt.imshow(all_targets[0, 0, :, :])
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.title("Predicted (Sample 0, t=0)")
    plt.imshow(all_preds[0, 0, :, :])
    plt.colorbar()
    
    plt.savefig("cnn_evaluation.png")
    print("Saved comparison plot to cnn_evaluation.png")

if __name__ == "__main__":
    evaluate()
