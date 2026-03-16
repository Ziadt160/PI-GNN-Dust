import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import xarray as xr
import numpy as np
import os
import glob
from tqdm import tqdm

from pi_dgn import PIDustModel, build_dust_graph, physics_loss

# --- Configuration ---
DATA_DIR = r"d:\Quantum Projects\DustProject\data"
BATCH_SIZE = 32
# EPOCHS = 50
EPOCHS = 50
LEARNING_RATE = 1e-3
# Fix: Accuracy Focus Configuration
PHYSICS_WEIGHT = 0.0 # Not used directly anymore (Learnable)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphDustDataset(Dataset):
    def __init__(self, sfc_file, era_files):
        super().__init__()
        # 1. Load and Merge Data
        print("Loading and merging data...")
        ds_pm10 = xr.open_dataset(sfc_file)
        
        era_datasets = []
        for f in era_files:
            if os.path.exists(f):
                era_datasets.append(xr.open_dataset(f))
        
        ds_era = xr.concat(era_datasets, dim='valid_time').sortby('valid_time')
        ds = xr.merge([ds_pm10, ds_era], join='inner')
        
        # Resample to 12H steps
        ds = ds.resample(valid_time='12H').mean()
        
        # 2. Extract Values
        # (Time, Lat, Lon) -> Flatten Lat/Lon to Nodes
        # Fix: Scale KG/M^3 to UG/M^3 (1e9)
        self.pm10_log = np.log1p(ds['pm10'].values * 1e9) # (T, 7, 12)
        self.u10 = ds['u10'].values
        self.v10 = ds['v10'].values
        self.t2m = ds['t2m'].values
        
        print(f"  Dataset Stats:")
        print(f"    PM10 Log (Scaled 1e9): Shape={self.pm10_log.shape}, Mean={np.nanmean(self.pm10_log):.4f}, Max={np.nanmax(self.pm10_log):.4f}")
        print(f"    U10:      Shape={self.u10.shape}, Mean={np.nanmean(self.u10):.4f}")
        print(f"    V10:      Shape={self.v10.shape}, Mean={np.nanmean(self.v10):.4f}")
        
        if self.pm10_log.shape[0] == 0:
             print("    WARNING: Dataset is empty after merge!")
        
        self.num_time_steps = self.pm10_log.shape[0]
        self.grid_shape = (7, 12)
        self.num_nodes = 7 * 12
        
        # 3. Pre-compute Graph Structure (Static)
        self.edge_index, self.pos = build_dust_graph(7, 12)
        
    def len(self):
        # We need t+1 for target, so length is T - 1
        return self.num_time_steps - 1

    def get(self, idx):
        # Input state at time t
        x_pm10 = self.pm10_log[idx].flatten()
        x_u10 = self.u10[idx].flatten()
        x_v10 = self.v10[idx].flatten()
        x_t2m = self.t2m[idx].flatten()
        
        # Target state at time t+1
        y_pm10 = self.pm10_log[idx+1].flatten()
        
        # Stack features: [PM10, U10, V10, T2M]
        x_features = np.stack([x_pm10, x_u10, x_v10, x_t2m], axis=1)
        
        # Convert to Tensor
        x = torch.tensor(x_features, dtype=torch.float32)
        y = torch.tensor(y_pm10, dtype=torch.float32).view(-1, 1) # [Nodes, 1]
        
        # Handle NaNs (Critical for real data)
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)

        # Create Data object
        # We share edge_index and pos, but cloning ensures safety in DataLoader
        data = Data(
            x=x, 
            edge_index=self.edge_index.clone(),
            pos=self.pos.clone(), 
            y=y
        )
        
        return data

def train():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Datasets
    sfc_file = os.path.join(DATA_DIR, 'data_sfc.nc')
    
    # Train: 2003-2006
    train_era = [os.path.join(DATA_DIR, f'era5_{year}.nc') for year in [2003, 2004, 2005, 2006]]
    # Val: 2007
    val_era = [os.path.join(DATA_DIR, 'era5_2007.nc')]
    
    print("Creating Training Set...")
    train_dataset = GraphDustDataset(sfc_file, train_era)
    print("Creating Validation Set...")
    val_dataset = GraphDustDataset(sfc_file, val_era)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Model & Optimizer
    model = PIDustModel().to(DEVICE)
    
    # Advanced Optimization: Learnable Loss Weights (Homoscedastic Uncertainty)
    # log_vars = [log_var_sup, log_var_phys]
    # Initialize to 0.0 (sigma=1.0)
    log_vars = nn.Parameter(torch.zeros(2, device=DEVICE))
    
    # Add model params AND log_vars to optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + [log_vars], lr=LEARNING_RATE)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    mse_loss = nn.MSELoss()
    
    
    best_val_loss = float('inf')
    
    # Annealing Setup - Removed as Homoscedastic Loss is used
    # current_phys_weight = PHYSICS_WEIGHT
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        p_loss_accum = 0
        supervised_loss_accum = 0
        
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            # Forward
            # Fix: Disable Masking for Production Deployment (Accuracy Focus)
            # mask_prob = 0.5
            # mask = torch.rand_like(data.x[:, 0]) < mask_prob
            
            # x_input = data.x.clone()
            # x_input[mask, 0] = 0.0 # Zero out PM10
            
            # Create a new Data object for the model to ensure it uses the modified x
            # data_input = Data(x=x_input, edge_index=data.edge_index, pos=data.pos)
            
            # prediction (next state), delta
            pm10_next, delta_pm10 = model(data)
            
            # Loss Calculation
            # 1. Supervised (MSE on PM10 values)
            l_sup = mse_loss(pm10_next, data.y)
            
            # 2. Physics Residual
            # Use ORIGINAL data.x (Ground Truth) for Physics check?
            # If we used x_input (0), the effective 'change' would be huge (0 -> Target).
            # We want the 'Transition' to be physical. 
            # Ideally: Predicted State should be consistent with Previous State.
            # If model reconstructs Previous State implicitly, then (Reconstructed -> Next) is physical.
            # But here model outputs Next directly.
            # Let's use ORIGINAL data.x to enforce that (True Old -> Predicted New) is valid physics.
            l_phys = physics_loss(
                data.x[:, 0:1], # Old state (True)
                pm10_next,      # Predicted New State
                data.x[:, 1],   # U10
                data.x[:, 2]    # V10
            )
            
            # Total Loss (Homoscedastic Uncertainty Weighting)
            # L = 0.5 * exp(-log_var) * Loss + 0.5 * log_var
            loss = (0.5 * torch.exp(-log_vars[0]) * l_sup + 0.5 * log_vars[0]) + \
                   (0.5 * torch.exp(-log_vars[1]) * l_phys + 0.5 * log_vars[1])
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            supervised_loss_accum += l_sup.item()
            p_loss_accum += l_phys.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        avg_sup = supervised_loss_accum / len(train_loader)
        avg_phys = p_loss_accum / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                pm10_next, _ = model(data)
                
                l_sup = mse_loss(pm10_next, data.y)
                l_phys = physics_loss(data.x[:, 0:1], pm10_next, data.x[:, 1], data.x[:, 2])
                
                l_sup = mse_loss(pm10_next, data.y)
                l_phys = physics_loss(data.x[:, 0:1], pm10_next, data.x[:, 1], data.x[:, 2])
                
                # Val Loss follows same weighted formula for consistency
                loss = (0.5 * torch.exp(-log_vars[0]) * l_sup + 0.5 * log_vars[0]) + \
                       (0.5 * torch.exp(-log_vars[1]) * l_phys + 0.5 * log_vars[1])
                
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Step Scheduler
        scheduler.step(avg_val_loss)
        
        # Calculate current sigmas for logging
        with torch.no_grad():
            sigma_sup = torch.exp(0.5 * log_vars[0]).item()
            sigma_phys = torch.exp(0.5 * log_vars[1]).item()
        
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} (Sup: {avg_sup:.4f}, Phys: {avg_phys:.4f}) | Val Loss {avg_val_loss:.4f} | Sigmas: Sup={sigma_sup:.3f}, Phys={sigma_phys:.3f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_pi_dgn.pth')
            print("  -> Saved Best Model")

if __name__ == "__main__":
    train()
