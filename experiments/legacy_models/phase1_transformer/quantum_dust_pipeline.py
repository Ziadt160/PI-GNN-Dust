import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_HOURS_PHYSICAL = 168 # 7 days
OUTPUT_HOURS_PHYSICAL = 72 # 3 days
RESAMPLE_FREQ = '12H' # Resample data to 12-hour intervals for speed
FREQ_HOURS = 12

# Calculate steps
INPUT_STEPS = INPUT_HOURS_PHYSICAL // FREQ_HOURS
OUTPUT_STEPS = OUTPUT_HOURS_PHYSICAL // FREQ_HOURS

GRID_H, GRID_W = 7, 12
BATCH_SIZE = 32 # Increased batch size slightly as seq len is smaller
LEARNING_RATE = 1e-3 # Slightly higher LR for coarser data
EPOCHS = 10
N_QUBITS = 4
N_QUANTUM_LAYERS = 2
USE_QUANTUM = True # Toggle to enable/disable quantum layer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")
print(f"Quantum Layer Enabled: {USE_QUANTUM}")

# --- Dataset ---
class DustDataset(Dataset):
    def __init__(self, nc_file_pm10, era5_files, input_steps, output_steps):
# ... (Dataset code remains same, skipping for brevity in replacement if possible, but replace tool needs context. 
# Actually, I'll target the Configuration block first then the Class separately to keep chunks small or use multi_replace if they are far apart. 
# They are far apart. Configuration is at top, Class is further down.
# Let's use separate replace calls or just one large replacement if I replace the whole file? No, replace_file_content is single block.
# I will use multi_replace_file_content for this.)

        super().__init__()
        self.input_len = input_steps
        self.output_len = output_steps
        
        # 1. Load PM10 Data (Target)
        print(f"Loading PM10 from {nc_file_pm10}...")
        ds_pm10 = xr.open_dataset(nc_file_pm10)
        
        # 2. Load and Concatenate ERA5 Data (Features)
        print(f"Loading ERA5 data from {len(era5_files)} files...")
        era_datasets = []
        for f in era5_files:
            if os.path.exists(f):
                # print(f"  Found {f}") 
                era_datasets.append(xr.open_dataset(f))
            else:
                print(f"  Warning: ERA5 file {f} not found.")
        
        if not era_datasets:
            raise FileNotFoundError("No ERA5 files found.")
            
        ds_era = xr.concat(era_datasets, dim='valid_time')
        ds_era = ds_era.sortby('valid_time')
        
        # 3. Merge Datasets (Inner Join on time)
        print("Merging Datasets...")
        ds = xr.merge([ds_pm10, ds_era], join='inner')
        print(f"Merged Dataset Shape (Original): {ds.dims}")
        
        # 3b. Resample
        print(f"Resampling to {RESAMPLE_FREQ}...")
        ds = ds.resample(valid_time=RESAMPLE_FREQ).mean()
        print(f"Resampled Dataset Shape: {ds.dims}")
        
        # 4. Extract Variables
        # PM10
        self.pm10_raw = ds['pm10'].values
        self.pm10_log = np.log1p(self.pm10_raw)
        
        # Met Vars
        self.u10 = ds['u10'].values
        self.v10 = ds['v10'].values
        self.t2m = ds['t2m'].values
        
        # 5. Standardize
        print("Standardizing features...")
        self.scalers = {}
        
        def standardize_feature(data, name):
            scaler = StandardScaler()
            shape = data.shape
            flat = data.reshape(shape[0], -1)
            scaled = scaler.fit_transform(flat).reshape(shape)
            self.scalers[name] = scaler
            return scaled

        self.pm10_scaled = standardize_feature(self.pm10_log, 'pm10')
        self.u10_scaled = standardize_feature(self.u10, 'u10')
        self.v10_scaled = standardize_feature(self.v10, 'v10')
        self.t2m_scaled = standardize_feature(self.t2m, 't2m')
        
        # Feature Stack
        self.features = np.stack([
            self.pm10_scaled, 
            self.u10_scaled, 
            self.v10_scaled, 
            self.t2m_scaled
        ], axis=-1)
        
        # 6. Time Features
        times = pd.to_datetime(ds.valid_time.values)
        self.hours = times.hour.values
        self.months = times.month.values
        self.dayofweek = times.dayofweek.values
        
        # Cyclic Encoding
        # Hour (0-23)
        self.sin_hour = np.sin(2 * np.pi * self.hours / 24)
        self.cos_hour = np.cos(2 * np.pi * self.hours / 24)
        
        # Month (1-12)
        self.sin_month = np.sin(2 * np.pi * self.months / 12)
        self.cos_month = np.cos(2 * np.pi * self.months / 12)
        
        # Season Feature
        self.seasons = (self.months % 12 + 3) // 3 - 1
        
        self.n_samples = len(times) - input_steps - output_steps + 1
        print(f"Total Samples for training: {self.n_samples}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Window indices
        in_start = idx
        in_end = idx + self.input_len
        out_end = in_end + self.output_len
        
        # X: (Input_Steps, H, W, 4)
        x_in = self.features[in_start:in_end]
        
        # y: (Output_Steps, H, W)
        y_out = self.pm10_scaled[in_end:out_end]
        
        # Context (last timestep)
        ctx_idx = in_end - 1
        
        return {
            'x_in': torch.tensor(x_in, dtype=torch.float32), 
            'y_out': torch.tensor(y_out, dtype=torch.float32),
            'season': torch.tensor(self.seasons[ctx_idx], dtype=torch.long),
            'dow': torch.tensor(self.dayofweek[ctx_idx], dtype=torch.long),
            # Cyclic Features (Float)
            'sin_hour': torch.tensor(self.sin_hour[ctx_idx], dtype=torch.float32),
            'cos_hour': torch.tensor(self.cos_hour[ctx_idx], dtype=torch.float32),
            'sin_month': torch.tensor(self.sin_month[ctx_idx], dtype=torch.float32),
            'cos_month': torch.tensor(self.cos_month[ctx_idx], dtype=torch.float32)
        }

# --- Quantum Layer ---
# Check for lightning
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
    print("Using PennyLane Lightning device.")
except:
    dev = qml.device("default.qubit", wires=N_QUBITS)
    print("PennyLane Lightning not found, falling back to default.qubit (Slow).")

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x):
        return self.q_layer(x)

# --- Spatio-Temporal Transformer ---
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, grid_h, grid_w, input_hours, output_hours, in_channels=4, use_quantum=True):
        super().__init__()
        
        self.d_model = 32 # Reduced from 64
        self.use_quantum = use_quantum
        
        # Optimization: Flatten Spatial Grid into Feature Vector per Time Step
        # Input to projection: (B, T, H*W*C)
        # Token Sequence Length = T (168)
        self.flat_feature_dim = grid_h * grid_w * in_channels
        self.token_proj = nn.Linear(self.flat_feature_dim, self.d_model)
        
        # Quantum branch
        if self.use_quantum:
            self.q_proj = nn.Linear(self.flat_feature_dim, N_QUBITS) 
            self.quantum_layer = QuantumLayer(N_QUBITS, N_QUANTUM_LAYERS)
            self.q_out_proj = nn.Linear(N_QUBITS, self.d_model)
        
        # Positional Encodings
        # Sequence Length = Input_Hours (168)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_hours, self.d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=2, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Context Embeddings / Features
        self.season_emb = nn.Embedding(4, 8)
        self.dow_emb = nn.Embedding(7, 8) 
        
        # Cyclic features are continuous (4 values: sin/cos hour, sin/cos month)
        # We can project them or use directly. Let's project them to match embedding dims roughly.
        self.cyclic_proj = nn.Linear(4, 16) # Project 4 cyclic floats to 16 dim vector
        
        # Final Head
        # Head input dim
        # d_model (pooled) + q_dim + season(8) + dow(8) + cyclic(16)
        q_dim = self.d_model if self.use_quantum else 0
        self.head_in_dim = self.d_model + q_dim + 32 # 8+8+16 = 32 context dims
        
        self.output_hours = output_hours

        self.forecast_head = nn.Sequential(
            nn.Linear(self.head_in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_hours * grid_h * grid_w)
        )
        
    def forward(self, x, season, dow, sin_hour, cos_hour, sin_month, cos_month):
        # x: (B, T, H, W, C)
        B, T, H, W, C = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.reshape(B, T, -1) 
        
        # --- Quantum Path ---
        if self.use_quantum:
            x_mean_time = x_flat.mean(dim=1) 
            q_in = torch.sigmoid(self.q_proj(x_mean_time)) * np.pi 
            q_out = self.quantum_layer(q_in) 
            q_feat = self.q_out_proj(q_out) 
        
        # --- Transformer Path ---
        tokens = self.token_proj(x_flat) 
        tokens = tokens + self.pos_embedding[:, :tokens.size(1), :]
        trans_out = self.transformer(tokens)
        pooled = trans_out.mean(dim=1) 
        
        # --- Context ---
        s_emb = self.season_emb(season)
        d_emb = self.dow_emb(dow)
        
        # Cyclic Time Features (B, 4)
        cyclic_feats = torch.stack([sin_hour, cos_hour, sin_month, cos_month], dim=1)
        c_emb = self.cyclic_proj(cyclic_feats)
        
        # --- Fusion ---
        # Concatenate: Pooled Trans + (Quantum) + Season + DoW + Cyclic
        if self.use_quantum:
            combined = torch.cat([pooled, q_feat, s_emb, d_emb, c_emb], dim=1)
        else:
            combined = torch.cat([pooled, s_emb, d_emb, c_emb], dim=1)
        
        # Forecast
        out = self.forecast_head(combined)
        return out.reshape(B, self.output_hours, H, W)

# --- Training ---
# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- Persistence Baseline ---
def persistence_baseline(dataloader):
    print("Calculating Persistence Baseline...")
    mse_sum = 0
    count = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch in dataloader:
            x = batch['x_in'].to(DEVICE) # (B, T, H, W, C)
            y = batch['y_out'].to(DEVICE) # (B, Out_T, H, W)
            
            # Persistence: Predict last observed PM10 frame for all future steps
            # x[..., 0] is normalized PM10
            last_frame = x[:, -1, :, :, 0] # (B, H, W)
            
            # Expand to output shape
            preds = last_frame.unsqueeze(1).expand(-1, y.shape[1], -1, -1)
            
            loss = criterion(preds, y)
            mse_sum += loss.item()
            count += 1
            
    avg_mse = mse_sum / count
    print(f"Persistence Baseline MSE: {avg_mse:.4f}")
    return avg_mse

# --- Training ---
def train_model(model, train_loader, val_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=7, min_delta=0.0005)
    
    loss_history = {'train': [], 'val': []}
    
    print(f"Starting Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # -- Training --
        model.train()
        train_loss = 0
        train_batches = 0
        for batch in train_loader:
            x = batch['x_in'].to(DEVICE)
            y = batch['y_out'].to(DEVICE)
            season = batch['season'].to(DEVICE)
            dow = batch['dow'].to(DEVICE)
            sin_h = batch['sin_hour'].to(DEVICE)
            cos_h = batch['cos_hour'].to(DEVICE)
            sin_m = batch['sin_month'].to(DEVICE)
            cos_m = batch['cos_month'].to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(x, season, dow, sin_h, cos_h, sin_m, cos_m)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
        avg_train_loss = train_loss / train_batches
        
        # -- Validation --
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x_in'].to(DEVICE)
                y = batch['y_out'].to(DEVICE)
                season = batch['season'].to(DEVICE)
                dow = batch['dow'].to(DEVICE)
                sin_h = batch['sin_hour'].to(DEVICE)
                cos_h = batch['cos_hour'].to(DEVICE)
                sin_m = batch['sin_month'].to(DEVICE)
                cos_m = batch['cos_month'].to(DEVICE)
                
                preds = model(x, season, dow, sin_h, cos_h, sin_m, cos_m)
                loss = criterion(preds, y)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        loss_history['train'].append(avg_train_loss)
        loss_history['val'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")
        
        # Early Stopping
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            print("Early stopping triggered!")
            break
        
    return loss_history

if __name__ == "__main__":
    # Define files
    MAX_DEPTH_DATA = "../data"
    files_train = [os.path.join(MAX_DEPTH_DATA, f) for f in ['era5_2003.nc', 'era5_2004.nc', 'era5_2005.nc', 'era5_2006.nc']]
    files_val = [os.path.join(MAX_DEPTH_DATA, 'era5_2007.nc')]
    sfc_file = os.path.join(MAX_DEPTH_DATA, 'data_sfc.nc')
    
    if os.path.exists(sfc_file):
        print("Initializing Training Dataset (2003-2006)...")
        train_ds = DustDataset(sfc_file, files_train, INPUT_STEPS, OUTPUT_STEPS)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        print("Initializing Validation Dataset (2007)...")
        val_ds = DustDataset(sfc_file, files_val, INPUT_STEPS, OUTPUT_STEPS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Calculate Baseline
        baseline_mse = persistence_baseline(val_loader)
        
        # Init Model
        model = SpatioTemporalTransformer(GRID_H, GRID_W, INPUT_STEPS, OUTPUT_STEPS, in_channels=4, use_quantum=USE_QUANTUM).to(DEVICE)
        
        print("Model initialized. Starting optimized training...")
        history = train_model(model, train_loader, val_loader, epochs=50)
        
        # Save Model
        torch.save(model.state_dict(), 'quantum_dust_model_optimized.pth')
        print("Model saved to quantum_dust_model_optimized.pth")
        
        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(history['train'], label='Train Loss')
        plt.plot(history['val'], label='Val Loss')
        plt.axhline(y=baseline_mse, color='r', linestyle='--', label='Persistence Baseline')
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.savefig("training_comparison.png")
        print("Plot saved to training_comparison.png")
    else:
        print("data_sfc.nc missing.")
