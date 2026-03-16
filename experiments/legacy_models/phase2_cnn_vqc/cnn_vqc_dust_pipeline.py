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
INPUT_HOURS_PHYSICAL = 168
OUTPUT_HOURS_PHYSICAL = 72
RESAMPLE_FREQ = '12H'
FREQ_HOURS = 12

INPUT_STEPS = INPUT_HOURS_PHYSICAL // FREQ_HOURS # 14
OUTPUT_STEPS = OUTPUT_HOURS_PHYSICAL // FREQ_HOURS # 6

GRID_H, GRID_W = 7, 12
BATCH_SIZE = 32
LEARNING_RATE = 1e-4 # Reduced for stability with CNN
EPOCHS = 50
N_QUBITS = 4
N_QUANTUM_LAYERS = 2
USE_QUANTUM = True 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# --- Dataset (Reused with Fix) ---
class DustDataset(Dataset):
    def __init__(self, nc_file_pm10, era5_files, input_steps, output_steps):
        super().__init__()
        self.input_len = input_steps
        self.output_len = output_steps
        
        # 1. Load PM10
        ds_pm10 = xr.open_dataset(nc_file_pm10)
        
        # 2. Load ERA5
        era_datasets = []
        for f in era5_files:
            if os.path.exists(f):
                era_datasets.append(xr.open_dataset(f))
        
        if not era_datasets:
            raise FileNotFoundError("No ERA5 files found.")
            
        ds_era = xr.concat(era_datasets, dim='valid_time').sortby('valid_time')
        
        # 3. Merge
        ds = xr.merge([ds_pm10, ds_era], join='inner')
        # Resample
        ds = ds.resample(valid_time=RESAMPLE_FREQ).mean()
        
        # 4. Extract
        self.pm10_log = np.log1p(ds['pm10'].values)
        self.u10 = ds['u10'].values
        self.v10 = ds['v10'].values
        self.t2m = ds['t2m'].values
        
        # 5. Scalers
        self.pm10_scaled = self._scale(self.pm10_log)
        self.features = np.stack([
            self.pm10_scaled, 
            self._scale(self.u10), 
            self._scale(self.v10), 
            self._scale(self.t2m)
        ], axis=-1) # (T, H, W, 4)
        
        # 6. Time Features
        times = pd.to_datetime(ds.valid_time.values)
        self.hours = times.hour.values
        self.months = times.month.values
        self.dayofweek = times.dayofweek.values
        
        # Cyclic
        self.sin_hour = np.sin(2 * np.pi * self.hours / 24)
        self.cos_hour = np.cos(2 * np.pi * self.hours / 24)
        self.sin_month = np.sin(2 * np.pi * self.months / 12)
        self.cos_month = np.cos(2 * np.pi * self.months / 12)
        
        self.seasons = (self.months % 12 + 3) // 3 - 1
        self.n_samples = len(times) - input_steps - output_steps + 1

    def _scale(self, data):
        scaler = StandardScaler()
        shape = data.shape
        flat = data.reshape(shape[0], -1)
        return scaler.fit_transform(flat).reshape(shape)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        in_start = idx
        in_end = idx + self.input_len
        out_end = in_end + self.output_len
        
        # (T_in, H, W, 4)
        x_in = self.features[in_start:in_end]
        y_out = self.pm10_scaled[in_end:out_end]
        
        ctx_idx = in_end - 1
        return {
            'x_in': torch.tensor(x_in, dtype=torch.float32), 
            'y_out': torch.tensor(y_out, dtype=torch.float32),
            'season': torch.tensor(self.seasons[ctx_idx], dtype=torch.long),
            'dow': torch.tensor(self.dayofweek[ctx_idx], dtype=torch.long),
            'sin_hour': torch.tensor(self.sin_hour[ctx_idx], dtype=torch.float32),
            'cos_hour': torch.tensor(self.cos_hour[ctx_idx], dtype=torch.float32),
            'sin_month': torch.tensor(self.sin_month[ctx_idx], dtype=torch.float32),
            'cos_month': torch.tensor(self.cos_month[ctx_idx], dtype=torch.float32)
        }

# --- Quantum ---
try:
    dev = qml.device("lightning.qubit", wires=N_QUBITS)
except:
    dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x):
        return self.q_layer(x)

# --- CNN-VQC Model ---
class CNNQCModel(nn.Module):
    def __init__(self, grid_h, grid_w, input_steps, output_steps, in_raw_channels=4, use_quantum=True):
        super().__init__()
        self.use_quantum = use_quantum
        self.output_steps = output_steps
        
        # 1. CNN Encoder
        # Input: (B, C_in, H, W) where C_in = input_steps * raw_channels
        self.c_in = input_steps * in_raw_channels
        
        self.cnn = nn.Sequential(
            nn.Conv2d(self.c_in, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # 7x12 -> 3x6
            nn.Dropout(0.2),
            
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # -> (B, 16, 1, 1)
        )
        
        self.feature_dim = 16
        
        # 2. Quantum Layer
        if self.use_quantum:
            self.q_proj = nn.Linear(self.feature_dim, N_QUBITS)
            self.vqc = QuantumLayer(N_QUBITS, N_QUANTUM_LAYERS)
            self.q_out_proj = nn.Linear(N_QUBITS, 16)
            
        # 3. Context
        self.season_emb = nn.Embedding(4, 4)
        self.dow_emb = nn.Embedding(7, 4)
        self.cyclic_proj = nn.Linear(4, 8)
        
        # 4. Forecaster
        # Input: CNN_feat(16) + (Quantum(16)) + Season(4) + DoW(4) + Cyclic(8)
        q_dim = 16 if self.use_quantum else 0
        self.head_in = 16 + q_dim + 4 + 4 + 8 
        
        self.head = nn.Sequential(
            nn.Linear(self.head_in, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_steps * grid_h * grid_w)
        )

    def forward(self, x, season, dow, sin_hour, cos_hour, sin_month, cos_month):
        # x: (B, T, H, W, C) -> Need (B, T*C, H, W) for CNN
        B, T, H, W, C = x.shape
        x = x.permute(0, 1, 4, 2, 3) # (B, T, C, H, W)
        x = x.reshape(B, T*C, H, W) # (B, 56, 7, 12)
        
        # CNN
        feat = self.cnn(x).view(B, -1) # (B, 16)
        
        # Quantum
        q_feat = None
        if self.use_quantum:
            q_in = torch.sigmoid(self.q_proj(feat)) * np.pi
            q_out = self.vqc(q_in)
            q_feat = self.q_out_proj(q_out)
        
        # Context
        s = self.season_emb(season)
        d = self.dow_emb(dow)
        cyc = self.cyclic_proj(torch.stack([sin_hour, cos_hour, sin_month, cos_month], dim=1))
        
        # Concat
        if self.use_quantum:
            combined = torch.cat([feat, q_feat, s, d, cyc], dim=1)
        else:
            combined = torch.cat([feat, s, d, cyc], dim=1)
            
        out = self.head(combined)
        return out.reshape(B, self.output_steps, H, W)

# --- Training (Reuse Logic) ---
# ... (Training loop same as before, omitted for brevity, user has it in pipeline)
# Creating full file execution block below

def train_model(model, train_loader, val_loader, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # Added L2 Regularization
    criterion = nn.MSELoss()
    # Simplified Early Stopping
    best_loss = None
    counter = 0
    patience = 7
    
    loss_history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for b in train_loader:
            optimizer.zero_grad()
            x = b['x_in'].to(DEVICE)
            y = b['y_out'].to(DEVICE)
            # Context
            s = b['season'].to(DEVICE)
            d = b['dow'].to(DEVICE)
            sh = b['sin_hour'].to(DEVICE)
            ch = b['cos_hour'].to(DEVICE)
            sm = b['sin_month'].to(DEVICE)
            cm = b['cos_month'].to(DEVICE)
            
            p = model(x, s, d, sh, ch, sm, cm)
            loss = criterion(p, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            
        t_loss /= len(train_loader)
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for b in val_loader:
                x = b['x_in'].to(DEVICE)
                y = b['y_out'].to(DEVICE)
                # Context
                s = b['season'].to(DEVICE)
                d = b['dow'].to(DEVICE)
                sh = b['sin_hour'].to(DEVICE)
                ch = b['cos_hour'].to(DEVICE)
                sm = b['sin_month'].to(DEVICE)
                cm = b['cos_month'].to(DEVICE)
                
                p = model(x, s, d, sh, ch, sm, cm)
                v_loss += criterion(p, y).item()
        
        v_loss /= len(val_loader)
        loss_history['train'].append(t_loss)
        loss_history['val'].append(v_loss)
        print(f"Epoch {epoch+1} | T: {t_loss:.4f} V: {v_loss:.4f}")
        
        # Early Stop
        if best_loss is None or v_loss < best_loss - 0.0005:
            best_loss = v_loss
            counter = 0
            torch.save(model.state_dict(), 'best_cnn_vqc.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping.")
                break
                
    return loss_history

if __name__ == "__main__":
    # Setup
    DATA_DIR = "../data"
    files_train = [os.path.join(DATA_DIR, f) for f in ['era5_2003.nc', 'era5_2004.nc', 'era5_2005.nc', 'era5_2006.nc']]
    files_val = [os.path.join(DATA_DIR, 'era5_2007.nc')]
    sfc_file = os.path.join(DATA_DIR, 'data_sfc.nc')

    train_ds = DustDataset(sfc_file, files_train, INPUT_STEPS, OUTPUT_STEPS)
    val_ds = DustDataset(sfc_file, files_val, INPUT_STEPS, OUTPUT_STEPS)
    
    t_load = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    v_load = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    model = CNNQCModel(GRID_H, GRID_W, INPUT_STEPS, OUTPUT_STEPS, use_quantum=USE_QUANTUM).to(DEVICE)
    history = train_model(model, t_load, v_load, epochs=EPOCHS)
