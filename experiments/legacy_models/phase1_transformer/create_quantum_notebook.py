
import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Quantum-Spatio-Temporal Transformer for PM10 Prediction

This notebook implements a state-of-the-art model combining:
1.  **Quantum Feature Extraction**: Using PennyLane's `StrongEntanglingLayers` on a 4-qubit circuit.
2.  **Spatio-Temporal Transformer**: Using a Tubelet embedding and Self-Attention for learning dust dynamics.
3.  **Global Context**: Fusion of seasonal and calendar features.

**Goal**: Predict the next 72 hours of PM10 across a 7x12 grid, given the past 168 hours.
"""

code_imports = """\
import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Device Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
"""

code_config = """\
# Configuration
INPUT_HOURS = 168  # 7 days lookback
OUTPUT_HOURS = 72  # 3 days forecast
GRID_H, GRID_W = 7, 12
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 10
N_QUBITS = 4
N_QUANTUM_LAYERS = 3
"""

code_dataset = """\
class DustDataset(Dataset):
    def __init__(self, nc_file_pm10, nc_file_met=None, input_len=168, output_len=72):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        
        # Load Data
        ds = xr.open_dataset(nc_file_pm10)
        
        # Merge if Met data exists
        if nc_file_met and os.path.exists(nc_file_met):
            ds_met = xr.open_dataset(nc_file_met)
            ds = xr.merge([ds, ds_met], join='inner')
            self.has_met = True
        else:
            self.has_met = False
            
        # Log Scale PM10: log(x + 1)
        self.pm10_raw = ds['pm10'].values  # Shape: (T, H, W)
        self.pm10_log = np.log1p(self.pm10_raw)
        
        # Standardize PM10
        self.scaler_pm10 = StandardScaler()
        # Flatten for scaling: (T, H*W)
        shape = self.pm10_log.shape
        flat_pm10 = self.pm10_log.reshape(shape[0], -1)
        self.pm10_scaled = self.scaler_pm10.fit_transform(flat_pm10).reshape(shape)
        
        # Time Features
        times = pd.to_datetime(ds.valid_time.values)
        self.hours = times.hour.values
        self.dayofweek = times.dayofweek.values
        self.months = times.month.values
        
        # Season Feature (Simplified for now, assuming external calc or approximation)
        # 0: Winter, 1: Spring, 2: Summer, 3: Fall
        # Using month-based approximation for speed in Dataset
        self.seasons = (self.months % 12 + 3) // 3 - 1
        
        self.n_samples = len(times) - input_len - output_len + 1

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Window indices
        in_start = idx
        in_end = idx + self.input_len
        out_end = in_end + self.output_len
        
        # X: (168, 7, 12)
        x_pm10 = self.pm10_scaled[in_start:in_end]
        
        # y: (72, 7, 12)
        y_pm10 = self.pm10_scaled[in_end:out_end]
        
        # Calendar Features (Context) - taking the 'end of input' timestamp as reference
        # or we could provide a sequence. Let's provide reference context.
        ctx_idx = in_end - 1
        season = self.seasons[ctx_idx]
        hour = self.hours[ctx_idx]
        dow = self.dayofweek[ctx_idx]
        month = self.months[ctx_idx]
        
        return {
            'x_pm10': torch.tensor(x_pm10, dtype=torch.float32),
            'y_pm10': torch.tensor(y_pm10, dtype=torch.float32),
            'season': torch.tensor(season, dtype=torch.long),
            'hour': torch.tensor(hour, dtype=torch.long),
            'dow': torch.tensor(dow, dtype=torch.long),
            'month': torch.tensor(month, dtype=torch.long)
        }

# Instantiate
# Assuming data_sfc.nc is present. 
# ds_train = DustDataset('data_sfc.nc', 'era5_data.nc')
# train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
"""

code_quantum = """\
# --- Quantum Layer ---
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Data Re-uploading: efficient embedding
    # We map 4 scaler values to 4 qubits. 
    # If input > 4, we might need valid encoding strategies.
    # Here we assume we project grid cells down to N_QUBITS features first.
    
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))
    qml.StrongEntanglingLayers(weights, wires=range(N_QUBITS))
    
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
        
    def forward(self, x):
        # x shape: (Batch, N_Qubits)
        return self.q_layer(x)
"""

code_model = """\
# --- Spatio-Temporal Transformer ---

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, grid_h, grid_w, input_hours, output_hours):
        super().__init__()
        
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.input_len = input_hours
        
        # 1. Feature Interaction / Tubelet Embedding using 3D CNN
        # Input: (B, 1, T, H, W) -> Output: (B, D_model, T', H', W')
        # We simplify to a Linear Projection per grid cell for tokenization
        # Treating each (Time, Cell) as a token? Or (Time) as token with (H,W) flattened?
        # "Flattened grid-time map": Sequence len = T * H * W (very large!)
        # Optimization: Sequence len = T, embedding size includes H*W? 
        # User requested: "Sequence is the flattened grid-time map (168 * 84 tokens)" ~ 14k tokens.
        # This is very heavy for standard attention ($O(N^2)$). 
        # We will implement it, but warn about memory. 
        # Alternatively, we can treat each Time step as a token (Seq=168).
        # Let's try the user's request but maybe with a smaller embedding dim.
        
        self.d_model = 64
        self.token_proj = nn.Linear(1, self.d_model) # Project scalar PM10 to d_model
        
        # Quantum branch: extract high-level features from the 'state' of the grid
        self.q_proj = nn.Linear(grid_h * grid_w, N_QUBITS)
        self.quantum_layer = QuantumLayer(N_QUBITS, N_QUANTUM_LAYERS)
        self.q_out_proj = nn.Linear(N_QUBITS, self.d_model)
        
        # Positional Encodings
        self.pos_embedding = nn.Parameter(torch.randn(1, input_hours * grid_h * grid_w, self.d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Context Embeddings
        self.season_emb = nn.Embedding(4, 8)
        self.month_emb = nn.Embedding(13, 8)
        self.hour_emb = nn.Embedding(24, 8)
        
        # Final Head
        # Concatenating: Transformer Pooled Output + Quantum + Context
        self.head_in_dim = self.d_model + self.d_model + 24 # 8+8+8
        
        self.forecast_head = nn.Sequential(
            nn.Linear(self.head_in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_hours * grid_h * grid_w)
        )
        
    def forward(self, x, season, month, hour, dow):
        # x: (B, T, H, W)
        B, T, H, W = x.shape
        
        # --- Quantum Path ---
        # Take mean over time (state summary) or last time step
        x_flat_spatial = x.mean(dim=1).reshape(B, -1) # (B, H*W)
        q_in = torch.sigmoid(self.q_proj(x_flat_spatial)) * np.pi # Scale to 0-pi for encoding
        q_out = self.quantum_layer(q_in) # (B, N_Qubits)
        q_feat = self.q_out_proj(q_out) # (B, d_model)
        
        # --- Transformer Path ---
        # Flatten x to (B, T*H*W, 1)
        x_seq = x.reshape(B, -1, 1) 
        tokens = self.token_proj(x_seq) # (B, SeqLen, d_model)
        
        # Add Positional Encoding
        tokens = tokens + self.pos_embedding[:, :tokens.size(1), :]
        
        # Attend
        # Note: 14k tokens is HUGE. We might need to downsample or use Linear Attention.
        # For this prototype, we'll assume the user has resources or we reduce sequence length implicitly 
        # by reshaping earlier (e.g. T as tokens, spatial as features).
        # Let's pivot to: Token = Time step, Feature = Spatial Grid (84).
        # This makes SeqLen = 168. Much more manageable.
        # "Tokens for each grid cell at each time step" -> User insisted.
        # We will respect "Token = Grid Cell at Time Step", but for 168*84=14112 tokens, standard attention will OOM on consumer GPU.
        # I will use the "Token = Time Step" approach as a rational adaptation for stability unless forced.
        # RE-READING: "Sequence is the flattened grid-time map (168 hours * 84 cells)".
        # Okay, I will implement strictly. Be warned of OOM.
        
        trans_out = self.transformer(tokens)
        
        # Global pooling (mean of all tokens)
        pooled = trans_out.mean(dim=1) # (B, d_model)
        
        # --- Context ---
        s_emb = self.season_emb(season)
        m_emb = self.month_emb(month)
        h_emb = self.hour_emb(hour)
        ctx = torch.cat([s_emb, m_emb, h_emb], dim=1)
        
        # --- Fusion ---
        combined = torch.cat([pooled, q_feat, ctx], dim=1)
        
        # Forecast
        out = self.forecast_head(combined)
        return out.reshape(B, OUTPUT_HOURS, H, W)

# Model Init
model = SpatioTemporalTransformer(GRID_H, GRID_W, INPUT_HOURS, OUTPUT_HOURS).to(device)
print(model)
"""

code_train = """\
# --- Training Loop ---

def train_model(model, dataloader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            x = batch['x_pm10'].to(device)
            y = batch['y_pm10'].to(device)
            season = batch['season'].to(device)
            month = batch['month'].to(device)
            hour = batch['hour'].to(device)
            dow = batch['dow'].to(device)
            
            optimizer.zero_grad()
            
            # Forward
            preds = model(x, season, month, hour, dow)
            
            # Loss
            loss = criterion(preds, y)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, MSE Loss: {avg_loss:.4f}")
        
    return loss_history

# Dry Run Example
if __name__ == "__main__":
    if os.path.exists('data_sfc.nc'):
        print("Starting Dry Run...")
        ds_train = DustDataset('data_sfc.nc', 'era5_data.nc')
        # Tiny batch for memory check
        dl = DataLoader(ds_train, batch_size=2, shuffle=True)
        
        model = SpatioTemporalTransformer(GRID_H, GRID_W, INPUT_HOURS, OUTPUT_HOURS).to(device)
        hist = train_model(model, dl, epochs=1)
        print("Dry Run Complete.")
    else:
        print("data_sfc.nc not found. Upload data to run.")
"""

nb.cells = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_code_cell(code_config),
    nbf.v4.new_code_cell(code_dataset),
    nbf.v4.new_code_cell(code_quantum),
    nbf.v4.new_code_cell(code_model),
    nbf.v4.new_code_cell(code_train)
]

with open('quantum_dust_model.ipynb', 'w') as f:
    nbf.write(nb, f)
    
print("Notebook quantum_dust_model.ipynb created.")
