import torch
from torch_geometric.data import Data, Dataset
import xarray as xr
import numpy as np
import os
from src.utils.graph import build_dust_graph

class GraphDustDataset(Dataset):
    """
    Dataset for loading and preprocessing atmospheric dust data.
    Merges PM10 concentrations with ERA5 meteorological features.
    """
    def __init__(self, sfc_file, era_files, lat_dim=7, lon_dim=12):
        super().__init__()
        # Load and Merge Data
        ds_pm10 = xr.open_dataset(sfc_file)
        
        era_datasets = []
        for f in era_files:
            if os.path.exists(f):
                era_datasets.append(xr.open_dataset(f))
        
        ds_era = xr.concat(era_datasets, dim='valid_time').sortby('valid_time')
        ds = xr.merge([ds_pm10, ds_era], join='inner')
        
        # Resample to 12H steps
        ds = ds.resample(valid_time='12H').mean()
        
        # Extract Features and scale PM10
        # Fix: Scale KG/M^3 to UG/M^3 (1e9)
        self.pm10_log = np.log1p(ds['pm10'].values * 1e9)
        self.u10 = ds['u10'].values
        self.v10 = ds['v10'].values
        self.t2m = ds['t2m'].values
        
        self.num_time_steps = self.pm10_log.shape[0]
        self.grid_shape = (lat_dim, lon_dim)
        
        # Pre-compute Static Graph Structure
        self.edge_index, self.pos = build_dust_graph(lat_dim, lon_dim)
        
    def len(self):
        # Need t and t+1
        return self.num_time_steps - 1

    def get(self, idx):
        # Features at time t
        x_features = np.stack([
            self.pm10_log[idx].flatten(),
            self.u10[idx].flatten(),
            self.v10[idx].flatten(),
            self.t2m[idx].flatten()
        ], axis=1)
        
        # Target at time t+1
        y_pm10 = self.pm10_log[idx+1].flatten()
        
        # Convert to Tensors and handle NaNs
        x = torch.nan_to_num(torch.tensor(x_features, dtype=torch.float32))
        y = torch.nan_to_num(torch.tensor(y_pm10, dtype=torch.float32).view(-1, 1))

        return Data(
            x=x, 
            edge_index=self.edge_index,
            pos=self.pos, 
            y=y
        )
