import os
import math
import torch
import numpy as np
import xarray as xr
from torch_geometric.data import Data, Dataset
from src.utils.graph import build_dust_graph


class GraphDustDataset(Dataset):
    """
    Loads and preprocesses atmospheric dust data into per-time-step graphs.

    Merges CAMS PM10 with ERA5 meteorology, resamples to a fixed cadence, and
    exposes ``(features_t -> PM10_{t+1})`` samples. Features are returned in
    *physical* units; standardisation is the model's job (fitted on the train
    split only) to avoid leakage.

    Feature layout (current frame first, so physics terms can always read
    indices 0/1/2 = log_pm10/u10/v10 and 3 = t2m):

        [ log_pm10_t, u_t, v_t, t2m_t,
          (log_pm10_{t-1}, u_{t-1}, v_{t-1}), ... for window-1 past frames,
          (wind_speed_t)            if add_speed,
          (doy_sin, doy_cos)        if add_doy ]

    Options:
        window     : number of frames of history (1 = memoryless / original).
        add_speed  : append 10 m wind speed (emission scales nonlinearly with it).
        add_doy    : append day-of-year sin/cos (captures dust seasonality).
        neighbors  : 4 (N/S/E/W) or 8 (adds diagonals).
        resample   : pandas offset alias, e.g. "12h", "6h", "3h".
    """

    def __init__(self, sfc_file, era_files, lat_dim=7, lon_dim=12,
                 window=1, add_speed=False, add_doy=False, neighbors=4, resample="12h"):
        super().__init__()
        self.window = max(1, int(window))
        self.add_speed = add_speed
        self.add_doy = add_doy
        self.cells = lat_dim * lon_dim

        ds_pm10 = xr.open_dataset(sfc_file)
        era_datasets = [xr.open_dataset(f) for f in era_files if os.path.exists(f)]
        if not era_datasets:
            raise FileNotFoundError(f"No ERA5 files found among: {era_files}")
        ds_era = xr.concat(era_datasets, dim="valid_time").sortby("valid_time")
        ds = xr.merge([ds_pm10, ds_era], join="inner").resample(valid_time=resample).mean()

        # kg/m^3 -> ug/m^3 (x1e9) then log1p; flatten spatial dims to [T, cells].
        self.pm10_log = np.nan_to_num(np.log1p(ds["pm10"].values * 1e9)).reshape(-1, self.cells)
        self.u10 = np.nan_to_num(ds["u10"].values).reshape(-1, self.cells)
        self.v10 = np.nan_to_num(ds["v10"].values).reshape(-1, self.cells)
        self.t2m = np.nan_to_num(ds["t2m"].values).reshape(-1, self.cells)
        doy = ds["valid_time"].dt.dayofyear.values.astype(np.float32)
        self.doy_sin = np.sin(2 * math.pi * doy / 365.25)
        self.doy_cos = np.cos(2 * math.pi * doy / 365.25)
        self.month = ds["valid_time"].dt.month.values.astype(int)  # for climatology baseline

        self.num_time_steps = self.pm10_log.shape[0]
        self.grid_shape = (lat_dim, lon_dim)
        self.edge_index, self.pos = build_dust_graph(lat_dim, lon_dim, neighbors)

    @property
    def in_channels(self):
        return (4 + 3 * (self.window - 1)
                + (1 if self.add_speed else 0)
                + (2 if self.add_doy else 0))

    def len(self):
        # Need (window-1) past frames and one future target frame.
        return self.num_time_steps - self.window

    def _assemble(self, idx):
        """idx = current-frame index -> feature matrix [cells, in_channels]."""
        cols = [self.pm10_log[idx], self.u10[idx], self.v10[idx], self.t2m[idx]]
        for j in range(1, self.window):
            cols += [self.pm10_log[idx - j], self.u10[idx - j], self.v10[idx - j]]
        if self.add_speed:
            cols += [np.sqrt(self.u10[idx] ** 2 + self.v10[idx] ** 2)]
        if self.add_doy:
            cols += [np.full(self.cells, self.doy_sin[idx]),
                     np.full(self.cells, self.doy_cos[idx])]
        return np.stack(cols, axis=1).astype(np.float32)

    def get(self, i):
        idx = i + (self.window - 1)  # current frame
        x = torch.tensor(self._assemble(idx))
        y = torch.tensor(self.pm10_log[idx + 1], dtype=torch.float32).view(-1, 1)
        return Data(x=x, edge_index=self.edge_index, pos=self.pos, y=y)

    def compute_feature_stats(self):
        """Per-feature mean/std over the train split's samples (no leakage)."""
        rows = [self._assemble(i + (self.window - 1)) for i in range(self.len())]
        feats = np.concatenate(rows, axis=0)
        return feats.mean(0).astype(np.float32), np.clip(feats.std(0), 1e-6, None).astype(np.float32)
