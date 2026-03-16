import xarray as xr
import numpy as np
import os

DATA_DIR = r"d:\Quantum Projects\DustProject\data"
SFC_FILE = os.path.join(DATA_DIR, "data_sfc.nc")

if not os.path.exists(SFC_FILE):
    print("SFC File not found")
    exit()

ds = xr.open_dataset(SFC_FILE)
print("Keys:", ds.keys())
if 'pm10' in ds:
    vals = ds['pm10'].values
    print(f"PM10 Shape: {vals.shape}")
    print(f"PM10 Mean: {np.nanmean(vals)}")
    print(f"PM10 Max: {np.nanmax(vals)}")
    print(f"PM10 Min: {np.nanmin(vals)}")
    print(f"Non-zero count: {np.count_nonzero(vals)}")
    print(f"NaN count: {np.isnan(vals).sum()}")
else:
    print("pm10 variable not found!")
    
# Check timestamps
print("Time Start:", ds.valid_time.min().values)
print("Time End:", ds.valid_time.max().values)
