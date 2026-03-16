
import xarray as xr
import os

file_path = 'data_sfc.nc'
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    try:
        ds = xr.open_dataset(file_path)
        print("Dataset Information:")
        print(ds)
        print("\nVariables:")
        for var in ds.data_vars:
            print(f"- {var}: {ds[var].attrs.get('long_name', 'No description')} ({ds[var].units if hasattr(ds[var], 'units') else 'No units'})")
    except Exception as e:
        print(f"Failed to open dataset: {e}")
