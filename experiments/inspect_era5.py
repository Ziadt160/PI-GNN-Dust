
import xarray as xr

try:
    ds = xr.open_dataset('era5_test.nc')
    print("Downloaded ERA5 Data:")
    print(ds)
    print("\nVariables:")
    for v in ds.data_vars:
        print(f"- {v}: {ds[v].shape}")
except Exception as e:
    print(f"Failed to open ERA5 test file: {e}")
