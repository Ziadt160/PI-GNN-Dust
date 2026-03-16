
import xarray as xr
import matplotlib.pyplot as plt
import os

# Load data
ds = xr.open_dataset('data_sfc.nc')
pm10 = ds['pm10']

# Stats
mean_val = pm10.mean().values
max_val = pm10.max().values
min_val = pm10.min().values

with open('exploration_summary.txt', 'w') as f:
    f.write("Data Exploration Summary\n")
    f.write("========================\n")
    f.write(f"Dataset: {ds}\n\n")
    f.write(f"Variable: pm10\n")
    f.write(f"Mean: {mean_val:.4e} {pm10.units}\n")
    f.write(f"Max:  {max_val:.4e} {pm10.units}\n")
    f.write(f"Min:  {min_val:.4e} {pm10.units}\n")

# Plot Time Series
pm10_mean_time = pm10.mean(dim=['latitude', 'longitude'])
plt.figure(figsize=(10, 6))
pm10_mean_time.plot()
plt.title('Spatially Averaged PM10 over Time')
plt.ylabel(f'PM10 ({pm10.units})')
plt.grid(True)
plt.savefig('pm10_timeseries.png')
print("Exploration complete. check exploration_summary.txt and pm10_timeseries.png")
