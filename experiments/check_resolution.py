
import xarray as xr
import numpy as np

ds = xr.open_dataset('../data/data_sfc.nc')

# Check Grid
lats = ds.latitude.values
lons = ds.longitude.values
lat_step = np.abs(np.diff(lats)[0])
lon_step = np.abs(np.diff(lons)[0])

print(f"Lat step (deg): {lat_step}")
print(f"Lon step (deg): {lon_step}")

# Estimate in km (approx at equator and mean lat)
# 1 deg lat ~ 111 km
# 1 deg lon ~ 111 * cos(lat) km
mean_lat = np.mean(lats)
km_lat = lat_step * 111.0
km_lon = lon_step * 111.0 * np.cos(np.deg2rad(mean_lat))

print(f"Resolution (Lat): ~{km_lat:.2f} km")
print(f"Resolution (Lon): ~{km_lon:.2f} km")
