
import cdsapi
import os
import xarray as xr

c = cdsapi.Client()

full_file = 'era5_data.nc'
years = range(2003, 2025)
files = []

# 1. Download chunks
for year in years:
    fname = f'era5_{year}.nc'
    files.append(fname)
    
    if os.path.exists(fname):
        print(f"Skipping {fname}, already exists.")
        continue
    
    print(f"Downloading {fname}...")
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                ],
                'year': str(year),
                'month': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12',
                ],
                'day': [
                    '01', '02', '03', '04', '05', '06',
                    '07', '08', '09', '10', '11', '12',
                    '13', '14', '15', '16', '17', '18',
                    '19', '20', '21', '22', '23', '24',
                    '25', '26', '27', '28', '29', '30',
                    '31',
                ],
                'time': [
                    '00:00', '03:00', '06:00', '09:00',
                    '12:00', '15:00', '18:00', '21:00',
                ],
                'area': [
                    28.5, 47.0, 24.0, 55.25,
                ],
                'grid': [0.75, 0.75],
            },
            fname)
        print(f"Downloaded {fname}")
    except Exception as e:
        print(f"Failed to download {year}: {e}")

# 2. Merge/Concatenate
print("Merging files...")
try:
    ds_list = [xr.open_dataset(f) for f in files if os.path.exists(f)]
    if ds_list:
        combined = xr.concat(ds_list, dim='valid_time')
        combined.to_netcdf(full_file)
        print(f"Successfully created {full_file}")
        
        # Optional cleanup
        # for f in files:
        #    os.remove(f)
    else:
        print("No files to merge.")
except Exception as e:
    print(f"Merge failed: {e}")
