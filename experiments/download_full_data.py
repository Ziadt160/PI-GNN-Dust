
import cdsapi

c = cdsapi.Client()

# Define years from 2003 to 2024
years = [str(y) for y in range(2003, 2025)]

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': [
            '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
        ],
        'year': years,
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
    'era5_data.nc')
