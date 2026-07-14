import os
import cdsapi


#years = ['2019', '2020', '2021', '2022', '2023']
months = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

#for year in years:
for year in range(1979, 2018):    #for month in months:
    year = str(year)
    year_dir = 'data/raw/' + year 
    if not os.path.isdir(year_dir):
        os.makedirs(year_dir)
    #file_name = year + '_' + month + '_sfc.nc'
    file_name = os.path.join(year_dir, year + '_sfc.nc')

    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'mean_sea_level_pressure', '10m_u_component_of_wind', 
                '10m_v_component_of_wind', '2m_temperature',
            ],
            'year': year,
            'month': months,
            'day': [
                '1', '2', '3', 
                '4', '5', '6', 
                '7', '8', '9', 
                '10', '11', '12', 
                '13', '14', '15', 
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',    
            ],
            'time': [
                '00:00', '06:00', '12:00',
                '18:00',
            ],
        },
        file_name)