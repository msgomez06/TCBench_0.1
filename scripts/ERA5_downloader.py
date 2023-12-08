#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 2023

@author: lpoulain
"""

import cdsapi
import numpy as np
import pandas as pd
import pickle

folder_path = '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/'
with open(folder_path+'valid_dates_1980_00_06_12_18.pkl','rb') as f:
    valid_dates = pickle.load(f)

# load client interface
client = cdsapi.Client()


#Single Pressure
data_origin = 'reanalysis-era5-pressure-levels'

datavars = [ #'vorticity'
              'geopotential', 
              'relative_humidity',
              'temperature', 
              'u_component_of_wind', 
              'v_component_of_wind',
              'vertical_velocity',
              #'divergence', 
              'specific_humidity',
           ]
       
plevels = {'geopotential':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'relative_humidity':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'temperature':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'u_component_of_wind':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'v_component_of_wind':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'vertical_velocity':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'divergence':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ],
           'specific_humidity':[1000, 925, 850, 700, 600, 500, 
                           400, 300, 250, 200, 150, 100, 50
                           ]
          }

times = ['00:00', '06:00', 
         '12:00', '18:00',
        ]

for var in datavars:
    for year in list(valid_dates.keys()):
        for month in list(valid_dates[year].keys()):
            data_params = {
                    'product_type': 'reanalysis',
                    'format': 'grib',
                    'variable': var,
                    'pressure_level': plevels[var], 
                    'year': f'{year}',
                    'month': f'{month}',
                    'day': list(valid_dates[year][month]),
                    'time': times,
                }
            
            target_path = f'{folder_path}ERA5_{year}_{month}_{var}_upper.grib'
            client.retrieve(name=data_origin, request=data_params, target=target_path)