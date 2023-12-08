#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 2023

@author: lpoulain
"""

import cdsapi
import numpy as np
import pickle

folder_path = '/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/ERA5/'
with open(folder_path+'valid_dates_1980_00_06_12_18.pkl','rb') as f:
    valid_dates = pickle.load(f)

# load client interface
client = cdsapi.Client()

data_origin = 'reanalysis-era5-single-levels'

datavars = ['10m_u_component_of_wind', 
            '10m_v_component_of_wind', 
            '100m_u_component_of_wind', 
            '100m_v_component_of_wind', 
            #'instantaneous_10m_wind_gust',
            
            #'2m_dewpoint_temperature',
            '2m_temperature', 
            
            "geopotential_at_surface",
            "land_sea_mask",
            
            #'convective_available_potential_energy', 
            #'convective_precipitation',
            #'convective_rain_rate',
            "total_precipitation_6hr",
            
            #'instantaneous_moisture_flux',
            #'instantaneous_surface_sensible_heat_flux', 
            #'mean_surface_direct_short_wave_radiation_flux', 
            #'mean_surface_latent_heat_flux',
            #'mean_surface_net_long_wave_radiation_flux', 
            #'mean_surface_sensible_heat_flux', 
            #'mean_top_downward_short_wave_radiation_flux',
            #'mean_top_net_long_wave_radiation_flux', 'mean_top_net_short_wave_radiation_flux',
            
            #'large_scale_rain_rate', 
            #'mean_large_scale_precipitation_rate',
            
            'mean_sea_level_pressure',  
            #'mean_total_precipitation_rate',
            #'mean_vertically_integrated_moisture_divergence', 
            #'sea_surface_temperature', 'surface_latent_heat_flux',
            'surface_pressure', 
            #'surface_sensible_heat_flux', 'total_column_cloud_ice_water',
            #'total_column_cloud_liquid_water', 'total_column_rain_water', 'total_column_supercooled_liquid_water',
            'total_column_water_vapour',
            #'vertical_integral_of_divergence_of_cloud_frozen_water_flux', 'vertical_integral_of_divergence_of_cloud_liquid_water_flux',
            #'vertical_integral_of_divergence_of_geopotential_flux', 'vertical_integral_of_divergence_of_kinetic_energy_flux', 'vertical_integral_of_divergence_of_mass_flux',
            #'vertical_integral_of_divergence_of_moisture_flux', 'vertical_integral_of_divergence_of_thermal_energy_flux', 'vertical_integral_of_divergence_of_total_energy_flux',
            #'vertical_integral_of_kinetic_energy', 'vertical_integral_of_potential_and_internal_energy', 'vertical_integral_of_potential_internal_and_latent_energy',
            #'vertical_integral_of_temperature', 'vertical_integral_of_thermal_energy', 'vertical_integral_of_total_energy',
            #'vertically_integrated_moisture_divergence',
           ]
       

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
                    'year': f'{year}',
                    'month': f'{month}',
                    'day': list(valid_dates[year][month]),
                    'time': times,
                }
            
            target_path = f'{folder_path}ERA5_{year}_{month}_{var}_surface.grib'
            client.retrieve(name=data_origin, request=data_params, target=target_path)