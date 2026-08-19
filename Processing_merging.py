 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:35:55 2026

@author: timkluiters
"""

# Importing all libraries
import sys
sys.path.append(r'C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\code')
from recalculate_speed_handling_functions import speed_calculator
import pandas as pd
import numpy as np
from numpy import trapz
from sklearn.metrics import auc
import time
import os
from datetime import timedelta
from datetime import datetime
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from geopy.distance import geodesic
import gpxpy
import pytz
#from scipy.integrate import simpson
#import simplekml
#import pillow

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tilemapbase


from scipy.signal import savgol_filter

#from analysis_functions_indent import *

# Import the function from data_handling module
from helper_functions.data_handling import *


#%%

# Helper functions: 
def text_to_df_dict_Aeris(path):
    # List all .txt files into dataframes (stored in a directory)
    txt_files = [f for f in os.listdir(path) if f.endswith(".txt")
    if not f.endswith("spectra.txt") and not f.endswith("Eng.txt")and not f.endswith("spectralite.txt")]
    
    
    # Creare an empty dictionary to store DataFrames
    dfs = {}
    
    # Loop through each .txt file and read it into a DataFrames
    for i, file in enumerate(txt_files):
        # Construct the full file path
        file_path = os.path.join(path, file)
        
        # Read the .txt file into a DataFrame
        df = pd.read_csv(file_path, sep=',')
        
        # Use custom keys for the DataFrames
        key = "data{}".format(i+1)
        dfs[key] = df
    return dfs
'''
def read_gps_T26(path_data):
    df_phone1_1 = gpx_to_df(path_data)

    
    df_phone1_1.index = df_phone1_1.index.tz_convert(None)

    
    
    # concatenate different gps dfs into one df

    #gps_final = pd.concat([df_phone1_1, df_phone1_2,df_phone1_3,df_phone1_4,gps_phone2], axis=0) 
    gps_final = pd.concat(df_phone1_1,axis=0) 
    # Rename columns
    gps_final.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'speed': 'Speed [m/s]'}, inplace=True)

    return gps_final #,df_phone1_1,df_phone1_2,df_phone1_3,df_phone1_4,gps_phone2_1,gps_phone2_2,gps_phone2_3
'''


def read_and_preprocess_T26(path_data,inlet_delay,Aeris_delay, path_res,bg_quantile=None,writecsv=False,name='name'):
    
    if not bg_quantile:
        bg_quantile = 0.1 # default value for CH4 background

   
    # =============================================================================
    #   Aeris Mira Ultra 
    # =============================================================================
    
    dfs_aeris = text_to_df_dict_Aeris(path_data)
    
    # Merge all dataframes in the dictionary into a single dataframe
    df_aeris = pd.concat(dfs_aeris.values(), ignore_index=True)
    cols_aeris    = ['Time Stamp','CH4 (ppm)'] #'CO2_dry', 
    aeris = df_aeris[cols_aeris].copy()
    
    aeris.rename(columns={'Time Stamp':'Datetime', 'CH4 (ppm)':'CH4_aeris'},inplace=True)
    aeris = aeris.set_index('Datetime', drop = True)
    aeris.index = pd.to_datetime(aeris.index)
    
    #  --- Time correction ---------------------------
    aeris.index = (aeris.index - timedelta(seconds=inlet_delay)
                               + timedelta(seconds=Aeris_delay)) 
    aeris.index = aeris.index.round('1s')
    aeris = aeris.groupby(level=0).agg({'CH4_aeris': 'max',})
    aeris.sort_index(inplace=True)
    
    # --- Calculate CH4 elevation ------------------
    
    aeris['CH4_aeris']     = calibrate(aeris['CH4_aeris'], 'aer', 'CH4')
    aeris.sort_index(inplace=True)
    aeris['bg_aeris'] = aeris['CH4_aeris'].rolling('5min',center=True).quantile(bg_quantile)
    aeris['CH4_ele_aeris'] = aeris['CH4_aeris'] - aeris['bg_aeris']
    
    
    aeris['Datetime_UTC'] = aeris.index
    
    
    # =============================================================================
    #   Merge with gps 
    # =============================================================================
    
    gps_final = gpx_to_df(path_data)
    df_gps = pd.concat(gps_final.values(), ignore_index=False)
    df_gps.rename(columns={'latitude': 'Latitude', 'longitude': 'Longitude', 'speed': 'Speed_provided_m_sec'}, inplace=True)
    df_gps.index = df_gps.index.tz_convert(None)
    #cols_gps = ['latitude','longitude','elevation','time','speed']
    #gps = df_gps[cols_gps].copy()
   
    aeris_gps = aeris.copy()


    # Merging
    df_merged = pd.merge(aeris, df_gps, left_index=True, right_index=True, how='outer')
    df_merged.interpolate(method='linear', inplace=True)
    df_merged = speed_calculator(df_merged, lat_col='Latitude', lon_col='Longitude', speed_col='Speed_provided_m_sec')
    #aeris_gps.loc[:,['Longitude', 'Latitude', 'Speed [m/s]']] = df_merged.loc[:,['Longitude', 'Latitude','Speed [m/s]']]
    
    

    
    # =============================================================================
    #   Save to csv
    # =============================================================================

    # Print data into csv
    if writecsv:
        filename = f"{name}.csv"
        df_merged.to_csv(os.path.join(path_res , filename))
        
    return df_merged



#%%
#os.path.join(path_to_city, file_name)
path_to_raw_data = r"C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\raw data\bike"
path_res = r"C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\standardised data\Surveys\bike"
path_res_bis = r"C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\standardised data\Surveys\Rotterdam"
path_res_dif = r"C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\standardised data\Surveys\Amsterdam"
# Aeris_data_20022026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '02-20'), 11, -7246, path_res, writecsv=True,name='Utrecht260220')
# Aeris_data_04032026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '03-04'), 11, -7246, path_res, writecsv=True,name='Utrecht260304')
# Aeris_data_09032026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '03-09'), 11, -7246, path_res, writecsv=True, name='Utrecht260309')
# Aeris_data_23032026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '03-23'), 11, 36, path_res, writecsv=True, name='Utrecht260323')
# Aeris_data_24032026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '03-24'), 7, 36, path_res, writecsv=True, name='Utrecht260324')
# Aeris_data_26032026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '03-26'), 7, 36, path_res, writecsv=True, name='Utrecht260326')
# Aeris_data_02042026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '04-02'), 7, 36, path_res, writecsv=True, name='Utrecht260402')
# Aeris_data_10042026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '04-10'), 15, 36, path_res, writecsv=True, name='Utrecht260410')
# Aeris_data_14042026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '04-14'), 15, 36, path_res, writecsv=True, name='Utrecht260414')
# Aeris_data_17042026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '04-17'), 7, 36, path_res, writecsv=True, name='Utrecht260417')
# Aeris_data_17062026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '06-17'), 7, 0, path_res, writecsv=True, name='Utrecht260617')
# Aeris_data_18062026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '06-18'), 8, 0, path_res, writecsv=True, name='Utrecht260618')
# Aeris_data_23062026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '06-23'), 7, 0, path_res, writecsv=True, name='Utrecht260623')
# Aeris_data_29062026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '06-29'), 7, 0, path_res, writecsv=True, name='Utrecht260629')
# Aeris_data_29062026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '06-30'), 7, 0, path_res_bis, writecsv=True, name='Utrecht260630')
# Aeris_data_02072026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '07-02'), 7, 0, path_res_bis, writecsv=True, name='Utrecht260702')
# Aeris_data_06072026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '07-06'), 7, 0, path_res, writecsv=True, name='Utrecht260706')
# Aeris_data_09072026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '07-09'), 7, 0, path_res, writecsv=True, name='Utrecht260709')
# Aeris_data_13072026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '07-13'), 7, 0, path_res, writecsv=True, name='Utrecht260713')
# Aeris_data_16072026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '07-16'), 7, 0, path_res, writecsv=True, name='Utrecht260716')
Aeris_data_31072026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '07-31'), 7, 0, path_res_dif, writecsv=True, name='Utrecht260731')
Aeris_data_31072026 = read_and_preprocess_T26(os.path.join(path_to_raw_data, '08-11'), 7, 0, path_res_dif, writecsv=True, name='Utrecht260811')


#%%Zeist_Utrecht_04-03-2026
# Loading in Gpx files:

#gpx0903 = gpx_to_df(os.path.join(path_to_raw_data, 'Zeist_Utrecht_09-03-2026'))
#gpx0903con = pd.concat(gpx0903.values(), ignore_index=False)
#gpx_withspeed = speed_calculator(gpx0903con, lat_col='latitude', lon_col='longitude', speed_col='speed')
'''
gpx0403 = gpx_to_df(os.path.join(path_to_raw_data, 'Zeist_Utrecht_04-03-2026'))
gpx0403con = pd.concat(gpx0403.values(), ignore_index=False)
gpx0403con.index = gpx0403con.index.tz_convert(None)
#Loading in txt files from Aeris:
raw0309 = text_to_df_dict_Aeris(os.path.join(path_to_raw_data, 'Zeist_Utrecht_09-03-2026'))
raw0403 = text_to_df_dict_Aeris(os.path.join(path_to_raw_data, 'Zeist_Utrecht_09-03-2026')) 


'''






