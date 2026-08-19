# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 14:34:01 2024

@author: Judith
"""


import pandas as pd
import numpy as np
from scipy.integrate import trapezoid as trapz
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




#%% SETTINGS

# Let's start
time_start = datetime.utcnow()


release_loc1 = (5.1647191, 52.0874256) #? (maybe 52.08850892,5.16532755 and 52.08860602,5.16401192)
release_loc2 = (5.1644029, 52.0885635) #?

# find_peaks arguments:
dist            = 10        # minimal amount of points between peaks
W               = (1,30)    #


#%% Read-in Functions for CH4 analyzers

def read_AERIS_PICO_to_df(path_to_data):
    # Inclusion Sept. 30th 2025 - Process ethane data
    dictionary_PICO = text_to_df_dict(path_to_data)
    df_PICO = pd.concat(dictionary_PICO.values(), ignore_index=True)
    df_PICO['DATETIME_UTC'] = pd.to_datetime(df_PICO['Time Stamp'])
    df_PICO.set_index('DATETIME_UTC', inplace=True)
    
    df_PICO = df_PICO.sort_index(ascending=True)
    
    df_PICO = df_PICO[df_PICO.index.notnull()]
    
    df_PICO['bg_AERIS_PICO'] = df_PICO['CH4 (ppm)'].rolling('5min',center=True).quantile(0.10) 
    df_PICO['bg_C2H6_AERIS_PICO'] = df_PICO['C2H6 (ppb)'].rolling('5min', center=True).quantile(0.50)
    
    

    
    df_PICO['CH4_ele_AERIS_PICO'] = df_PICO['CH4 (ppm)'] - df_PICO['bg_AERIS_PICO']
    df_PICO['C2H6_ele_AERIS_PICO'] = df_PICO['C2H6 (ppb)'] - df_PICO['bg_C2H6_AERIS_PICO']
    df_PICO = df_PICO.rename(columns={'CH4 (ppm)': 'CH4_AERIS_PICO',
                                      'C2H6 (ppb)': 'C2H6_AERIS_PICO'})
    
    df_PICO = df_PICO[~df_PICO.index.duplicated(keep='first')]
    
    

    
    return df_PICO

def read_Fake_French_DSO(path_data_DSOs):
    
    pkl_files = [f for f in os.listdir(path_data_DSOs) if f.endswith(".pkl")]

    # Creare an empty dictionary to store DataFrames
    dfs = {}
    
    # Loop through each .txt file and read it into a DataFrames
    for i, file in enumerate(pkl_files):
        # Construct the full file path
        file_path = os.path.join(path_data_DSOs, file)
        
        # Read the .pkl file into a DataFrame
        minidf = pd.read_pickle(file_path)
        
        # Use custom keys for the DataFrames
        key = "data{}".format(i+1)
        dfs[key] = minidf

    df = pd.concat(dfs.values(), ignore_index=False)
    df = df.sort_index(ascending=True)
    # df['DATETIME_UTC'] = pd.to_datetime(df['DATETIME_UTC'])
    # df.set_index('DATETIME_UTC', inplace=True)
    
    df['bg_Picarros_ABB'] = df['CH4_cal_cor'].rolling('5min', center=True).quantile(0.10)
    df['CH4_ele_Picarros_ABB'] = df['CH4_cal_cor'] - df['bg_Picarros_ABB']
    df['bg_Picarros_ABB'] = df['bg_Picarros_ABB'] * 0.001
    df['CH4_ele_Picarros_ABB'] = df['CH4_ele_Picarros_ABB'] * 0.001
    df['CH4_cal_cor_ppm'] = df['CH4_cal_cor'] * 0.001
    
    df = df[~df.index.duplicated(keep='first')]
    
    return df


def read_UUI_CRE_Pickle(path_data_DSOs, analyzer_name):
    
    pkl_files = [f for f in os.listdir(path_data_DSOs) if f.endswith(f"{analyzer_name}.pkl")]

    # Creare an empty dictionary to store DataFrames
    dfs = {}
    
    # Loop through each .txt file and read it into a DataFrames
    for i, file in enumerate(pkl_files):
        # Construct the full file path
        file_path = os.path.join(path_data_DSOs, file)
        
        # Read the .pkl file into a DataFrame
        minidf = pd.read_pickle(file_path)
        
        # Use custom keys for the DataFrames
        key = "data{}".format(i+1)
        dfs[key] = minidf

    df = pd.concat(dfs.values(), ignore_index=False)
    df = df.sort_index(ascending=True)
    # df['DATETIME_UTC'] = pd.to_datetime(df['DATETIME_UTC'])
    # df.set_index('DATETIME_UTC', inplace=True)
    
    df[f'bg_{analyzer_name}'] = df['CH4 (ppm)'].rolling('5min', center=True).quantile(0.10)
    df[f'CH4_ele_{analyzer_name}'] = df[f'CH4 (ppm)'] - df[f'bg_{analyzer_name}']
    
    df = df[~df.index.duplicated(keep='first')]
    
    return df
    
#%% Utrecht
# =============================================================================

''' Il Lettore GPS deve leggere un file gpx da cartella '''
''' Questa funzione dovrà essere modificata come quella che legge i csv dei dati CH4 : Fatto! '''
def read_gps_U(path_data):
    #   Read GPS data (gpx file)
    
    dict_G2301 = gpx_to_df(path_data)
    gps_list = [single_df for single_df in dict_G2301.values()]
    gps = pd.concat(gps_list, axis=0)
    gps['time'] = pd.to_datetime(gps['time'])
    print('Ma il GPS ha una velocità???')
    print(gps)
    gps.rename(columns={'time': 'Datetime', 'latitude': 'Latitude', 'longitude': 'Longitude', 'speed': 'Speed [m/s]'}, inplace=True)
    gps.set_index('Datetime',inplace=True)
    
    # Proietto sulla timezone Europe/Amsterdam la mia coordinata temporale
    # Aggiunta 8/8/2024 - Localizzare come GMT e poi proiettare su Europe/Amsterdam
    if gps.index.tz is None:
        # Localizzare come UTC se tz-naive e poi convertire in Europe/Amsterdam
        gps.index = gps.index.tz_localize('UTC').tz_convert('Europe/Amsterdam')
    else:
        # Solo convertire se l'indice è già tz-aware
        gps.index = gps.index.tz_convert('Europe/Amsterdam')
    
    return gps # Il GPS è TZ-aware 

# =============================================================================
#   CH4 analyzer  
# =============================================================================

# --------------- G2301 & G4302  ---------------


''' Verificare quale specie è quella del CH4 '''
def read_G2301_and_MIRO(path_data):
    dfs_G2301 = dat_to_df_dict(path_data + r'\CH4\G2301')
    dfs_MIRO = text_to_df_dict(path_data + r'\CH4\MIRO AERIS')
    print("MIRO DATA -------------------------------------")
    print(dfs_MIRO)
    print("MIRO DATA -------------------------------------")
    
    # Merge all dataframes in the dictionary into a single dataframe
    df_G2301 = pd.concat(dfs_G2301.values(), ignore_index=True)
    #df_G4 = pd.concat(dfs_G4.values(), ignore_index=True)
    df_MIRO = pd.concat(dfs_MIRO.values(), ignore_index=True)
    
    cols_G2301    = ['DATE','TIME','CH4_dry','species','ALARM_STATUS'] #'CO2_dry', 
    #cols_G43    = ['DATE','TIME','CH4_dry','ALARM_STATUS'] #'C2H6_dry',
    cols_MIRO = ['Time Stamp', 'CH4 (ppm)']
    
    
    G2301 = df_G2301[cols_G2301].copy()
    #G4302 = df_G4[cols_G43].copy()
    MIRO = df_MIRO[cols_MIRO].copy()
    print('\n')
    print(MIRO)
    # Combine date and time into datetime
    #G4302.loc[:,'Datetime'] = pd.to_datetime(G4302['DATE'] + ' ' + G4302['TIME'])
    G2301.loc[:,'Datetime'] = pd.to_datetime(G2301['DATE'] + ' ' + G2301['TIME'])
    MIRO.loc[:, 'Time Stamp'] = pd.to_datetime(MIRO['Time Stamp'])
    
    #G4302 = G4302.drop(['DATE','TIME'],axis=1) #We already have it in 'Datetime'
    G2301 = G2301.drop(['DATE','TIME'],axis=1) #We already have it in 'Datetime'
    G2301 = G2301[G2301["species"]==1] 
    '''What is this species ?
    La specie è quella a cui si riferiscono i dati della riga: ci sono CH4, C2H6, e CO2 --> CH4 dovrebbe essere la specie 1
    '''
    
    G2301.set_index('Datetime',inplace = True,drop=False)
    #G4302.set_index('Datetime',inplace = True,drop=True)
    MIRO.set_index('Time Stamp', inplace = True, drop = False)
    
    
    #G4302['CH4_G43']     = calibrate(G4302['CH4_dry'], 'G43', 'CH4')
    G2301['CH4_G23']     = calibrate(G2301['CH4_dry'], 'G23', 'CH4')
    MIRO['CH4_MIRO'] = calibrate(MIRO['CH4 (ppm)'], 'aer', 'CH4')
    
    G2301['bg_G23']      = G2301['CH4_G23'].rolling('5min',center=True).quantile(0.10) 
    G2301['CH4_ele_G23'] = (G2301['CH4_G23'] - G2301['bg_G23'])
    
    MIRO['bg_MIRO'] = MIRO['CH4_MIRO'].rolling('5min', center=True).quantile(0.10)
    MIRO['CH4_ele_MIRO'] = (MIRO['CH4_MIRO'] - MIRO['bg_MIRO'])
    
    #G4302['bg_G43']      = G4302['CH4_G43'].rolling('5min',center=True).quantile(0.10) 
    #G4302['CH4_ele_G43'] = (G4302['CH4_G43'] - G4302['bg_G43'])
    
    G2301.rename(columns={'ALARM_STATUS':'ALARM_STATUS_G23'},inplace=True)
    #G4302.rename(columns={'ALARM_STATUS':'ALARM_STATUS_G43'},inplace=True)
    
    plt.figure()
    plt.title('AERUS original date')
    plt.plot(MIRO.index, MIRO['CH4_ele_MIRO'])
    
    """
    Porto le time corrections nella funzione successiva
    #  Time correction(s)
    
    # Bring MIRO 11.06 secondsd backwards to correct inlet delay
    delay_MIRO = (11.06 + 11.62)/2
    MIRO.index = (MIRO.index - timedelta(seconds=delay_MIRO) + timedelta(seconds=42))
    
    #Bring G2301 15.96 seconds backwards to correct inlet delay
    delay_G2301 = (15.96 + 17.04)/2
    G2301.index = (G2301.index - timedelta(seconds=delay_G2301))
    
    # Final correction
    
    #G2301.index = (G2301.index - timedelta(seconds=59)) # match with G4302 (from metadata excel file
    # which contains passing time of the measurement car it seems G4 is correct, but G2 shifted)
    """
    return G2301, MIRO # ,G4302 : No GasScouter available # G2301 e MIRO sono TZ-naive


def read_and_preprocess_G2301U_and_MIRO(path_data, path_res,writexlsx=False):
    
    # Il GPS è TZ-aware, G2301 e MIRO sono TZ-naive
    gps = read_gps_U(path_data + r'/GPS/Phone 1')
    G2301, MIRO = read_G2301_and_MIRO(path_data)
    
    # G2301 fornisce timestamp naive locali (CEST) : rendiamolo consapevole di ciò
    Utrecht_TZ = pytz.timezone('Europe/Amsterdam')
    
    
    G2301['Datetime'] = G2301['Datetime'].dt.tz_localize(Utrecht_TZ)

    G2301.set_index('Datetime', inplace=True, drop=True)
    
    # Compenso l'inlet delay di G2301
    delay_G2301 = (15.96 + 17.04)/2
    G2301.index = (G2301.index - timedelta(seconds=delay_G2301))
    
    G2301.index = G2301.index.tz_convert('Europe/Amsterdam')
    # print('::::::::::::::::::::::::::::::::::::::::::::::::::..')
    # print('E infine vediamo il GPS:')
    # print(gps.head())
    MIRO['Datetime'] = pd.to_datetime(MIRO['Time Stamp']).dt.tz_localize(Utrecht_TZ)
    MIRO.set_index('Datetime', inplace=True, drop=True)
    
    # Compenso l'inlet delay di Aeris Technologies
    delay_MIRO = (11.06 + 11.62)/2
    MIRO.index = (MIRO.index - timedelta(seconds=delay_MIRO) + timedelta(seconds=42))
    MIRO.index = MIRO.index.tz_convert('Europe/Amsterdam')
    
    # All'interno di merge_with_gps tutti i datetime diventano tz-naive
    
    # G4302_gps = merge_with_gps(G4302, gps) # in: data_handling.py in helper_functions GasScouter not available
    G2301_gps = merge_with_gps(G2301, gps) # in: data_handling.py in helper_functions
    MIRO_gps = merge_with_gps(MIRO, gps)
    
    # G2301_gps e MIRO_gps sono tz-naive
    # Assicurati che l'indice sia naive prima di scrivere su Excel
    #G2301_gps.index = G2301_gps.index.tz_convert(None)

    if writexlsx:
        writer  = pd.ExcelWriter(path_res + 'U_G23andMIRO.xlsx', engine = 'xlsxwriter')
        #G4302_gps.to_excel(writer, sheet_name='G4302')
        G2301_gps.to_excel(writer, sheet_name='G2301')
        MIRO_gps.to_excel(writer, sheet_name='MIRO')
        writer.book.close()
    
    return G2301_gps, MIRO_gps #, G4302_gps : GasCouter not avaialble



#%% Toronto

def merge_interpolate_left(df1,df2,col):
        if df1.index.name == col:
            combined = pd.merge(df1,df2,on=col,how='left')
            combined = combined.sort_values(by=col)
            combined = combined.interpolate(method='linear')
        else:
            print('failed')
            combined = pd.merge(df1,df2,on=col,how='left')
            combined = combined.sort_values(by=col)
            combined = combined.interpolate(method='linear')
        return combined

def calc_gps_T(df, create_gpskml=False):
    
    gps = df[['Latitude','Longitude']].copy(deep=True)
    
    gps['Speed [m/s]'] = gps.apply(lambda row: 0, axis=1)  # Initialize speed column with 0
    
    for i in range(1, len(gps)):
        # Calculate distance between consecutive rows
        prev_lat = gps.iloc[i-1, 0] # 0=Latitude
        prev_lon = gps.iloc[i-1, 1]  # 1=Longitude
        curr_lat = gps.iloc[i, 0]
        curr_lon = gps.iloc[i, 1]
        distance = geodesic((prev_lat, prev_lon), (curr_lat, curr_lon)).meters
    
        # Calculate speed using distance and time (assuming time between consecutive rows is constant)
        time_diff = gps.index[i] - gps.index[i-1]
        time_diff_seconds = time_diff.total_seconds()
        speed = distance / time_diff_seconds
    
        # Assign calculated speed to the 'speed' column
        gps.iloc[i, 2] = speed
    
    if create_gpskml: #TODO: DELTE???
        gps1 = gps
        gps1 = gps1.set_index(gps.Datetime)
        gpslist = gps1.values.tolist() # prepare to create kml file
        # gpslist = gpslist[~np.isnan(gpslist)]
        gpskml(gpslist)               # function to create kml file
    
    return gps

def calc_gps_T2(df, create_gpskml=False):
    
    #gps = df[['Latitude','Longitude']].copy(deep=True)
    
    df['Speed [m/s]'] = df.apply(lambda row: 0, axis=1)  # Initialize speed column with 0
    
    for i in range(1, len(df)):
        # Calculate distance between consecutive rows
        prev_lat = df.iloc[i-1, 0] # 0=Latitude
        prev_lon = df.iloc[i-1, 1]  # 1=Longitude
        curr_lat = df.iloc[i, 0]
        curr_lon = df.iloc[i, 1]
        distance = geodesic((prev_lat, prev_lon), (curr_lat, curr_lon)).meters
    
        # Calculate speed using distance and time (assuming time between consecutive rows is constant)
        time_diff = df.index[i] - df.index[i-1]
        time_diff_seconds = time_diff.total_seconds()
        speed = distance / time_diff_seconds
    
        # Assign calculated speed to the 'speed' column
        df.loc['Speed [m/s]'].iloc[i] = speed
    
    if create_gpskml: #TODO: DELTE???
        gps1 = df
        gps1 = gps1.set_index(gps.Datetime)
        gpslist = gps1.values.tolist() # prepare to create kml file
        # gpslist = gpslist[~np.isnan(gpslist)]
        gpskml(gpslist)               # function to create kml file
    
    return df


def read_and_preprocess_BikeandCar_T(path_dataT, path_res,writexlsx=False, create_gpskml=False):
    
    T_1bike = pd.read_csv(path_dataT + 'sync_data_bike_2021-10-20.csv', index_col='gps_time', parse_dates=['gps_time'])
    T_1c = pd.read_csv(path_dataT + 'sync_data_eccc_car_2021-10-20.csv', index_col='gps_time', parse_dates=['gps_time'])
    T_2c = pd.read_csv(path_dataT + 'sync_data_eccc_car_2021-10-24.csv', index_col='gps_time', parse_dates=['gps_time'])

    
    # =============================================================================
    #       20.10.21 - Bike
    # =============================================================================
    
    # Release Time
    r1_start = datetime(2021,10,20,20,13)
    r4_finish = datetime(2021,10,20,20,49)
    
    starttime = r1_start - timedelta(minutes=3)
    endtime = r4_finish + timedelta(minutes=3)
    
    T_1bike.index.names = ['Datetime']
    T_1bike.rename(columns={'ch4': 'CH4_LGR','lat': 'Latitude', 'lon':'Longitude', 'heading':'Heading'}, inplace=True)
    T_1bike = T_1bike[:-2]
    T_1bike = T_1bike.dropna(subset=['Latitude'])
    
    T_1bike = T_1bike.loc[:, ['CH4_LGR', 'Latitude', 'Longitude', 'Heading','wd_corr','ws_corr']]

    
    # --- GPS ----------------------------
    
    gps = calc_gps_T(T_1bike, create_gpskml) 
    
    # --- Time correction (inlet delay) ----------------------------
    
    T_1bike = T_1bike.drop(['Latitude', 'Longitude'], axis=1)
    T_1bike.index = (T_1bike.index
                      - timedelta(seconds=30)) # account for inlet delay
    # merge gps with time shifted data (since gps is not affected by inlet no shift needed
    # but since Datetime index was shifted, merging is needed. Interpolation is
    # necessary since the index was not shifted by number of observations (e.g. 30 obs.),
    # but by a time (30s) which can lead to the case that there is no datetime index
    # anymore which fits to the index of the gps)
    T_1bike = merge_interpolate_left(T_1bike, gps,'Datetime')
    
    # --- Calculate CH4 elevation -------------------------------
    
    T_1bike['bg_LGR']      = T_1bike['CH4_LGR'].rolling('5min',center=True).quantile(0.10) 
    T_1bike['CH4_ele_LGR'] = (T_1bike['CH4_LGR'] - T_1bike['bg_LGR'])
    
    # ---  Save to CSV -------------------------------
    
    if writexlsx:
        T_1bike.to_csv(path_res + 'T_1bike_LGR.csv', index=True) # way faster than excel
    
    LGR_1b = T_1bike[starttime:endtime]
    
    
    # =============================================================================
    #       20.10.21 - Car
    # =============================================================================
    
    r1_start = datetime(2021,10,20,20,11)
    r4_finish = datetime(2021,10,20,20,49)

    starttime = r1_start - timedelta(minutes=3)
    endtime = r4_finish + timedelta(minutes=3)

    G2401_1c = T_1c[starttime:endtime].copy(deep=True)

    G2401_1c.index.names = ['Datetime']
    G2401_1c.rename(columns={'ch4': 'CH4_G24','lat': 'Latitude', 'lon':'Longitude', 'heading':'Heading'}, inplace=True)

    # Filter the DataFrame to keep only the first occurrence of each number (remove duplicates)
    G2401_1c['FirstOccurrence'] = (G2401_1c['CH4_G24'] != G2401_1c['CH4_G24'].shift(1)).cumsum()
    G2401_1c = G2401_1c.drop_duplicates(subset = ['FirstOccurrence'], keep='first')

    G2401_1c = G2401_1c.loc[:, ['CH4_G24', 'Latitude', 'Longitude', 'Heading','wd_corr','ws_corr']]


    # fig,ax = plt.subplots()
    # plt.plot(T_1c.index,T_1c['CH4'])
    #T_1c = T_1c[:-2]

    # --- GPS ----------------------------
    
    gps = calc_gps_T(G2401_1c, create_gpskml) # in postprocessing.U_readin_data 
        
    # --- Time correction (inlet delay) ----------------------------
    
    G2401_1c = G2401_1c.drop(['Latitude', 'Longitude'], axis=1)
    # G2301_1c.index = (G2301_1c.index
    #                   - timedelta(seconds=30)) # account for inlet delay
    # # merge gps with time shifted data (since gps is not affected by inlet no shift needed
    # # but since Datetime index was shifted, merging is needed. Interpolation is
    # # necessary since the index was not shifted by number of observations (e.g. 30 obs.),
    # # but by a time (30s) which can lead to the case that there is no datetime index
    # # anymore which fits to the index of the gps)
    # G2301_1c = merge_interpolate_left(G2301_1c, gps, col='Datetime')

    G2401_1c = pd.concat([G2401_1c, gps],axis=1, ignore_index=False)
    G2401_1c.index = pd.to_datetime(G2401_1c.index)
    
    #G2401_1c = merge_with_gps(G2401_1c, gps)  # ???

    # --- Calculate CH4 elevation -------------------------------
    
    G2401_1c['bg_G24']      = G2401_1c['CH4_G24'].rolling('5min',center=True).quantile(0.10) 
    G2401_1c['CH4_ele_G24'] = (G2401_1c['CH4_G24'] - G2401_1c['bg_G24'])
    
    # ---  Save to CSV -------------------------------

    if writexlsx:
        G2401_1c.to_csv(path_res + 'T_1car_G24.csv', index=True) # way faster than excel

    
    # =============================================================================
    #       24.10.21 - Car
    # =============================================================================

    # Release Time
    r1_start = datetime(2021,10,24,13,47)
    r4_finish = datetime(2021,10,24,14,37)

    starttime = r1_start - timedelta(minutes=3)
    endtime = r4_finish + timedelta(minutes=3)

    G2401_2c = T_2c[starttime:endtime].copy(deep=True)

    G2401_2c.index.names = ['Datetime']
    G2401_2c.rename(columns={'ch4': 'CH4_G24','lat': 'Latitude', 'lon':'Longitude', 'heading':'Heading'}, inplace=True)

    # Filter the DataFrame to keep only the first occurrence of each number (remove duplicates)
    G2401_2c['FirstOccurrence'] = (G2401_2c['CH4_G24'] != G2401_2c['CH4_G24'].shift(1)).cumsum()
    G2401_2c = G2401_2c.drop_duplicates(subset = ['FirstOccurrence'], keep='first')

    G2401_2c = G2401_2c.loc[:, ['CH4_G24', 'Latitude', 'Longitude', 'Heading','wd_corr','ws_corr']]

    # --- GPS ----------------------------
    
    gps = calc_gps_T(G2401_2c, create_gpskml) # in postprocessing.U_readin_data 
    print(gps)
    # --- Time correction (inlet delay) ----------------------------
    
    G2401_2c = G2401_2c.drop(['Latitude', 'Longitude'], axis=1) 
    G2401_2c = pd.concat([G2401_2c, gps],axis=1, ignore_index=False)
    G2401_2c.index = pd.to_datetime(G2401_2c.index) # ???
    #G2401_2c = merge_with_gps(G2401_2c, gps)

    # --- Calculate CH4 elevation -------------------------------
    
    G2401_2c['bg_G24']      = G2401_2c['CH4_G24'].rolling('5min',center=True).quantile(0.10) 
    G2401_2c['CH4_ele_G24'] = (G2401_2c['CH4_G24'] - G2401_2c['bg_G24'])
    
    # ---  Save to CSV -------------------------------

    if writexlsx:
        G2401_2c.to_csv(path_res + 'T_2car_G24.csv', index=True) # way faster than excel


    
    
    return LGR_1b, G2401_1c, G2401_2c




#%% London

def read_and_preprocess_L(path_data, path_res,writexlsx=False, create_gpskml=False):

    
    # =============================================================================
    #       Day 2
    # =============================================================================
    
    datapath   = path_data + 'Day2/'
    
    df_prox  = []
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            df_prox_release_x = pd.read_csv(datapath + file, sep=",") #, usecols= cols_G23
            df_prox.append(df_prox_release_x)


    df_all        = pd.concat(df_prox).reset_index(drop=True)

    # Combine date and time into datetime
    df_all['Datetime'] = pd.to_datetime(df_all['DATE'] + ' ' + df_all['TIME_cor'], format='%d/%m/%Y %H:%M:%S')
    df_all.set_index('Datetime',inplace = True,drop=True)
    
    # Drop unnecessary columns and rows with Nans
    df_all = df_all.loc[:,['ALARM_STATUS','species','CH4_cal','GPS_ABS_LAT','GPS_ABS_LONG','Heading','Speed_ms-1',
                        'LGRdata','Run']]
    df_all = df_all.drop(pd.NaT)   
    #G2301 = G2301[G2301["species"]==3] # 3 is CH4
    
    # Rename Columns
    df_all.rename(columns={'CH4_cal':'CH4_G23','LGRdata':'CH4_LGR','GPS_ABS_LONG':'Longitude','GPS_ABS_LAT':'Latitude','Speed_ms-1':'Speed [m/s]'},inplace=True)
    #df_all.rename(columns={'LGRdata':'CH4_LGR'},inplace=True)

    
    split_index = df_all.columns.get_loc('CH4_LGR')  # LGR data start

    G2301_d2 = df_all.iloc[:, :split_index].copy(deep=True)  # Split data into two sets, one for G2301, one for LGR measurememnts
    LGR_d2 = df_all.iloc[:, split_index:].copy(deep=True)
    
    
    LGR_d2['Longitude'] = G2301_d2['Longitude']
    LGR_d2['Latitude'] = G2301_d2['Latitude']
    LGR_d2['Speed [m/s]'] = G2301_d2['Speed [m/s]']

    # Prevent having several measurements with same time stamp:
    G2301_d2 = G2301_d2[G2301_d2["species"]==3] 
    # Resetting the index to convert it to a regular column
    LGR_d2.reset_index(inplace=True)
    # Drop duplicate rows based on the index (time in this case)
    LGR_d2.drop_duplicates(subset='Datetime', keep='first', inplace=True)
    # If you want to set the index back to the original index column
    LGR_d2.set_index('Datetime', inplace=True)

    # --- Calculate CH4 elevation -------------------------------
    
    #G2301_d2.loc[:, 'CH4_bg05'] = G2301_d2.loc[:, 'CH4_G2'].rolling('5min', center=True).quantile(0.05)
    G2301_d2.loc[:, 'bg_G23'] = G2301_d2.loc[:, 'CH4_G23'].rolling('5min', center=True).quantile(0.10)
    G2301_d2.loc[:, 'CH4_ele_G23'] = G2301_d2.loc[:, 'CH4_G23']-G2301_d2.loc[:, 'bg_G23']
    
    #LGR_d2.loc[:, 'CH4_bg05'] = LGR_d2.loc[:, 'CH4_LGR'].rolling('5min', center=True).quantile(0.05)
    LGR_d2.loc[:, 'bg_LGR'] = LGR_d2.loc[:, 'CH4_LGR'].rolling('5min', center=True).quantile(0.10)
    LGR_d2.loc[:, 'CH4_ele_LGR'] = LGR_d2.loc[:, 'CH4_LGR']-LGR_d2.loc[:, 'bg_LGR']

    
    # --- Time correction (inlet delay) ----------------------------

    #print('     Performing time correction')
    # already done by using corrected time in datasheet. It was corrected for
    # Picarro and using the same corrected time aligns (more r less) also the 
    # LGR data
    LGR_d2.index = (LGR_d2.index - timedelta(seconds=2))
    

    # =============================================================================
    #       Day 3
    # =============================================================================

    datapath   = path_data + 'Day3/'

    df_prox  = []
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            df_prox_release_x = pd.read_csv(datapath + file, sep=",") #, usecols= cols_G23
            df_prox.append(df_prox_release_x)


    df_all        = pd.concat(df_prox).reset_index(drop=True)

    # Combine date and time into datetime
    df_all['Datetime'] = pd.to_datetime(df_all['DATE'] + ' ' + df_all['TIME_cor'], format='%d/%m/%Y %H:%M:%S')
    df_all.set_index('Datetime',inplace = True,drop=True)

    # Drop unnecessary columns and rows with Nans
    df_all = df_all.loc[:,['ALARM_STATUS','species','CH4_cal','GPS_ABS_LAT','GPS_ABS_LONG','Heading','Speed_ms-1',
                         'Run','LGRTime','LGRdata','LGRrun#','LiCOR_ppm','Run#']]
    df_all = df_all.drop(pd.NaT)   
    #G2301 = G2301[G2301["species"]==3] # 3 is CH4

    # Rename Columns
    df_all.rename(columns={'CH4_cal':'CH4_G23','LGRdata':'CH4_LGR','LiCOR_ppm':'CH4_Licor','GPS_ABS_LONG':'Longitude','GPS_ABS_LAT':'Latitude',
                           'Speed_ms-1':'Speed [m/s]','Run':'Run_G23','Run#':'Run_Licor','LGRrun#':'Run_LGR'},inplace=True)
    #df_all.rename(columns={'LGRdata':'CH4_LGR'},inplace=True)


    G2301_d3 = df_all.copy(deep=True)
    G2301_d3.drop(columns=['CH4_LGR','CH4_Licor','Run_Licor','Run_LGR','LGRTime'],inplace=True)

    LGR_d3 = df_all.copy(deep=True)
    LGR_d3.drop(columns=['species','CH4_G23','CH4_Licor','Run_Licor','Run_G23'],inplace=True)

    Licor_d3 = df_all.copy(deep=True)
    Licor_d3.drop(columns=['species','CH4_LGR','CH4_G23','Run_G23','Run_LGR','LGRTime'],inplace=True)


    # Prevent having several measurements with same time stamp:
    G2301_d3 = G2301_d3[G2301_d3["species"]==3] 
    # Resetting the index to convert it to a regular column
    LGR_d3.reset_index(inplace=True)
    Licor_d3.reset_index(inplace=True)
    # Drop duplicate rows based on the index (time in this case)
    LGR_d3.drop_duplicates(subset='Datetime', keep='first', inplace=True)
    Licor_d3.drop_duplicates(subset='Datetime', keep='first', inplace=True)
    # If you want to set the index back to the original index column
    LGR_d3.set_index('Datetime', inplace=True)
    Licor_d3.set_index('Datetime', inplace=True)

    # --- Calculate CH4 elevation -------------------------------

    #G2301_d3.loc[:, 'CH4_bg05'] = G2301_d3.loc[:, 'CH4_G2'].rolling('5min', center=True).quantile(0.05)
    G2301_d3.loc[:, 'bg_G23'] = G2301_d3.loc[:, 'CH4_G23'].rolling('5min', center=True).quantile(0.10)
    G2301_d3.loc[:, 'CH4_ele_G23'] = G2301_d3.loc[:, 'CH4_G23']-G2301_d3.loc[:, 'bg_G23']

    #LGR_d3.loc[:, 'CH4_bg05'] = LGR_d3.loc[:, 'CH4_LGR'].rolling('5min', center=True).quantile(0.05)
    LGR_d3.loc[:, 'bg_LGR'] = LGR_d3.loc[:, 'CH4_LGR'].rolling('5min', center=True).quantile(0.10)
    LGR_d3.loc[:, 'CH4_ele_LGR'] = LGR_d3.loc[:, 'CH4_LGR']-LGR_d3.loc[:, 'bg_LGR']

    Licor_d3.loc[:, 'bg_Licor'] = Licor_d3.loc[:, 'CH4_Licor'].rolling('5min', center=True).quantile(0.10)
    Licor_d3.loc[:, 'CH4_ele_Licor'] = Licor_d3.loc[:, 'CH4_Licor']-Licor_d3.loc[:, 'bg_Licor']

    # --- Time correction (inlet delay) ----------------------------

    #print('     Performing time correction')
    # already done by using corrected time in datasheet. It was corrected for
    # Picarro and using the same corrected time aligns (more r less) also the 
    # LGR data
    #LGR_d3.index = (LGR_d3.index - timedelta(seconds=25)) ????
    
    
    # =============================================================================
    #       Day 4
    # =============================================================================

    datapath   = path_data + 'Day4/'

    df_prox  = []
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            df_prox_release_x = pd.read_csv(datapath + file, sep=",") #, usecols= cols_G23
            df_prox.append(df_prox_release_x)


    df_all        = pd.concat(df_prox).reset_index(drop=True)

    # Combine date and time into datetime
    df_all['Datetime'] = pd.to_datetime(df_all['DATE'] + ' ' + df_all['TIME_cor'], format='%d/%m/%Y %H:%M:%S')
    df_all.set_index('Datetime',inplace = True,drop=True)

    # Drop unnecessary columns and rows with Nans
    df_all = df_all.loc[:,['ALARM_STATUS','species','CH4_cal','GPS_ABS_LAT','GPS_ABS_LONG','Heading','Speed_ms-1',
                         'Run','LGRTime','LGRdata','LGRrun#','LiCOR_ppm','Run#']]
    df_all = df_all.drop(pd.NaT)   
    #G2301 = G2301[G2301["species"]==3] # 3 is CH4

    # Rename Columns
    df_all.rename(columns={'CH4_cal':'CH4_G23','LGRdata':'CH4_LGR','LiCOR_ppm':'CH4_Licor','GPS_ABS_LONG':'Longitude','GPS_ABS_LAT':'Latitude',
                           'Speed_ms-1':'Speed [m/s]','Run':'Run_G23','Run#':'Run_Licor','LGRrun#':'Run_LGR'},inplace=True)
    #df_all.rename(columns={'LGRdata':'CH4_LGR'},inplace=True)


    G2301_d4 = df_all.copy(deep=True)
    G2301_d4.drop(columns=['CH4_LGR','CH4_Licor','Run_Licor','Run_LGR','LGRTime'],inplace=True)

    LGR_d4 = df_all.copy(deep=True)
    LGR_d4.drop(columns=['species','CH4_G23','CH4_Licor','Run_Licor','Run_G23'],inplace=True)

    Licor_d4 = df_all.copy(deep=True)
    Licor_d4.drop(columns=['species','CH4_LGR','CH4_G23','Run_G23','Run_LGR','LGRTime'],inplace=True)


    # Prevent having several measurements with same time stamp:
    G2301_d4 = G2301_d4[G2301_d4["species"]==3] 
    # Resetting the index to convert it to a regular column
    LGR_d4.reset_index(inplace=True)
    Licor_d4.reset_index(inplace=True)
    # Drop duplicate rows based on the index (time in this case)
    LGR_d4.drop_duplicates(subset='Datetime', keep='first', inplace=True)
    Licor_d4.drop_duplicates(subset='Datetime', keep='first', inplace=True)
    # If you want to set the index back to the original index column
    LGR_d4.set_index('Datetime', inplace=True)
    Licor_d4.set_index('Datetime', inplace=True)

    # --- Calculate CH4 elevation -------------------------------

    #G2301_d4.loc[:, 'CH4_bg05'] = G2301_d4.loc[:, 'CH4_G2'].rolling('5min', center=True).quantile(0.05)
    G2301_d4.loc[:, 'bg_G23'] = G2301_d4.loc[:, 'CH4_G23'].rolling('5min', center=True).quantile(0.10)
    G2301_d4.loc[:, 'CH4_ele_G23'] = G2301_d4.loc[:, 'CH4_G23']-G2301_d4.loc[:, 'bg_G23']

    #LGR_d4.loc[:, 'CH4_bg05'] = LGR_d4.loc[:, 'CH4_LGR'].rolling('5min', center=True).quantile(0.05)
    LGR_d4.loc[:, 'bg_LGR'] = LGR_d4.loc[:, 'CH4_LGR'].rolling('5min', center=True).quantile(0.10)
    LGR_d4.loc[:, 'CH4_ele_LGR'] = LGR_d4.loc[:, 'CH4_LGR']-LGR_d4.loc[:, 'bg_LGR']

    Licor_d4.loc[:, 'bg_Licor'] = Licor_d4.loc[:, 'CH4_Licor'].rolling('5min', center=True).quantile(0.10)
    Licor_d4.loc[:, 'CH4_ele_Licor'] = Licor_d4.loc[:, 'CH4_Licor']-Licor_d4.loc[:, 'bg_Licor']

    # --- Time correction (inlet delay) ----------------------------

    #print('     Performing time correction')
    # already done by using corrected time in datasheet. It was corrected for
    # Picarro and using the same corrected time aligns (more r less) also the 
    # LGR data
    #LGR_d4.index = (LGR_d4.index - timedelta(seconds=25)) ?????
    
    
    # =============================================================================
    #       Day 5
    # =============================================================================
    
    
    datapath   = path_data + 'Day5/'
    
    df_prox  = []
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            df_prox_release_x = pd.read_csv(datapath + file, sep=",") 
            df_prox.append(df_prox_release_x)


    df_all        = pd.concat(df_prox).reset_index(drop=True)

    # Combine date and time into datetime
    df_all['Datetime'] = pd.to_datetime(df_all['DATE'] + ' ' + df_all['TIME_cor'], format='%d/%m/%Y %H:%M:%S')
    df_all.set_index('Datetime',inplace = True,drop=True)
    
    # Drop unnecessary columns and rows with Nans 
    G2301_d5 = df_all.loc[:,['ALARM_STATUS','species','CH4_cal','GPS_ABS_LAT','GPS_ABS_LONG', 'Heading', 
                           'Distance','Speed_ms-1','Run','Inlet height Height']]
    G2301_d5 = G2301_d5.drop(pd.NaT)
    
    # Rename Columns
    G2301_d5.rename(columns={'CH4_cal':'CH4_G23','GPS_ABS_LONG':'Longitude','GPS_ABS_LAT':'Latitude',
                             'Speed_ms-1':'Speed [m/s]'},inplace=True)
    
    # Prevent having several measurements with same time stamp:
    G2301_d5 = G2301_d5[G2301_d5["species"]==3] 
    
    # --- Calculate CH4 elevation -------------------------------
    #G2301_d5.loc[:, 'CH4_bg05'] = G2301_d5.loc[:, 'CH4_G2'].rolling('5min', center=True).quantile(0.05)
    G2301_d5.loc[:, 'bg_G23'] = G2301_d5.loc[:, 'CH4_G23'].rolling('5min', center=True).quantile(0.10)
    G2301_d5.loc[:, 'CH4_ele_G23'] = G2301_d5.loc[:, 'CH4_G23']-G2301_d5.loc[:, 'bg_G23']
    
    
    
    
    # =============================================================================
    #       Save to CSV
    # =============================================================================
    if writexlsx:
        writer  = pd.ExcelWriter(path_res + 'L_G23andLGRandLicor.xlsx', engine = 'xlsxwriter')
        G2301_d2.to_excel(writer, sheet_name='D2_G2301')
        LGR_d2.to_excel(writer, sheet_name='D2_LGR')
        G2301_d3.to_excel(writer, sheet_name='D3_G2301')
        Licor_d3.to_excel(writer, sheet_name='D3_Licor')
        G2301_d4.to_excel(writer, sheet_name='D4_G2301')
        Licor_d4.to_excel(writer, sheet_name='D4_Licor')
        G2301_d5.to_excel(writer, sheet_name='D5_G2301')
        writer.book.close()
    
    
    return LGR_d2, G2301_d2, LGR_d3, G2301_d3, Licor_d3, LGR_d4, G2301_d4, Licor_d4, G2301_d5




#%% Rotterdam


def read_and_preprocess_R(path_data, path_res,writexlsx=False):
    
    starttime       = pd.to_datetime('2022-09-06 06:50:00')
    endtime         = pd.to_datetime('2022-09-06 12:59:00')

    morning_start   = pd.to_datetime('2022-09-06 07:05:00')
    morning_end     = pd.to_datetime('2022-09-06 10:44:00')

    afternoon_start = pd.to_datetime("2022-09-06 11:05:00")
    afternoon_end = pd.to_datetime('2022-09-06 12:26:00')
    
    # All directory paths

    G2301path   = path_data + 'Car/G2301/'
    G4302path   = path_data + 'G4302/'
    aerispath   = path_data + 'Car/Aeris/'
    gpspath     = path_data + 'Car/'
    TNOPATH     = path_data + 'TNO/'
    
    
    
    
    # =============================================================================
    #       G2301 & G4302 - UU (&TNO) car
    # =============================================================================
    
    # G4302 gets transfered in the afternoon from the UU car to the TNO car
        
    cols_G23    = ['DATE','TIME','CH4_dry_sync',
                   'species','ALARM_STATUS'] #'CO2_sync','CH4_sync','H2O_sync'
    cols_G43    = ['DATE','TIME','CH4_dry','ALARM_STATUS']

    G2301_prox  = []
    for file in os.listdir(G2301path):
        if file.endswith(".dat"):
            df_prox = pd.read_csv(G2301path + file, sep="\s+") #, usecols= cols_G23
            G2301_prox.append(df_prox)

    G4302_prox  = []
    for file in os.listdir(G4302path):
        if file.endswith(".dat"):
            df_prox = pd.read_csv(G4302path+file, sep="\s+", usecols= cols_G43)
            G4302_prox.append(df_prox)

    G2301        = pd.concat(G2301_prox).reset_index(drop=True)
    G4302        = pd.concat(G4302_prox, ignore_index=True, sort=False)

    # Combine date and time into datetime
    G4302['Datetime'] = pd.to_datetime(G4302['DATE'] + ' ' + G4302['TIME'])
    G2301['Datetime'] = pd.to_datetime(G2301['DATE'] + ' ' + G2301['TIME'])

    G4302 = G4302.drop(['DATE','TIME'],axis=1)
    G2301 = G2301.drop(['DATE','TIME'],axis=1)
    G2301 = G2301[G2301["species"]==1]
    
    G2301.rename(columns={'CH4_dry_sync':'CH4_dry'},inplace=True)
    G2301.rename(columns={'ALARM_STATUS':'ALARM_STATUS_G23'},inplace=True)
    G4302.rename(columns={'ALARM_STATUS':'ALARM_STATUS_G43'},inplace=True)

    
    G4302['CH4_G43']     = calibrate(G4302['CH4_dry'], 'G43', 'CH4')
    G2301['CH4_G23']     = calibrate(G2301['CH4_dry'], 'G23', 'CH4')
    
    G2301.set_index('Datetime',inplace = True,drop=True)
    G4302.set_index('Datetime',inplace = True,drop=True)

    
    # --- Calculate CH4 elevation -------------------------------
    
    G2301['bg_G23'] = G2301['CH4_G23'].rolling('5min',center=True).quantile(0.10)
    G2301['CH4_ele_G23'] = G2301['CH4_G23'] - G2301['bg_G23']
    
    G4302['bg_G43'] = G4302['CH4_G43'].rolling('5min',center=True).quantile(0.10)
    G4302['CH4_ele_G43'] = G4302['CH4_G43'] - G4302['bg_G43']
    
    # --- Time correction (inlet delay) ----------------------------
    
    G4302.index = (G4302.index
                      - timedelta(seconds=16.9)) # match with TNO
    G2301.index = (G2301.index
                      + timedelta(seconds=44)       # clock correction
                      - timedelta(seconds=2)        # inlet delay
                      - timedelta(seconds=5.29))    # matching peaks with G43
    
    
    
    # =============================================================================
    #       Aeris - UU (&TNO) car
    # =============================================================================
    
    # Aeris gets transfered in the afternoon from the UU car to the TNO car
    
    
    aerfile = 'Pico100221_220906_072814.txt'
    aeris = pd.read_csv(aerispath + aerfile, sep=',').add_prefix('aer_')
    aeris.rename(columns={'aer_Time Stamp':'Datetime', 'aer_CH4 (ppm)':'CH4_aeris'},inplace=True)
    aeris = aeris.set_index('Datetime', drop = True)
    aeris.index = pd.to_datetime(aeris.index)

    aeris['CH4_aeris']     = calibrate(aeris['CH4_aeris'], 'aer', 'CH4')
    
    plt.plot(aeris.index, aeris.values)


    # --- Calculate CH4 elevation -------------------------------
    
    aeris['bg_aeris'] = aeris['CH4_aeris'].rolling('5min',center=True).quantile(0.10)
    aeris['CH4_ele_aeris'] = aeris['CH4_aeris'] - aeris['bg_aeris']
    
     
    # --- Time correction (inlet delay) ----------------------------
    
    aeris.index = (aeris.index
                      - timedelta(hours=1, minutes=1, seconds=8) # clock correction
                      - timedelta(seconds=2) # inlet delay
                      - timedelta(seconds=1.68)) # matching peaks with G43


    # =============================================================================
    #       GPS - UU car
    # =============================================================================
    
    
    gps = pd.read_excel(gpspath + '2022-09-06_GPS.xlsx',
                            header = 0,
                            names=['Datetime', 'Latitude', 'Longitude', 'speed']) #usecols = [0,1,2,3]

    gps.Datetime = pd.to_datetime(gps.Datetime)
    gps.set_index('Datetime',inplace=True)
    gps = gps.loc['2022-09-06 07:00:02':]
    gps.loc[:,'speed'] = gps.loc[:,'speed']/3.6
    gps.rename(columns={'speed':'Speed [m/s]'},inplace=True)
    
    # G2301_gps = merge_interpolate(G2301, gps, col='Datetime') # I used this for MA
    # G4302_gps = merge_interpolate(G4302, gps, col='Datetime')
    # aeris_gps = merge_interpolate(aeris, gps, col='Datetime')
    
    G2301_gps = merge_with_gps(G2301, gps) # consistent with Utrecht?
    G4302_gps = merge_with_gps(G4302, gps)
    aeris_gps = merge_with_gps(aeris, gps)
    
    
    # =============================================================================
    #   Miro - TNO car
    # =============================================================================
    
    miro     = pd.read_csv(
        TNOPATH+'TNO_miro_06092022.csv',
        index_col=(0),sep=';',
        usecols=[0,8,19,20,21,22],dtype={'comment':'str','geometry':'str',}) #,30-TNO calculated bg, 34-CO2
    
    miro.index = pd.to_datetime(miro.index,dayfirst=True)
    miro.rename(columns={'6_CH4':'CH4_miro','speed':'Speed [m/s]','distance':'Distance'},inplace=True)
    miro = miro.rename_axis('Datetime') # Rename the index column
    miro['CH4_miro'] = miro['CH4_miro']/1000 #convert to ppm
    miro['bg_miro'] = miro['CH4_miro'].rolling('5min',center=True).quantile(0.10)
    miro['CH4_ele_miro'] = miro['CH4_miro'] - miro['bg_miro']
    miro[['Point','Lat_Lon']]  = miro.geometry.str.split(' ',1,expand=True)
    miro[['Longitude','Latitude']]  = miro.Lat_Lon.str.split(' ',1,expand=True)    
    miro['Longitude'] = pd.to_numeric(miro['Longitude'].str[1:],errors='coerce')
    miro['Latitude'] = pd.to_numeric(miro['Latitude'].str[:-1],errors='coerce')
    miro.drop(['geometry','Point','Lat_Lon'],axis=1,inplace=True)    
    
    miro['Speed [m/s]'].where(miro['Speed [m/s]']>0.0,np.nan,inplace=True)
    miro.loc[:,'Speed [m/s]'] = (miro.loc[:,'Speed [m/s]'].interpolate()/3.6)
    
    column_name = miro.columns[3]
    miro.drop(column_name,axis=1, inplace=True) # added, deletes column with text, otherwise not possible to take mean
    miro = miro.groupby(miro.index,axis=0).mean()
    
    
    # =============================================================================
    #   Aerodyne - TNO car
    # =============================================================================
    

    aerodyne = pd.read_csv(TNOPATH+'TNO_aero_06092022.csv',
                           sep = ';',index_col=0,
                           usecols=['datetime', '2_CH4','distance', 'speed',
                                    'comment', 'geometry','wsp_processed', 'wdir_processed']) #'ch4cal_processed' - TNO calculated bg
    
    aerodyne.index                      = pd.to_datetime(aerodyne.index,dayfirst=True)
    aerodyne.rename(columns={'2_CH4':'CH4_aero','speed':'Speed [m/s]','distance':'Distance'},inplace=True)
    aerodyne = aerodyne.rename_axis('Datetime') # Rename the index column
    aerodyne['CH4_aero'] = aerodyne['CH4_aero']/1000
    aerodyne['bg_aero'] = aerodyne['CH4_aero'].rolling('5min', center=True).quantile(0.10)
    aerodyne['CH4_ele_aero'] = aerodyne['CH4_aero'] - aerodyne['bg_aero']
    
    aerodyne[['Point','Lat_Lon']]       = aerodyne.geometry.str.split(' ',1,expand=True)
    aerodyne[['Longitude','Latitude']]  = aerodyne.Lat_Lon.str.split(' ',1,expand=True)    
    aerodyne['Longitude']               = pd.to_numeric(aerodyne['Longitude'].str[1:],errors='coerce')
    aerodyne['Latitude']                = pd.to_numeric(aerodyne['Latitude'].str[:-1],errors='coerce')
    aerodyne.drop(['geometry','Point','Lat_Lon'],
                  axis=1,inplace=True)    
    aerodyne.loc[:,'Speed [m/s]']             = (
        aerodyne.loc[:,'Speed [m/s]'].interpolate())
    aerodyne                            = aerodyne.groupby(aerodyne.index,
                                                           axis=0).mean()
    
    # =============================================================================
    #       Change GPS of G4302&Aeris in Afternoon to TNO GPS
    # =============================================================================
    
    # originally: 
        # if spec == 'G43':
        # dat.index.name  = 'datetime'
        # gps_dat         = merge_interpolate(dat,dat_miro,'datetime')
        # speed           = gps_dat['speed_y']
        # else: gps_dat       = dat
    # in Script Analysis_afterQC before calculation of area starts
    
    # G4302 -------------------------------------------
    
    gps = miro.loc[afternoon_start:afternoon_end, ['Longitude','Latitude','Speed [m/s]']].copy()
    gps.rename(columns={'Longitude': 'Longitude_miro', 'Latitude': 'Latitude_miro','Speed [m/s]': 'Speed_miro'}, inplace=True)
    
    merged_df = pd.merge(G4302_gps.loc[afternoon_start:afternoon_end], gps, left_index=True, right_index=True, how='outer')
    merged_df.interpolate(method='linear', inplace=True)
    merged_df.drop(columns=['Longitude', 'Latitude','Speed [m/s]'], inplace=True)
    merged_df.rename(columns={'Longitude_miro': 'Longitude', 'Latitude_miro': 'Latitude', 'Speed_miro':'Speed [m/s]'}, inplace=True)
    G4302_gps.loc[afternoon_start:afternoon_end,['Longitude', 'Latitude', 'Speed [m/s]']] = merged_df.loc[afternoon_start:afternoon_end,['Longitude', 'Latitude','Speed [m/s]']]
    
    # Aeris -------------------------------------------
    
    gps = miro.loc[afternoon_start:afternoon_end, ['Longitude','Latitude','Speed [m/s]']].copy()
    gps.rename(columns={'Longitude': 'Longitude_miro', 'Latitude': 'Latitude_miro','Speed [m/s]': 'Speed_miro'}, inplace=True)
    
    merged_df = pd.merge(aeris_gps.loc[afternoon_start:afternoon_end], gps, left_index=True, right_index=True, how='outer')
    merged_df.interpolate(method='linear', inplace=True)
    merged_df.drop(columns=['Longitude', 'Latitude','Speed [m/s]'], inplace=True)
    merged_df.rename(columns={'Longitude_miro': 'Longitude', 'Latitude_miro': 'Latitude', 'Speed_miro':'Speed [m/s]'}, inplace=True)
    aeris_gps.loc[afternoon_start:afternoon_end,['Longitude', 'Latitude', 'Speed [m/s]']] = merged_df.loc[afternoon_start:afternoon_end,['Longitude', 'Latitude','Speed [m/s]']]
      

    # =============================================================================
    #   Save to CSV
    # =============================================================================
    
    miro = miro.loc[starttime - timedelta(minutes=30):endtime + timedelta(minutes=30)]
    aerodyne = aerodyne.loc[starttime - timedelta(minutes=30):endtime + timedelta(minutes=30)]
    G4302_gps = G4302_gps.loc[starttime - timedelta(minutes=30):endtime + timedelta(minutes=30)]
    G2301_gps = G2301_gps.loc[starttime - timedelta(minutes=30):endtime + timedelta(minutes=30)]
    aeris_gps = aeris_gps.loc[starttime - timedelta(minutes=30):endtime + timedelta(minutes=30)]
    
    # Print data into csv
    if writexlsx:
        G4302_gps.to_csv(path_res + 'R_G4302.csv', index=True) # way faster than excel
        G2301_gps.to_csv(path_res + 'R_G2301.csv', index=True)
        aeris_gps.to_csv(path_res + 'R_aeris.csv', index=True)
        miro.to_csv(path_res + 'R_miro.csv', index=True)
        aerodyne.to_csv(path_res + 'R_aerodyne.csv', index=True)  
        
    return G4302_gps, G2301_gps, aeris_gps, miro, aerodyne
    
    


#%% End Script



