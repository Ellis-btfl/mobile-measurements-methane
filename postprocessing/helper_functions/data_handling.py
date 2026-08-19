# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 15:35:58 2024

@author: Judith -- adapted by roberto
"""

# Import necessary packages
import os
import pandas as pd
import gpxpy


# Read the .dat files into dataframes (stored in a dictionary)
def dat_to_df_dict(path):
    # List all .dat files in the directory
    dat_files = [f for f in os.listdir(path) if f.endswith(".dat")]
    
    # Create an empty dictionary to store dataframes
    dfs = {}
    
    # Loop through each .dat file and read it into a dataframe
    for i, file in enumerate(dat_files):
        # Construct the full file path
        file_path = os.path.join(path, file)
        
        # Read the .dat file into a dataframe
        df = pd.read_csv(file_path, delim_whitespace=True)
        
        # Use custom keys for the dataframes, e.g., data1, data2, data3, ...
        key = "data{}".format(i+1)
        dfs[key] = df
    return dfs

def text_to_df_dict(path):
    # List all .txt files into dataframes (stored in a directory)
    txt_files = [f for f in os.listdir(path) if f.endswith(".txt")]
    
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

# Real the -gpx files into dataframe (stored in a directory)
def gpx_to_df(path):
    # List all -gpx files in the directory
    
    gpx_files = [f for f in os.listdir(path) if f.endswith('.gpx')]
    
    # Create an empty dictionary to store dataframes
    dfs = {}
    
    # Loopthrough each .gpx file and read it into a dataframe 
    for i, file in enumerate(gpx_files):
        # Construct full file path
        file_path = os.path.join(path, file)
        
        # Parse the .gpx file
        with open(file_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            
        # Extract data from GPX file
        data = []
         
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation,
                        'time': point.time,
                        'speed': point.speed
                        })
        # Convert data to dataframe
        df = pd.DataFrame(data)
         
        # Use custom keys for accessing the dataframe
        key = 'data{}'.format(i+1)
        dfs[key] = df

    return dfs


# calibrate CH4 measurements with the instruments Picarro G2301, Picarro G4302
# and Aeris (obtained from experiments in the lab by Carina van der Veen)
def calibrate(x, inst, typ):
        if inst == 'G23':
            if typ == 'CH4': 
                return 1.03127068196 * x - 0.15799666857 
            if typ == 'CO2':
                return 1.0088 * x + 0.320
        elif inst =='G43':
            if typ == 'CH4':
                methane = 1.01924906721 * x - 0.05887406866
                return methane
            elif typ == 'C2H6':
                ethane = 0.9950 * x            
                return ethane    
        elif inst == 'aer':
            if typ == 'CH4':
                return 1.01354227768 * x - 0.05055326961
            
# ...
def merge_with_gps(df_CH4,gps):
    df_merged = df_CH4.copy(deep=True)
    # Verifica del tipo di index
        
    #Aggiunta della colonna time_round
    df_merged['time_round'] = df_merged.index.round('1s')
    
    # Conversione in datetime
    df_merged['time_round'] = pd.to_datetime(df_merged['time_round'])
    gps['time_round'] = pd.to_datetime(gps.index)
    
    # Rimozione del timezone per rendere più gestibili i dati temporali
    df_merged['time_round'] = df_merged['time_round'].dt.tz_convert(None)
    
    gps['time_round'] = gps['time_round'].dt.tz_convert(None)
    
    df_merged = df_merged.join(gps.set_index('time_round'),on='time_round')
    df_merged.drop(['time_round'],axis=1,inplace=True)
    
    # Assicurati che l'indice sia naive
    df_merged.index = df_merged.index.tz_convert(None)
    
    return df_merged

# def merge_interpolate(df1,df2,col): # used for Rotterdam UU car
#         if df1.index.name == col:
#             combined = pd.merge(df1,df2,on=col,how='outer')
#             combined = combined.sort_values(by=col)
#             combined = combined.interpolate(method='linear')


def merge_interpolate_left(df1,df2,col):
    
    combined = pd.merge(df1,df2,on=col,how='left')
    combined = combined.sort_values(by=col)
    combined = combined.interpolate(method='linear')
    return combined

def combine_loc_and_rr(row):
    return (row['Loc'], row['Release_rate'])


def merge_WS_data(dfCH4, df_WS):
    
    combined = pd.merge(dfCH4, df_WS, left_index=True, right_index=True, how='left')
    combined = combined.sort_index()
    combined = combined.interpolate(method='linear')
    return combined




