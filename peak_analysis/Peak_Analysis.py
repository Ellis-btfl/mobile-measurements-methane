# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:08:42 2025

@author: rober
"""
#%% Libraries and Toolbox
import pandas as pd
import numpy as np
import gpxpy
import gpxpy.gpx
from scipy.interpolate import interp1d
import os
import matplotlib.pyplot as plt
import pytz
from scipy.signal import find_peaks
from geopy.distance import geodesic
from datetime import timedelta
import seaborn as sns
from scipy import stats
import folium
from folium import PolyLine, Polygon
from folium.plugins import MarkerCluster




def find_right_side_index_old(df,row,CH4_column):
    left_base_index = row['Peakstart'] #left base index
    peak_max_index = row.name # peak max
    print(peak_max_index)
    
    left_base_value = df.loc[left_base_index, CH4_column]
    portion_after_peak = df.loc[peak_max_index:peak_max_index+ pd.Timedelta(seconds=30)].copy()
    print('________________________________')
    print('Left base value è: ', left_base_value)
    print('con indice: ', left_base_index)
    print('______________________________________')
    print('E la porzione dopo il picco è: ')
    print(portion_after_peak)
    # Find the first index where the value is within the range of left base value plus/minus 0.02 ppm
    tolerance = 0.02
    matching_indices = portion_after_peak.index[(portion_after_peak[CH4_column] <= left_base_value + tolerance)]
    
    if len(matching_indices) > 0:
        # Get the first index from the matching indices
        index_of_closest_value = matching_indices[0]
        peand_found = 1
    else:
        # If no matching index found, handle the case as per your requirement
        # For example, raise an exception or set index_of_closest_value to a default value.
        index_of_closest_value = portion_after_peak[CH4_column].sub(left_base_value).abs().idxmin()
        peand_found = 0
        
    print('Index closest value is: ', index_of_closest_value)
    return index_of_closest_value, peand_found

def find_right_side_index(df, row, CH4_column, tolerance=0.05, gap_threshold=5):
    left_base_index = row['Peakstart']
    peak_max_index = row.name
    
    left_base_value = df.loc[left_base_index, CH4_column]
    portion_after_peak = df.loc[peak_max_index:peak_max_index + pd.Timedelta(seconds=30)].copy()

    if portion_after_peak.empty:
        print(f"⚠️ Warning: No data after peak at {peak_max_index}. Marking as incomplete.")
        return None, 0, 0  # Picco incompleto

    # **Controllo su Peakend**
    matching_indices = portion_after_peak.index[
        (portion_after_peak[CH4_column] <= left_base_value + tolerance)
    ]

    if len(matching_indices) > 0:
        index_of_closest_value = matching_indices[0]
        peand_found = 1
    else:
        index_of_closest_value = portion_after_peak[CH4_column].sub(left_base_value).abs().idxmin()
        peand_found = 0

    # **Controllo su Peakstart**
    peak_start_value = df.loc[left_base_index, CH4_column]
    
    if peak_start_value > (left_base_value + tolerance):
        print(f"⚠️ Warning: Peakstart at {left_base_index} is above tolerance. Marking as unreliable.")
        pstart_found = 0
    else:
        pstart_found = 1

    

    return index_of_closest_value, peand_found, pstart_found




def improve_base_indices(df,row,CH4_column):
    left_base_index = row['Peakstart'] #left base index
    left_base_index_new = left_base_index
    peak_max_index = row.name # peak max
    print(peak_max_index)
    #---new 04.04.24
    # peak_length = (row['Peakend'] - row['Peakstart']).dt.total_seconds()
    # peak_length_1stpart = (row.index - row['Peakstart']).dt.total_seconds()
    # peak_length_2ndpart = (row['Peakend'] - row.index).dt.total_seconds()
    
    left_base_value = df.loc[left_base_index, CH4_column]
    portion_after_peak = df.loc[peak_max_index:peak_max_index+ pd.Timedelta(minutes=1)].copy()
   
    # Find the first index where the value is within the range of left base value plus/minus 2
    tolerance = 0.01
    matching_indices = portion_after_peak.index[(portion_after_peak[CH4_column] >= left_base_value - tolerance) & (portion_after_peak[CH4_column] <= left_base_value + tolerance)]
    
    if len(matching_indices) > 0:
        # Get the first index from the matching indices
        right_base_index_new = matching_indices[0]
    else:
        # If no matching index found, handle the case as per your requirement
        # For example, raise an exception or set index_of_closest_value to a default value.
        right_base_index_new = portion_after_peak[CH4_column].sub(left_base_value).abs().idxmin()
    
    #---new 04.04.24
    peak_length = (right_base_index_new - row['Peakstart']).total_seconds()
    peak_length_1stpart = (row.name - row['Peakstart']).total_seconds()
    peak_length_2ndpart = (right_base_index_new - row.name).total_seconds()
    
    if peak_length > 120: #(mean length is U:18s - T1b:69s)
        if peak_length_1stpart > peak_length_2ndpart:
            left_base_index_new = peak_max_index - timedelta(seconds=20)
        else:
            right_base_index_new = peak_max_index + timedelta(seconds=60) 
    #---
    return left_base_index_new, right_base_index_new

def process_peak_data(df, distance=3, width=None, writexlsx=False, writer=None, overviewplot=False, savepath = None):
    df = df.copy() 
    bg = df['bg_PICARRO'] 
    CH4data = df['CH4_ele_PICARRO']

    scp_peaks, properties = find_peaks(
        CH4data.values, 
        height=0.02 * bg.values, 
        distance=distance,  # Ridotto per rilevare picchi più stretti
        prominence=0.02 * bg.values,  # Maggiore per evitare falsi positivi
        width=width
    )
    N = len(scp_peaks)
    print(f'Found {N} peaks for PICARRO')

    peakdf = df.iloc[scp_peaks].copy()
    peakdf['peak max'] = np.around(properties['peak_heights'], 2)
    peakdf['Peakstart'] = df.iloc[properties['left_bases']].index
    #right_side_indices = peakdf.apply(find_right_side_index, axis=1)
    peakdf[['Peakend', 'peand_found', 'pstart_found']] = peakdf.apply(
    lambda row: find_right_side_index(df, row, 'CH4_ele_PICARRO'),
    axis=1,
    result_type='expand'
    )

    # Filtriamo solo i picchi affidabili
    peakdf = peakdf[(peakdf['peand_found'] == 1) & (peakdf['pstart_found'] == 1)]
    # peakdf['Width (s)'] = (peakdf['Peakend'] - peakdf['Peakstart']).dt.seconds
    #peakdf['BG'] = bg[scp_peaks].copy()
    peakdf = peakdf.rename(columns={'bg_PICARRO': 'BG'})

    peakdf.index = pd.to_datetime(peakdf.index)
    peakdf.index = peakdf.index.strftime('%Y-%m-%d %H:%M:%S.%f')

    #drop peaks with Peakend not found
    peakdf = peakdf[peakdf['peand_found']==1]
    print('Right side indices: ', peakdf['Peakend'])

    # savepath = None
    # if path_fig:
    #     savepath = path_fig + f"U_Peakplots/peakfinder_{spec}.jpg"
    
    peakdf['DATETIME_UTC'] = pd.to_datetime(peakdf['DATETIME_UTC'])
    peakdf.set_index('DATETIME_UTC', inplace=True, drop=True)
    
    return peakdf

def indices_correction(row, peak_index, df, gap_threshold=10):
    
    my_start = row['Peakstart']
    my_end = row['Peakend']
    mypeak = row['peak max']
    
    before_max_index = pd.to_datetime(peak_index) - pd.Timedelta(seconds=30)
    after_max_index = pd.to_datetime(peak_index) + pd.Timedelta(seconds=30)
    
    portion_before_max = df.loc[before_max_index:pd.to_datetime(peak_index)]
    portion_after_max = df.loc[pd.to_datetime(peak_index):after_max_index]
    
    # Find latest (highest x) value lower than 2% threshold
    
    bg_value = df['bg_PICARRO'].loc[peak_index]
    print('Peak Index: ', peak_index)
    threshold = 0.02 * bg_value
    print('bg value: ', bg_value)
    max_before_index = portion_before_max[portion_before_max['CH4_ele_PICARRO'] < threshold].last_valid_index()
    
    min_after_index = portion_after_max[portion_after_max['CH4_ele_PICARRO'] < threshold].first_valid_index()
    
    if pd.isna(min_after_index):
        min_after_index = portion_after_max[portion_after_max == min(portion_after_max)].index[-1]
        print('Minimo forzato', min_after_index)
        
    if pd.isna(max_before_index):
        max_before_index = portion_before_max[portion_before_max == min(portion_before_max)].index[-1]
        print('Massimo forzato', max_before_index)
        
    # **Controllo sui dati mancanti**
    subset = df.loc[max_before_index:min_after_index]
    time_gaps = subset.index.to_series().diff().dt.total_seconds()
    print('time gaps: ', time_gaps)
    
    if (time_gaps > gap_threshold).any():
        print(f"⚠️ Warning: Large time gap detected within peak at {row.name}. Excluding.")
        return None, None  # Picco incompleto
   
    return max_before_index, min_after_index


def calculate_peak_area(df_CH4_full, df_peaks_row, v_default=4.17):
    # Print row for check
    # Get peak start and stop index    
    peak_start = df_peaks_row['Peakstart_QC']
    peak_stop = df_peaks_row['Peakend_QC']
    # Get CH4 data between peakstart and peakstop
    print('Now I procces the row starting at: ', peak_start)
    my_data = df_CH4_full[(df_CH4_full.index >= peak_start) & (df_CH4_full.index <= peak_stop)].copy()
    
    # Rename CH4 column based on analytics
    if 'CH4_ele_G23' in my_data.columns:
        my_data = my_data.rename(columns={'CH4_ele_G23': 'CH4 analyzer data (ppm)'})
    elif 'CH4_ele_Aeris' in my_data.columns:
        my_data = my_data.rename(columns={'CH4_ele_Aeris': 'CH4 analyzer data (ppm)'})
    elif 'CH4_ele_PICARRO' in my_data.columns:
        my_data = my_data.rename(columns={'CH4_ele_PICARRO': 'CH4 analyzer data (ppm)'})
    
    v_mean = np.mean(my_data['CAR_SPEED'].copy().dropna().values)
    if v_mean == 0:
        raise Exception('Nul mean speed error')
    if pd.isna(v_mean):
        v_mean = v_default
        # raise Exception('NaN mean speed error')
    #print(f'Mean speed is: ', v_default)

    
    # Calculate area
    
    area_method_1 = 0
    area_method_2 = 0
    
    method_1_proportion_count = []
    
    #Initialize left_index
    
    left_index = my_data.index[0]

    for right_index, row in my_data.iterrows():
        if not right_index==my_data.index[0]:
            
            dt = (right_index - left_index).total_seconds()
            if not pd.isna(row['CAR_SPEED']):
                area_method_1 += dt*row['CH4 analyzer data (ppm)']*row['CAR_SPEED']
                method_1_proportion_count.append(1)
                

            else:
                method_1_proportion_count.append(0)
                area_method_1 += dt*row['CH4 analyzer data (ppm)']*v_mean
                
            area_method_2 += dt*row['CH4 analyzer data (ppm)']*v_mean
            
        left_index = right_index  
        
        
        
    method_1_clean = sum(method_1_proportion_count) / len(method_1_proportion_count) * 100
    
    return area_method_1, area_method_2, v_mean, method_1_clean


def wind_preprocessing(u_e, v_n):
    
    v_speed = np.sqrt(u_e**2 + v_n**2)
    v_dir = np.arctan2(v_n, u_e)
    
    v_dir = v_dir + np.pi # flip incoming direction to go to going direction
    
    v_dir_angle = np.degrees(v_dir) % 360
    
    return v_speed, v_dir_angle



def average_wind_direction(directions):
    # Convert wind directions to radians
    angles = np.radians(directions) # directions enter teh function as degrees (°)
    
    # Compute mean x and y components
    mean_x = np.mean(np.cos(angles))
    mean_y = np.mean(np.sin(angles))
    
    # Calculate average direction in radians and convert to degrees
    mean_angle = np.arctan2(mean_y, mean_x)
    mean_direction = np.degrees(mean_angle) % 360
    
    # Adjust to [0, 360] range
    if mean_direction < 0:
        mean_direction += 360
    return mean_direction # Mean direction is given back in degrees (°)

def confidence_interval_wind_direction(directions, confidence=0.95):
    angles = np.radians(directions) # directions enter the function as degrees (°)
    mean_angle = average_wind_direction(directions) # directions enter the function as degrees (°), and mean angle is given in degrees (°)
    mean_x = np.mean(np.cos(angles))
    mean_y = np.mean(np.sin(angles))
    
    # angles are in radians, so r is calculated in radians
    r = np.sqrt(mean_x**2 + mean_y**2)
    
    # Circular standard deviation
    circ_std_dev = np.sqrt(-2 * np.log(r)) # The circular standard deviation is related to radians
    
    # Confidence interval calculation using standard normal distribution
    z = stats.norm.ppf((1 + confidence) / 2)
    ci = circ_std_dev * z # the ci width is in radians
    
    ci_width_rad = ci  # ci was derived from the standard deviation in radians
    
    mean_angle_rad = np.radians(mean_angle) # the mean angle is in degrees (°) 

    # Calculate lower and upper bounds directly in radians
    ci_low_rad = (mean_angle_rad - ci_width_rad) % (2 * np.pi)
    ci_high_rad = (mean_angle_rad + ci_width_rad) % (2 * np.pi)
    
    # Convert bounds back to degrees for use
    ci_low = np.degrees(ci_low_rad)
    ci_high = np.degrees(ci_high_rad)
    
    return ci_low, ci_high


def triangulate_source_with_ci(peaks, wind_directions, wind_speeds, time_window): 
    positions = []
    for index, peak in peaks.iterrows():
        lat, lon = peak['GPS_ABS_LAT'], peak['GPS_ABS_LONG']
        
        wind_data = wind_directions[index - pd.Timedelta(seconds=time_window) : index + pd.Timedelta(seconds=time_window)]
        wind_speed_data = wind_speeds[index - pd.Timedelta(seconds=time_window) : index + pd.Timedelta(seconds=time_window)]
        
        avg_wind_direction = average_wind_direction(wind_data)  # Use wind direction as-is (no 180° adjustment)
        ci_low, ci_high = confidence_interval_wind_direction(wind_data)
        
        avg_wind_speed = np.mean(wind_speed_data)
        distance = avg_wind_speed * 60  # Assuming a 1-minute timeframe for displacement calculation

        # Calculate central source position using avg_wind_direction
        angle_rad = np.radians(avg_wind_direction)
        lat_center = lat + (distance / 111320) * np.cos(angle_rad)
        lon_center = lon + (distance / (111320 * np.cos(np.radians(lat)))) * np.sin(angle_rad)
        
        # Calculate positions for CI bounds
        low_angle_rad = np.radians(ci_low)
        lat_ci_low = lat + (distance / 111320) * np.cos(low_angle_rad)
        lon_ci_low = lon + (distance / (111320 * np.cos(np.radians(lat)))) * np.sin(low_angle_rad)
        
        high_angle_rad = np.radians(ci_high)
        lat_ci_high = lat + (distance / 111320) * np.cos(high_angle_rad)
        lon_ci_high = lon + (distance / (111320 * np.cos(np.radians(lat)))) * np.sin(high_angle_rad)
        
        # Append the central position and CI bounds as a tuple
        positions.append(((lat_center, lon_center), (lat_ci_low, lon_ci_low), (lat_ci_high, lon_ci_high)))
        
        
    
    return positions


def correct_ci_polygon(mean_direction_deg, ci_low_deg, ci_high_deg): # This function corrects the colorfilled area when the ci width is more that 180°
    # Normalize angles to [0, 360)
    ci_low_deg = ci_low_deg % 360
    ci_high_deg = ci_high_deg % 360
    mean_direction_deg = mean_direction_deg % 360

    # Determine CI width in degrees
    ci_width = (ci_high_deg - ci_low_deg) % 360
    if ci_width > 180:
        # If the CI width is more than 180°, swap ci_low and ci_high to cover the larger arc
        ci_low_deg, ci_high_deg = ci_high_deg, ci_low_deg + 360  # Adjust to ensure larger arc is covered

    # Generate points along the arc
    arc_points = []
    num_points = 5  # Increase for smoother polygon edges
    for angle in np.linspace(ci_low_deg, ci_high_deg, num_points):
        angle_rad = np.radians(angle % 360)  # Convert to radians and wrap around
        arc_points.append((np.cos(angle_rad), np.sin(angle_rad)))

    return arc_points # arc_points is a couple cos - sin

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calcola la distanza in metri tra due punti geografici (lat1, lon1) e (lat2, lon2).
    Utilizza la formula della distanza geodetica.
    """
    R = 6372797  # Raggio terrestre in metri
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c # distance is provided in meters


def generate_ci_polygon_points(peak_lat, peak_lon, mean_direction, ci_width, distance, num_points=100):
    points = []
    
    # Start at the lower CI angle
    low_angle = mean_direction - ci_width / 2
    angle_increment = ci_width / num_points
    
    # Include the peak coordinates as the first point of the polygon
    points.append((peak_lat, peak_lon))

    
    for i in range(num_points):
        # Calculate the current angle
        current_angle = low_angle + i * angle_increment
        # Convert the angle to radians
        angle_rad = np.radians(current_angle)
        
        # Calculate the coordinates using the provided distance - distance must be provided in meters
        lat_ci = peak_lat + (distance / 111320) * np.cos(angle_rad)
        lon_ci = peak_lon + (distance / (111320 * np.cos(np.radians(peak_lat)))) * np.sin(angle_rad)
        
        points.append((lat_ci, lon_ci))
    
    return points

def plot_results_with_ci_on_tilemap(peaks, source_positions, ci_bounds=False):
    # Get the center for map initialization
    center_lat = peaks['GPS_ABS_LAT'].mean()
    center_lon = peaks['GPS_ABS_LONG'].mean() 

    # Create the map centered on the average position of the peaks
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # Add a cluster for ppm peaks
    peak_cluster = MarkerCluster(name="Picchi di ppm").add_to(m)
    for index, row in peaks.iterrows():
        folium.Marker(
            location=[row['GPS_ABS_LAT'], row['GPS_ABS_LONG']],
            popup=f'PPM: {row["CH4_ele_PICARRO"]}',
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(peak_cluster)

    # Add lines from peak positions to the estimated source positions
    for index, pos in enumerate(source_positions):
        peak_lat = peaks.iloc[index]['GPS_ABS_LAT']
        peak_lon = peaks.iloc[index]['GPS_ABS_LONG']
        
        # Extract the central position and CI bounds if available
        if ci_bounds and len(pos) == 3:
            (lat_center, lon_center), (lat_ci_low, lon_ci_low), (lat_ci_high, lon_ci_high) = pos
            
            # Calculate the corrected CI polygon points
            mean_direction_deg = np.degrees(np.arctan2(lon_center - peak_lon, lat_center - peak_lat))
            ci_low_deg = np.degrees(np.arctan2(lon_ci_low - peak_lon, lat_ci_low - peak_lat))
            ci_high_deg = np.degrees(np.arctan2(lon_ci_high - peak_lon, lat_ci_high - peak_lat))

            arc_points = correct_ci_polygon(mean_direction_deg, ci_low_deg, ci_high_deg) # arc points is a list of cos - sin couples

            # Convert arc points to latitude/longitude coordinates
            # Calculate distance between the center and peak points
            distance = calculate_distance(lat_center, lon_center, peak_lat, peak_lon)
            ci_distance_low = calculate_distance(lat_ci_low, lon_ci_low, peak_lat, peak_lon)


            # Generate CI polygon points based on mean angle and CI width
            ci_polygon_coords = generate_ci_polygon_points(peak_lat, peak_lon, mean_direction_deg, ci_high_deg-ci_low_deg, distance)
            
            # Line to the central estimated source position
            folium.PolyLine(
                locations=[(peak_lat, peak_lon), (lat_center, lon_center)],
                color="red",
                weight=2,
                opacity=0.6
            ).add_to(m)
            
            # Fill the polygon on the map
            folium.Polygon(
                locations=ci_polygon_coords,
                color="purple",
                fill=True,
                fill_opacity=0.1,
                opacity=0
            ).add_to(m)
        
        else:
            # Fallback: only plot the central position if no CI bounds are available
            lat_center, lon_center = pos[0], pos[1]
            folium.PolyLine(
                locations=[(peak_lat, peak_lon), (lat_center, lon_center)],
                color="blue",
                weight=2,
                opacity=0.6
            ).add_to(m)

    # Add single point sources (example given)
    folium.Marker(
        location=[43.412671743747566, -0.6427637342922369],
        popup="Source 2",
        icon=folium.Icon(color='tab:gold', icon="star")
    ).add_to(m)
    
    folium.Marker(
        location=[43.412884871406995, -0.6424078582891425],
        popup="Source 1",
        icon=folium.Icon(color='tab:gold', icon="star")
    ).add_to(m)

    # Add a layer control
    folium.LayerControl().add_to(m)

    # Return the map
    return m


def process_trajectory(df):
    """
    Processa un DataFrame contenente coordinate geografiche (LAT, LON) e aggiunge
    una colonna con il vettore di traiettoria tra il punto precedente e il successivo.
    Inoltre, aggiunge una colonna "NORMAL_SURFACE" con l'angolo della direzione del versore normale
    alla superficie verticale del movimento.
    
    :param df: DataFrame con indice datetime e colonne ['LAT', 'LON']
    :return: DataFrame con colonne 'TRAJECTORY' e 'NORMAL_SURFACE'
    """
    if df.empty:
        raise ValueError("Il DataFrame è vuoto")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("L'indice del DataFrame deve essere un DatetimeIndex")
    
    df = df.sort_index()  # Assicura che i dati siano ordinati temporalmente
    
    # Calcola i vettori di traiettoria
    df['TRAJECTORY'] = list(zip(df['GPS_ABS_LAT'].shift(-1) - df['GPS_ABS_LAT'], df['GPS_ABS_LONG'].shift(-1) - df['GPS_ABS_LONG']))
    
    # Calcola l'angolo della normale alla traiettoria
    def compute_normal_angle(trajectory):
        delta_lat, delta_lon = trajectory
        if delta_lat == 0 and delta_lon == 0:
            return np.nan
        trajectory_angle = np.arctan2(delta_lat, delta_lon) 
        
        if ((trajectory_angle >= (-np.pi/2)) & (trajectory_angle <= (np.pi/2))):
            normal_angle = trajectory_angle + (np.pi / 2)
        elif ((trajectory_angle > (np.pi/2)) | (trajectory_angle < -(np.pi/2))):
            normal_angle = trajectory_angle + (np.pi / 2) + np.pi
        return np.degrees(normal_angle) % 360  # Converti in gradi e normalizza tra 0 e 360°
    
    df['NORMAL_ANGLE'] = df['TRAJECTORY'].apply(lambda t: compute_normal_angle(t) if not pd.isna(t[0]) else np.nan)
    
    return df

def cosine_between_angles(angle1, angle2):
    """
    Calcola il coseno dell'angolo compreso tra due angoli espressi in gradi.
    
    :param angle1: Primo angolo in gradi
    :param angle2: Secondo angolo in gradi
    :return: Coseno dell'angolo compreso tra i due vettori corrispondenti
    """
    # Converti gli angoli in radianti
    rad1 = np.radians(angle1)
    rad2 = np.radians(angle2)
    
    # Determina i versori dei due angoli
    vec1 = np.array([np.cos(rad1), np.sin(rad1)])
    vec2 = np.array([np.cos(rad2), np.sin(rad2)])
    
    # Calcola il prodotto scalare
    dot_product = np.dot(vec1, vec2)
    
    
    return dot_product




def evaluate_mass_balance_flow(df_CH4_full, df_peaks_row, peak_height):
    
    df_CH4 = df_CH4_full.copy()
    
    peakstart = df_peaks_row['Peakstart_QC']
    peakstop = df_peaks_row['Peakend_QC']
    
    df_CH4 = df_CH4[(df_CH4.index >= peakstart) & (df_CH4.index >= peakstop)]
    
    
    
    df_CH4['CH4_wind_product'] = df_CH4.apply(
        lambda row: row['CH4_ele_PICARRO'] * row['WIND_SPEED'] * cosine_between_angles(row['NORMAL_ANGLE'], np.deg2rad(row['WIND_DIRECTIION'])), 
        axis=1
    )
    
    print(df_CH4['CH4_wind_product'])
    
    G_spec = df_CH4['CH4_wind_product'].sum(skipna=True)
    
    print(G_spec)

    
    G_estimate = G_spec * peak_height
    
    return G_estimate
    
    


def plot_time_series_and_boxplots(Max_data, Area_data, Kurtosis_data, Lasting_data):
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(4, 2, figsize=(14, 10), gridspec_kw={'width_ratios': [3, 1]})
    
    data_dict = {
        'Max CH4': Max_data, 
        'Area M1 (ppm*m)': Area_data, 
        'Kurtosis von Fischer': Kurtosis_data, 
        'Peak Duration': Lasting_data
    }
    
    # Time series plots
    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple']
    for i, (label, data) in enumerate(data_dict.items()):
        axes[i, 0].plot(data, marker='o', linestyle='-', color=colors[i], label=label)
        axes[i, 0].set_title(f"{label} Variation")
        axes[i, 0].set_ylabel(label)
        axes[i, 0].legend()
        
        # Boxplot
        sns.boxplot(y=data, ax=axes[i, 1], color=colors[i])
        axes[i, 1].set_title(f"{label} Boxplot")
    
    axes[-1, 0].set_xlabel("Time Index")
    plt.tight_layout()
    plt.show()
    
    # Compute variability statistics
    stats = pd.DataFrame({
        'Mean': [Max_data.mean(), Area_data.mean(), Kurtosis_data.mean(), Lasting_data.mean()],
        'Std Dev': [Max_data.std(), Area_data.std(), Kurtosis_data.std(), Lasting_data.std()],
        'Range': [Max_data.max() - Max_data.min(), Area_data.max() - Area_data.min(), 
                  Kurtosis_data.max() - Kurtosis_data.min(), Lasting_data.max() - Lasting_data.min()]
    }, index=['Max CH4', 'Area M1', 'Kurtosis', 'Peak Duration'])
    
    print("Variability Statistics:\n", stats)
    
    return stats