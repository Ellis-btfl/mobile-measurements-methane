# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:08:16 2025

@author: rober
"""
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
import math


def scatter_charts(x, y, style, name, color='red', trasparenza=0.8, my_yaxis='y1'):
    
    trace = go.Scatter(
        x=x.values,
        y=y.values,
        mode = style,
        name=name,
        line=dict(color=color, width=3, dash='dash'), # Dashed blue line
        marker=dict(size=2, color=color), # Red markers
        # marker=dict(
        #     color=color
        # ),
        opacity=trasparenza,
        yaxis=my_yaxis
    )
    return trace


def create_heatmap_with_trajectory(df):
    """
    Funzione che prende un dataframe con i dati di CH4 e le coordinate GPS
    del sensore, e restituisce una mappa interattiva con heatmap e traiettoria.
    
    Parametro:
    df: pandas.DataFrame con le colonne 'datetime', 'CH4', 'latitude', 'longitude'.
    
    Restituisce:
    Una mappa interattiva.
    """
    # Controlliamo che il dataframe contenga le colonne necessarie
    required_columns = ['DATETIME_UTC', 'CH4', 'GPS_ABS_LAT', 'GPS_ABS_LONG']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Il dataframe deve contenere le colonne: {required_columns}")
    
    # Creiamo la mappa centrata sulla media delle coordinate
    mean_lat = df['GPS_ABS_LAT'].mean()
    mean_lon = df['GPS_ABS_LONG'].mean()
    
    # Creiamo la mappa con visualizzazione satellitare
    mappa = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles='OpenStreetMap')  # Cambia con Esri o altri per visione satellitare
    
    
    # Aggiungiamo la HeatMap
    heat_data = [[row['GPS_ABS_LAT'], row['GPS_ABS_LONG'], row['CH4']] for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(mappa)
    
    # Aggiungiamo la traiettoria del sensore
    traj_points = list(zip(df['GPS_ABS_LAT'], df['GPS_ABS_LONG']))
    folium.PolyLine(traj_points, color="blue", weight=2.5, opacity=1).add_to(mappa)
    
    return mappa


def create_map_with_wind_vectors(df):
    """
    Crea una mappa con frecce che mostrano la direzione del vento in base ai dati
    di velocità del vento nelle colonne 'WIND_N' e 'WIND_E'.
    
    Parametri:
    df: pandas.DataFrame con le colonne 'datetime', 'latitude', 'longitude', 'WIND_N', 'WIND_E'.
    
    Restituisce:
    Una mappa interattiva con le frecce che indicano la direzione del vento.
    """
    # Controlliamo che il dataframe contenga le colonne necessarie
    required_columns = ['DATETIME_UTC', 'GPS_ABS_LAT', 'GPS_ABS_LONG', 'WIND_N', 'WIND_E']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Il dataframe deve contenere le colonne: {required_columns}")
    
    # Creiamo la mappa centrata sulla media delle coordinate
    mean_lat = df['GPS_ABS_LAT'].mean()
    mean_lon = df['GPS_ABS_LONG'].mean()
    mappa = folium.Map(location=[mean_lat, mean_lon], zoom_start=12, tiles='OpenStreetMap')
    
    # Aggiungiamo i vettori del vento
    for index, row in df.iterrows():
        # Estrai le componenti di velocità del vento
        wind_n = row['WIND_N']
        wind_e = row['WIND_E']
        lat = row['GPS_ABS_LAT']
        lon = row['GPS_ABS_LONG']
        print(wind_n)
        # Calcola la direzione del vento (in gradi) rispetto all'asse nord (0°)
        angle = math.degrees(math.atan2(wind_n, wind_e))  # angolo della direzione del vento
        
        # La lunghezza della linea può essere un fattore di scala, qui moltiplichiamo per una costante
        wind_speed = math.sqrt(wind_n**2 + wind_e**2)  # velocità del vento
        scale_factor = 0.001  # Usa un fattore di scala per controllare la lunghezza della linea
        line_length = wind_speed * scale_factor  # Lunghezza della linea

        # Creiamo un punto finale della linea (opposto alla direzione del vento)
        # Semplificando con un piccolo passo (distanza in gradi) per mostrare la direzione
        # Spostiamo leggermente il punto per disegnare la linea in direzione opposta
        
        # Calcola il nuovo punto per la linea (opposto alla direzione del vento)
        new_lat = lat - line_length * (wind_n / wind_speed)
        new_lon = lon - line_length * (wind_e / wind_speed)
        
        # Aggiungi la linea sulla mappa
        folium.PolyLine(
            locations=[(lat, lon), (new_lat, new_lon)], 
            color="blue", 
            weight=2,
            opacity=0.6
        ).add_to(mappa)

    return mappa
