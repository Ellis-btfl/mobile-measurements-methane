# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 14:03:05 2026

@author: rober
This code reads the picklefile surveys and runs the peak detection algorithm.
The path to the surveys-containig folder needs to be provided along with the path to the analyzer_specs spreadsheet containing the analyzer/survey information. 

The output is a dataframe of peaks with bases (Peakstart and Peakend index), and the integrated area. This output is saved and extracted as a picklefile. Additionally, two visualization methods of the data are provided: an HTML interactive view of the CH$ datastream and an HTML map of the peaks. 
"""

#%% Libraries and Toolbox
import pandas as pd

#test de modif pour vieux pickle
import pandas.core.internals as internals
import pandas._libs.internals as _libs_internals

def _unpickle_block(*args, **kwargs):
    return internals.make_block(*args, **kwargs)

if not hasattr(internals, '_unpickle_block'):
    internals._unpickle_block = _unpickle_block
if not hasattr(_libs_internals, '_unpickle_block'):
    _libs_internals._unpickle_block = _unpickle_block

import numpy as np
import os


#%% Supporting Functions
from postprocessing.readin_data import *
from peak_analysis.find_analyze_peaks import *
from plotting.general_plots import *
from plotting.Viewing import *
from helper_functions.data_handling import *
from peak_analysis.Peak_Analysis import *


#%% Local area quantification function

def compute_area_line_integral(df, start_ts, end_ts, ch4_col, lat_col, lon_col, speed_col):
    """
    Integrate CH4_enhancement * mean_speed * Δt over [start_ts, end_ts].
    mean_speed is estimated as great-circle distance(start,end)/duration.
    Units are proportional to ppm·m (assuming CH4 is in ppm and speed in m/s).
    """
    try:
        # Slice interval
        myslice = df[(df.index >= start_ts) & (df.index <= end_ts)].copy()
        if myslice.empty:
            return np.nan, np.nan

        # Resolve endpoints to existing rows
        start_idx = myslice.index[0]
        end_idx   = myslice.index[-1]
        
        # Integrate over the slice
        area_total = 0.0
        peaklength_total = 0.0
        t_prev = start_idx
        # Ensure CH4 column exists
        if ch4_col not in myslice.columns:
            return np.nan, np.nan
        for t_curr, row in myslice.iterrows():
            if t_curr == start_idx:
                continue
            delta_t = (t_curr - t_prev).total_seconds()
            ch4 = row[ch4_col]
            if pd.notna(ch4):
                area_total += delta_t * ch4  * row[speed_col]
                # area_total += ch4 * delta_t
                peaklength_total = peaklength_total + (row[speed_col] * delta_t)
            t_prev = t_curr

        return area_total, peaklength_total
    except Exception as e:
        print("compute_area_line_integral error:", type(e), e)
        return np.nan, np.nan


def _safe_max(series):
        # Return NaN if empty; otherwise series.max()
        return series.max() if len(series) else np.nan

def single_peak_quantification_method(single_area):
    """
    Calcule le rE pour un seul pic basé sur son aire individuelle (en ppm·m).
    """
    if pd.isna(single_area) or single_area <= 0:
        return np.nan
    
    # Application directe de la formule de régression log-normale
    log_area = np.log(single_area)
    LI_rate_area = np.exp((1.292 * log_area) - 2.377)
    
    return LI_rate_area

#%% Path to Data

folder_path = r'C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\standardised data'

city_surveys = os.listdir(os.path.join(folder_path, r'Surveys\Amsterdam'))

city_data = dict.fromkeys(city_surveys)
fs_df = pd.read_excel(os.path.join(folder_path, 'metadata', 'analyzer_specs.xlsx')) # structure: survey_name, spec_name, fs_spec

#%% Storage space for results
survey_results = dict.fromkeys(city_surveys)


#%% Read-in Data

# Iterate over all city survey data and read them as DataFrames
# Load them into the dictionary with survey name as key     
city_data = {} 

for mysurvey in city_surveys:
    if mysurvey.endswith('.csv'):
        file_path = os.path.join(folder_path, r'Surveys\Amsterdam', mysurvey)
        
        # forcing to keep datetime
        survey_data = pd.read_csv(file_path, sep=',', index_col=0, parse_dates=True)
        
        survey_data['Datetime_UTC'] = pd.to_datetime(survey_data.index)
        
        city_data[mysurvey] = survey_data

#%% Peak detection algorithm v1.5

for mysurvey, df in city_data.items():
    
    spec_name = fs_df[fs_df['survey_name']==mysurvey]['spec_name'].values[0]
    fs_main = fs_df[fs_df['survey_name']==mysurvey]['fs_spec'].values[0]
    
    # df[f'bg_{spec_name}'] = df[f'CH4_{spec_name}'].rolling('5min',center=True).quantile(0.10) 
    # df[f'bg_{spec_name}'] = df[f'CH4_{spec_name}'].rolling('5min', center=True).median()
    df[f'CH4_ele_{spec_name}'] = df[f'CH4_{spec_name}'] - df[f'bg_{spec_name}'] 
    
    # 2) Detection su e_s (smoothing solo per detection)
    
    # --- NETTOYAGE DES DONNÉES AVANT DÉTECTION ---
    spec_col = f'CH4_ele_{spec_name}'
    df[spec_col] = df[spec_col].replace([np.inf, -np.inf], np.nan).fillna(0) #remplace Nan par 0 et supprimes les infs
    
    peaks, props, e, e_s, time_array, fs_used = detect_peaks_excess(
        df, spec_name,
        fs=fs_main,
        lod_ppm=0.01,              # 10–30 ppb → 0.01–0.03 ppm (tune)
        win_smooth_s=11, poly=2,
        prom_k=3, mad_win_s=180,
        w_min_s=6, w_max_s=90, dist_min_s=12
    )
    
    df['Smoothed_CH4'] = pd.Series(e_s, index=df.index)
    
    # 3) Bases 
    left_bases, right_bases = bases_valley_threshold(
        df, spec_name,
        peaks, e_s, fs_used,
        theta_rel_down=0.003, theta_rel_up=0.004,
        lod_ppm=0.006,
        W_left_s=30, W_right_s=30,
        sustain_s=0.3,
        mad_alpha=0.4, min_plateau_s=4.0,
        left_anchor='near_peak', right_anchor='near_peak',
        enable_hysteresis_refine=False,
        max_shift_from_valley_s=1.0,
        safe_gap_s=1.0
    )
    
    # 4) Fix overlap
    left_bases, right_bases = resolve_overlaps(peaks, left_bases, right_bases, e_s)
    
    # 5) Build peakdf + area on enhancement(t) data
    peakdf = df.iloc[peaks].copy()
    left_bases = left_bases.astype(int)
    right_bases = right_bases.astype(int)
    peakdf['Peakstart'] = df.index[left_bases]
    peakdf['Peakend']   = df.index[right_bases]
    # Altezza del picco dal segnale GREZZO (coerente con il modello)
    peakdf['peak max']  = np.round(df.iloc[peaks][f'CH4_ele_{spec_name}'].to_numpy(), 3)
    peakdf['smoothed_peakmax'] = np.round(df.iloc[peaks]['Smoothed_CH4'].to_numpy(), 3)
    
    peakdf['Peakmax_between_bases_raw'] = [
            _safe_max(df.loc[start:end, f'CH4_ele_{spec_name}'])
            for start, end in zip(peakdf['Peakstart'], peakdf['Peakend'])
        ]

    peakdf['Width (s)'] = (peakdf['Peakend'] - peakdf['Peakstart']).dt.total_seconds()
    
    
    # Calculate peaks' area in survey
    
    # peakdf[['Peak_Area_ppmm', 'Peak_Length_m']] = peakdf.apply(lambda row: 
    #                                                            compute_area_line_integral(df, row['Peakstart'], row['Peakend'], f'CH4_ele_{spec_name}', 'Latitude', 'Longitude', 'Speed_provided_m_sec'),
    #                                                            axis=1, result_type='expand')    
    # à la place de compute_area_line_integral                                                          
    areas = []
    lengths = []
    mean_speeds = []
    for idx, row in peakdf.iterrows():
        # on prends les lignes situés entre Peakstart et Peakend, comme ça si il y a des minis erreurs d'arrondi dû au numérique on a qd même qqc
        mask = (df.index >= row['Peakstart']) & (df.index <= row['Peakend'])
        peak_data = df.loc[mask]
        
        if not peak_data.empty and len(peak_data) > 1: #pour ne calculer que si on a bien des pics
        
            
            # Temps écoulé (en secondes), ici c'est tjrs 1s du fait de l'arrondi mais on sait jamais
            temps_en_secondes = (df['Datetime_UTC'] - df['Datetime_UTC'].iloc[0]).dt.total_seconds()
            
            # Calcul de la longueur : méthode des trapèzes
            L = np.trapezoid(peak_data['Speed_provided_m_sec'], x=temps_en_secondes.loc[mask])
            
            # Calcul de l'aire : méthode des trapèzes
            y_values = peak_data[f'CH4_ele_{spec_name}'] * peak_data['Speed_provided_m_sec']
            A = np.trapezoid(y_values, x=temps_en_secondes.loc[mask])
            
            # Calcul de la vitesse moyenne durant le pic
            Speed_avg = peak_data['Speed_provided_m_sec'].mean()
            
            areas.append(A if A > 0 else np.nan)
            lengths.append(L if L > 0 else np.nan)
            mean_speeds.append(Speed_avg)
        else:
            areas.append(np.nan)
            lengths.append(np.nan)
            mean_speeds.append(np.nan)

    #remplissage de Peak_Area_ppmm et de Peak_Length_m
    peakdf['Peak_Area_ppmm'] = areas
    peakdf['rE_Peak_slpm'] = [single_peak_quantification_method(a) for a in areas]
    peakdf['Peak_Length_m'] = lengths
    peakdf['Peak_Mean_Speed_m_s'] = mean_speeds
    peakdf = peakdf[peakdf['Peak_Mean_Speed_m_s'] > 1] #retire les lignes où on est à l'arrêt complet et où il y a des Nan
    peakdf['lenteur'] = peakdf['Peak_Mean_Speed_m_s']
    # On classe les points par vitesse pour retrouver les potentiels pics dûs à des arrêts, marches arrières, hautes vitesses...
    conditions = [(peakdf['Peak_Mean_Speed_m_s'] > 14),
              (peakdf['Peak_Mean_Speed_m_s'] > 2.5) & 
              (peakdf['Peak_Mean_Speed_m_s'] <= 14),(peakdf['Peak_Mean_Speed_m_s'] <= 2.5)]

    choix = ['vitesse rapide', 'vitesse normale', 'vitesse très lente']
    peakdf['lenteur'] = np.select(conditions, choix, default='inconnu')
    print(f"Vérification : {peakdf['Peak_Area_ppmm'].notna().sum()} pics calculés avec succès.")


        
    # Generate map output

    # Save results
    survey_results[mysurvey] = peakdf
    
#%% Extract all stuff
path_to_peaks_csv = r'C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\results'

# Concatenate all the DataFrames in the dictionary by adding the city name as a MultiIndex
all_peaks_df = pd.concat(survey_results.values(), keys=survey_results.keys())

# Save all detected peaks to a single CSV file
global_csv_path = os.path.join(path_to_peaks_csv, 'MySurveys_Amsterdam.csv')
all_peaks_df.to_csv(global_csv_path, sep=';', index=True)

# ‘unserialized_data’ dictionary, recreated from the main CSV file if necessary:
unserialized_data = {
    city: all_peaks_df.xs(city, level=0) 
    for city in all_peaks_df.index.levels[0]
}

#%% Visualize

import plotly.graph_objects as go

for mysurvey, mydf in city_data.items():
    print(mysurvey)
    mypeaks = survey_results[mysurvey]
    
    spec_name = fs_df[fs_df['survey_name']==mysurvey]['spec_name'].values[0]
    
    mydf["Datetime_UTC"] = mydf.index
    
    fig = go.Figure()

    # --- Full CH4 timeseries trace
    trace_ch4 = scatter_charts(
        x=mydf.index,
        y=mydf[f'CH4_ele_{spec_name}'],
        color='blue',
        style="lines+markers",
        name=f"CH₄ (ppm) - {spec_name}",
        my_yaxis='y1'
    )
    fig.add_trace(trace_ch4)
    
    # smoothed trace
    trace_smooth = scatter_charts(
        x=mydf.index,
        y=mydf[f'Smoothed_CH4'],
        color='red',
        style="lines",
        name=f"Smooth CH₄ (ppm) - {spec_name}",
        my_yaxis='y1'
    )
    fig.add_trace(trace_smooth)
    
    # --- Shaded areas under peaks
    if {'Peakstart', 'Peakend'}.issubset(mypeaks.columns):
        for _, peak in mypeaks.iterrows():
            start_time = peak['Peakstart']
            end_time   = peak['Peakend']
            if pd.notna(start_time) and pd.notna(end_time):
                mask = (mydf['Datetime_UTC'] >= start_time) & (mydf['Datetime_UTC'] <= end_time)
                
                if mask.sum() > 0:
                    fig.add_trace(go.Scatter(
                        x=mydf.loc[mask, 'Datetime_UTC'],
                        y=mydf.loc[mask, f'CH4_ele_{spec_name}'],
                        mode="lines",
                        line=dict(color="purple", width=0),
                        fill="tozeroy",       # <-- riempi fino a y=0
                        marker=dict(size=2, color="red"),
                        fillcolor="rgba(240,128,128)",
                        opacity=0.3,
                        showlegend=False,
                        name="Peak area",
                        yaxis='y1'
                    ))
                        
    # --- Peak markers (downward-pointing triangles)
    trace_peaks = scatter_charts(
        x=mypeaks.index,
        y=mypeaks[f'CH4_ele_{spec_name}'],
        color="green",
        style="markers",
        name="Detected peaks",
        my_yaxis='y1'
    )
    trace_peaks.marker.symbol = "triangle-down"
    trace_peaks.marker.size = 10
    fig.add_trace(trace_peaks)
    
    # --- Adding the threshold lines
    trhch4 = scatter_charts(
        x=mydf.index,
        y=0.02*mydf[f'bg_{spec_name}'],
        color="lemonchiffon",
        style="lines",
        name="bg",
        my_yaxis='y1'
    )
    fig.add_trace(trhch4)
    
    
    # --- Layout formatting
    
    fig.update_layout(
        title=f"CH₄ and C₂H₆ timeseries with detected peaks {spec_name}",
        xaxis_title="Time [UTC]",
        yaxis=dict(
            title=dict(
                text="CH₄ concentration [ppm]",
                font=dict(color="blue") # On met la couleur ici
            ),
            tickfont=dict(color="blue")
        ),
        yaxis2=dict(
            title=dict(
                text="C₂H₆ concentration [ppb]",
                font=dict(color="gray") # On met la couleur ici
            ),
            tickfont=dict(color="gray"),
            overlaying="y",
            side="right"
        ),
        legend=dict(orientation="h", y=-0.2),
        template="plotly_white"
    )
    
    # Save interactive figure as HTML
    path_to_results = r'C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\results\cities\Amsterdam'
    timeseries_html = os.path.join(path_to_results, "_"+mysurvey[:-4]+"_CH4_timeseries_peaks_comparison_Amsterdam.html")
    fig.write_html(timeseries_html)
    