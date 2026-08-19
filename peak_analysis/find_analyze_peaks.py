# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 16:32:30 2024

@author: Judith - Adapted by Roberto

Collection of functions necessary for finding and analyzing peaks.
To be used in file ? postprocessing_script.py ?

"""

# Import necessary packages
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from geopy.distance import geodesic
from scipy.signal import savgol_filter


from plotting.general_plots import *


#%% New functions

# ----------------------- Utility -----------------------
def ensure_fs(fs):
    if fs is None:
        raise ValueError("fs è None. Passa esplicitamente la sampling frequency (Hz).")
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs deve essere > 0, ricevuto: {fs}")
    return fs

def to_samples(param_seconds, fs, min_samples=1, make_odd=False):
    s = int(round(param_seconds * fs))
    s = max(min_samples, s)
    if make_odd and s % 2 == 0:
        s += 1
    return s

def rolling_mad(x, win_samples):
    n = len(x)
    out = np.full(n, np.nan)
    half = max(1, win_samples // 2)
    for i in range(n):
        s = max(0, i - half); e = min(n, i + half + 1)
        seg = x[s:e]
        med = np.nanmedian(seg)
        out[i] = np.nanmedian(np.abs(seg - med))
    return out

# ------------------- Smoothing (solo detection) -------------------
def smooth_excess_for_detection(e, fs, win_s=11, poly=4):
    """
    Smoothing centrato (Savitzky-Golay) su e(t) per DETECTION.
    Restituisce e_s. NON usare e_s per l'integrale dell'area.
    """
    wl = to_samples(win_s, fs, min_samples=5, make_odd=True)
    e_s = savgol_filter(e, window_length=wl, polyorder=poly, mode='interp')
    return e_s

# ------------------- Detection su e_s -------------------
def detect_peaks_excess(df, spec, fs,
                        lod_ppm=0.02,       # 20 ppb
                        win_smooth_s=11, poly=2,
                        prom_k=3, mad_win_s=180,
                        w_min_s=6, w_max_s=90, dist_min_s=12):
    """
    Trova picchi su e_s con soglie fisiche:
    - height_min = max(0.02*bg, lod_ppm)
    - prominence_min = prom_k * median(sigma_loc), con sigma_loc = 1.4826 * MAD(e_s)
    - width/distance in secondi → campioni
    Ritorna: peaks, props (SciPy su e_s), e_s (per valli/basi), fs
    """
    fs = ensure_fs(fs)
    e = df[f'CH4_ele_{spec}'].to_numpy()   # e(t) = excess grezzo (per area)
    b = df[f'bg_{spec}'].to_numpy()        # background centrato (dal loader)
    ttt = df[f'CH4_ele_{spec}'].index.to_numpy()

    e_s = smooth_excess_for_detection(e, fs, win_smooth_s, poly)

    mad_win = to_samples(mad_win_s, fs, min_samples=5)
    sigma_loc = 1.4826 * rolling_mad(e_s, mad_win)
    prom_min = prom_k * np.nanmedian(sigma_loc)

    height_min = np.maximum(0.02 * b, lod_ppm)

    w_min   = to_samples(w_min_s,   fs)
    w_max   = to_samples(w_max_s,   fs)
    dist_min = to_samples(dist_min_s, fs)

    peaks, props = find_peaks(e_s, height=height_min,
                              prominence=prom_min,
                              width=(w_min, w_max),
                              distance=dist_min)
    return peaks, props, e, e_s, ttt, fs

# ------------------- Flat-top guard -------------------
def detect_flat_top(e_s, p, fs, eps_rel=0.03, min_dur_s=2.0):
    """
    Rileva un flat-top attorno al massimo del picco p.
    Restituisce (has_flat, left_ft, right_ft) in indici.
    Definizione: e_s resta entro ±eps_rel * e_s[p] per >= min_dur_s.
    """
    n = len(e_s)
    peak_val = e_s[p]
    thr_low  = peak_val * (1 - eps_rel)
    thr_high = peak_val * (1 + eps_rel)

    i = p
    while i > 0 and thr_low <= e_s[i] <= thr_high:
        i -= 1
    left_ft = i + 1

    j = p
    while j < n - 1 and thr_low <= e_s[j] <= thr_high:
        j += 1
    right_ft = j - 1

    dur_s = (right_ft - left_ft + 1) / fs
    has_flat = (dur_s >= min_dur_s)
    return has_flat, left_ft, right_ft

# ------------------- Valli/Plateau di coda -------------------
def local_mad_on_segment(e_s, start, end, default=np.nan):
    if end <= start + 2: return default
    seg = e_s[start:end]
    med = np.nanmedian(seg)
    mad = np.nanmedian(np.abs(seg - med))
    return mad

def find_valley_or_tail_plateau(e_s, start, end, fs, tau_down,
                                mad_alpha=0.5, min_plateau_s=3.0,
                                anchor_mode='near_peak'):
    """
    Individua valle o plateau DI CODA nella finestra [start, end].
    Plateau accettato SOLO se il suo livello medio <= tau_down medio.
    Altrimenti usa argmin (valle classica).
    """
    start = max(0, start); end = min(len(e_s), end)
    if end <= start: return start

    seg = e_s[start:end]
    delta = float(np.nanmax(seg) - np.nanmin(seg))
    duration_s = (end - start) / fs
    mad_loc = local_mad_on_segment(e_s, start, end, default=np.nan)
    mean_seg = float(np.nanmean(seg))
    mean_tau = float(np.nanmean(tau_down[start:end]))

    is_plateau = (np.isfinite(mad_loc) and delta <= mad_alpha * mad_loc
                  and duration_s >= min_plateau_s and mean_seg <= mean_tau)

    if is_plateau:
        if anchor_mode == 'center':
            return start + (end - start) // 2
        else:
            # near_peak: bordo verso il picco (caller imposta lato)
            return max(start, min(end - 1, end - 1))
    else:
        return start + int(np.nanargmin(seg))

# ------------------- Crossing sostenuto -------------------
def sustained_crossing(y, thr, start, end, sustain_samples):
    step = 1 if end >= start else -1
    cnt = 0; i = start
    while (i - end) * step <= 0:
        if y[i] <= thr[i]:
            cnt += 1
            if cnt >= sustain_samples:
                return i - (sustain_samples - 1) * step
        else:
            cnt = 0
        i += step
    return start

# ------------------- Basi: valli + soglie + clamp -------------------
def bases_valley_threshold(
    df, spec, peaks, e_s, fs,
    theta_rel_down=0.003, theta_rel_up=0.004,
    lod_ppm=0.02,
    W_left_s=120, W_right_s=120,
    sustain_s=0.3,
    mad_alpha=0.4, min_plateau_s=4.0,
    left_anchor='center', right_anchor='near_peak',
    enable_hysteresis_refine=False,
    max_shift_from_valley_s=1.0,
    safe_gap_s=1.0
):
    """
    Calcola basi su e_s:
    - valli/plateau di coda con flat-top guard e clamp,
    - crossing sostenuto con isteresi opzionale (limitata),
    - clamp: lb >= lv, rb <= rv + ±max_shift.
    """
    fs = ensure_fs(fs)
    b = df[f'bg_{spec}'].to_numpy()
    tau_down = np.maximum(theta_rel_down * b, lod_ppm)
    tau_up   = np.maximum(theta_rel_up   * b, lod_ppm)

    Wl = to_samples(W_left_s,  fs)
    Wr = to_samples(W_right_s, fs)
    sustain   = to_samples(sustain_s, fs)
    max_shift = to_samples(max_shift_from_valley_s, fs)
    safe_gap  = to_samples(safe_gap_s, fs)

    n = len(e_s)
    left_bases, right_bases = [], []

    for p in peaks:
        has_flat, ftL, ftR = detect_flat_top(e_s, p, fs, eps_rel=0.03, min_dur_s=2.0)

        sL = max(0, p - Wl); eL = p
        if has_flat: eL = max(sL + 1, ftL - safe_gap)

        sR = p; eR = min(n, p + Wr)
        if has_flat: sR = min(eR - 1, ftR + safe_gap)

        lv = find_valley_or_tail_plateau(e_s, sL, eL, fs, tau_down,
                                         mad_alpha=mad_alpha, min_plateau_s=min_plateau_s,
                                         anchor_mode=left_anchor)
        rv = find_valley_or_tail_plateau(e_s, sR, eR, fs, tau_down,
                                         mad_alpha=mad_alpha, min_plateau_s=min_plateau_s,
                                         anchor_mode=right_anchor)

        lb = sustained_crossing(e_s, tau_down, start=lv, end=p, sustain_samples=sustain)
        rb = sustained_crossing(e_s, tau_down, start=p,  end=rv, sustain_samples=sustain)

        if enable_hysteresis_refine:
            i = lb; steps = 0
            while i > sL and e_s[i] <= tau_up[i] and steps < max_shift:
                i -= 1; steps += 1
            lb = i
            i = rb; steps = 0
            while i < (eR - 1) and e_s[i] <= tau_up[i] and steps < max_shift:
                i += 1; steps += 1
            rb = i

        # CLAMP rispetto alle valli/plateau e max_shift
        lb = max(lb, lv); rb = min(rb, rv)
        lb = int(np.clip(lb, lv - max_shift, lv + max_shift))
        rb = int(np.clip(rb, rv - max_shift, rv + max_shift))

        lb = max(sL, min(lb, p - 1))
        rb = min(eR - 1, max(rb, p + 1))

        left_bases.append(lb); right_bases.append(rb)

    return np.array(left_bases), np.array(right_bases)

# ------------------- Risoluzione overlap tra picchi -------------------
def resolve_overlaps(peaks, left_bases, right_bases, e_s):
    """
    Se [lb_i, rb_i] e [lb_j, rb_j] si sovrappongono,
    fissare il confine nel minimo di e_s tra i picchi (1D-watershed).
    """
    lb = left_bases.copy(); rb = right_bases.copy()
    order = np.argsort(peaks)
    for k in range(len(peaks) - 1):
        i = order[k]; j = order[k + 1]
        if rb[i] >= lb[j]:  # overlap
            s = max(peaks[i], lb[j]); e = min(peaks[j], rb[i])
            split = (s + int(np.argmin(e_s[s:e]))) if e > s else ((rb[i] + lb[j]) // 2)
            rb[i] = min(rb[i], split)
            lb[j] = max(lb[j], split + 1)
    return lb, rb

# ------------------- Area su e(t) grezzo -------------------
def area_per_peak(df, spec, peaks, left_bases, right_bases):
    """
    Integra e(t) (NON smussato) tra le basi (trapz con dt reale).
    Unità: ppm·s.
    """
    e = df[f'CH4_ele_{spec}'].to_numpy()
    t = pd.to_datetime(df.index).values.astype('datetime64[ns]').astype(np.int64) / 1e9  # sec
    areas = []
    for p, lb, rb in zip(peaks, left_bases, right_bases):
        if rb <= lb:
            areas.append(0.0); continue
        xi = e[lb:rb+1]; ti = t[lb:rb+1]
        area = np.trapz(xi, ti)
        areas.append(float(area))
    return np.array(areas)


    """
    Verifica/ottiene la frequenza di campionamento (Hz).
    - Se fs è fornito ed è valido (>0), lo restituisce.
    - Altrimenti, se default_fs è fornito ed è valido, lo restituisce.
    - Altrimenti, se time_index è fornito, prova a stimare fs dai timestamp (median dt).
    - Se non riesce, solleva un ValueError con messaggio chiaro.
    """
    if fs is not None:
        try:
            fs = float(fs)
        except Exception:
            raise ValueError(f"Sampling frequency fs non convertibile a float: {fs!r}")
        if fs <= 0:
            raise ValueError(f"Sampling frequency fs deve essere > 0, ricevuto: {fs}")
        return fs

    if default_fs is not None:
        try:
            default_fs = float(default_fs)
        except Exception:
            raise ValueError(f"default_fs non convertibile a float: {default_fs!r}")
        if default_fs <= 0:
            raise ValueError(f"default_fs deve essere > 0, ricevuto: {default_fs}")
        return default_fs

    if time_index is not None and len(time_index) > 1:
        # Stima dell'fs dai timestamp (in secondi) con mediana dell'intervallo
        import numpy as np, pandas as pd
        t = pd.to_datetime(time_index)
        dt = np.diff(t.values).astype('timedelta64[ns]').astype(float) / 1e9
        median_dt = np.nanmedian(dt)
        if np.isfinite(median_dt) and median_dt > 0:
            return 1.0 / median_dt

    raise ValueError("Sampling frequency fs non fornita e non stimabile dai timestamp.")


#%% Old functions
def find_right_side_index(df,row,CH4_column):
    left_base_index = row['Peakstart'] #left base index
    peak_max_index = row.name # peak max
    print(peak_max_index)
    
    left_base_value = df.loc[left_base_index, CH4_column]
    print(left_base_value)
    portion_after_peak = df.loc[peak_max_index:peak_max_index+ pd.Timedelta(minutes=1)].copy()
   
    # Find the first index where the value is within the range of left base value plus/minus 2
    tolerance = 0.01
    matching_indices = portion_after_peak.index[(portion_after_peak[CH4_column] >= (left_base_value - tolerance)) & (portion_after_peak[CH4_column] <= (left_base_value + tolerance))]
    
    if len(matching_indices) > 0:
        # Get the first index from the matching indices
        index_of_closest_value = matching_indices[0]
    else:
        # If no matching index found, handle the case as per your requirement
        # For example, raise an exception or set index_of_closest_value to a default value.
        index_of_closest_value = portion_after_peak[CH4_column].sub(left_base_value).abs().idxmin()
    return index_of_closest_value


def revise_left_side_index(df,row,CH4_column):
    left_base_index = row['Peakstart'] #left base index
    peak_max_index = row.name # peak max
    print(peak_max_index)
    
    left_base_value = df.loc[left_base_index, CH4_column]
    print(left_base_value)
    portion_before_peak = df.loc[left_base_index:peak_max_index].copy()
   
    # Find the first index where the value is within the range of left base value plus/minus 2
    tolerance = 0.01
    matching_indices = portion_before_peak.index[(portion_before_peak[CH4_column] >= (left_base_value - tolerance)) & (portion_before_peak[CH4_column] <= (left_base_value + tolerance))]
    
    if len(matching_indices) > 0:
        # Get the first index from the matching indices
        index_of_closest_value = matching_indices[-1]
    else:
        # If no matching index found, handle the case as per your requirement
        # For example, raise an exception or set index_of_closest_value to a default value.
        index_of_closest_value = portion_before_peak[CH4_column].sub(left_base_value).abs().idxmin()
    return index_of_closest_value

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



def process_peak_data_withC2H6(df, spec, distance=10, width=None, writexlsx=False, writer=None, overviewplot=False, savepath = None):
    df = df.copy() #.loc[morning_start:morning_end]
    bg = df[('bg_'+spec)] #spec = G2 or aero
    # Inclusion Sept. 30th 2025 - Ethane Data
    bg_ethane = df[('bg_C2H6_'+spec)] # ethane background added
    CH4data = df[('CH4_ele_'+spec)]
    C2H6data = df[('C2H6_ele_'+spec)]
    
    C2H6data_smooth = savgol_filter(C2H6data.values, window_length=11, polyorder=3)

    scp_peaks, properties = find_peaks(CH4data.values, height=0.02 * bg.values, distance=distance, width=width)
    N = len(scp_peaks)
    print(f'Found {N} peaks for {spec}')
    
    ethane_peaks, ethane_prop = find_peaks(C2H6data.values, height=1, distance=distance, width=width)
    N_ethane = len(ethane_peaks)
    print(f"Found {N} methane peaks and {N_ethane} ethane peaks for {spec}")
    
    smooth_ethane_peaks, smooth_ethane_prop = find_peaks(C2H6data_smooth, height=0.10 * bg_ethane.values, distance=distance, width=width)
    N_ethane = len(ethane_peaks)
    print(f"Found {N} methane peaks and {N_ethane} smooth ethane peaks for {spec}")

    peakdf = df.iloc[scp_peaks].copy()
    peakdf['peak max'] = np.around(properties['peak_heights'], 2)
    peakdf['Peakstart'] = df.iloc[properties['left_bases']].index
    #right_side_indices = peakdf.apply(find_right_side_index, axis=1)
    right_side_indices = peakdf.apply(lambda row: find_right_side_index(df, row, ('CH4_ele_'+spec)), axis=1)
    print('Right side indices: ', right_side_indices)
    peakdf['Peakend'] = right_side_indices
    peakdf['Peakstart'] = peakdf.apply(lambda row: revise_left_side_index(df, row, ('CH4_ele_'+spec)), axis=1)
    peakdf['Width (s)'] = (peakdf['Peakend'] - peakdf['Peakstart']).dt.seconds
    #peakdf['BG'] = bg[scp_peaks].copy()
    peakdf = peakdf.rename(columns={'bg_'+spec: 'BG'})

    peakdf.index = pd.to_datetime(peakdf.index)
    peakdf.index = peakdf.index.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    ethane_df = df.iloc[ethane_peaks].copy()
    ethane_df['peak max'] = np.around(ethane_prop['peak_heights'], 2)
    ethane_df['Peakstart'] = df.iloc[ethane_prop['left_bases']].index
    right_side_indices_ethane = ethane_df.apply(lambda row: revise_left_side_index(df, row, ('C2H6_ele_'+spec)), axis=1)
    ethane_df['Peakend'] = right_side_indices_ethane
    ethane_df['Peakstart'] =  peakdf.apply(lambda row: revise_left_side_index(df, row, ('C2H6_ele_'+spec)), axis=1)
    ethane_df['Width (s)'] = (ethane_df['Peakend'] - ethane_df['Peakstart']).dt.seconds
    ethane_df = ethane_df.rename(columns={'bg_C2H6_'+spec: 'BG_ethane'})
    
    ethane_df.index = pd.to_datetime(ethane_df.index)
    ethane_df.index = ethane_df.index.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    smooth_etane_df = df.iloc[smooth_ethane_peaks].copy()
    smooth_etane_df.index = pd.to_datetime(smooth_etane_df.index)
    smooth_etane_df.index = smooth_etane_df.index .strftime('%Y-%m-%d %H:%M:%S.%f')

    # savepath = None
    # if path_fig:
    #     savepath = path_fig + f"U_Peakplots/peakfinder_{spec}.jpg"
    
    
    
    if overviewplot:
        overview_plot_ver2(CH4data, df['Latitude'], scp_peaks, spec, N, bg=bg, th=1.1 * bg, savepath=savepath)
    
    print("qui arrivo")
    if peakdf.empty:
        print("No CH4")
    
    if ethane_df.empty:
        print("No C2H6")
    if smooth_etane_df.empty:
        print("No smooth")
    print("Returning:", peakdf is not None, ethane_df is not None, 'smooth_etane_df' in locals())
    return peakdf, ethane_df, smooth_etane_df



def process_peak_data(df, spec, distance=10, width=None, writexlsx=False, writer=None, overviewplot=False, savepath = None):
    df = df.copy() #.loc[morning_start:morning_end]
    bg = df[('bg_'+spec)] #spec = G2 or aero
    # Inclusion Sept. 30th 2025 - Ethane Data
    CH4data = df[('CH4_ele_'+spec)]
    

    scp_peaks, properties = find_peaks(CH4data.values, height=0.02 * bg.values, distance=distance, width=width)
    N = len(scp_peaks)
    print(f'Found {N} peaks for {spec}')
    
    peakdf = df.iloc[scp_peaks].copy()
    peakdf['peak max'] = np.around(properties['peak_heights'], 2)
    peakdf['Peakstart'] = df.iloc[properties['left_bases']].index
    #right_side_indices = peakdf.apply(find_right_side_index, axis=1)
    right_side_indices = peakdf.apply(lambda row: find_right_side_index(df, row, ('CH4_ele_'+spec)), axis=1)
    print('Right side indices: ', right_side_indices)
    peakdf['Peakend'] = right_side_indices
    peakdf['Peakstart'] = peakdf.apply(lambda row: revise_left_side_index(df, row, ('CH4_ele_'+spec)), axis=1)
    peakdf['Width (s)'] = (peakdf['Peakend'] - peakdf['Peakstart']).dt.seconds
    #peakdf['BG'] = bg[scp_peaks].copy()
    peakdf = peakdf.rename(columns={'bg_'+spec: 'BG'})

    peakdf.index = pd.to_datetime(peakdf.index)
    peakdf.index = peakdf.index.strftime('%Y-%m-%d %H:%M:%S.%f')
    
    # savepath = None
    # if path_fig:
    #     savepath = path_fig + f"U_Peakplots/peakfinder_{spec}.jpg"
    
    
    
    if overviewplot:
        overview_plot_ver2(CH4data, df['Latitude'], scp_peaks, spec, N, bg=bg, th=1.1 * bg, savepath=savepath)
    
    print("qui arrivo")
    if peakdf.empty:
        print("No CH4")
    
    return peakdf, None, None

    
    
    
def distance(lon_ref, lat_ref, lon, lat):
        R = 6373000.0 # approximate radius of earth in km
        lat1 = np.deg2rad(lat_ref)
        lon1 = np.deg2rad(lon_ref)
        lat2 = np.deg2rad(lat)
        lon2 = np.deg2rad(lon)
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        distance = R * c
        
        return distance 
    
    
    
        
        
        
      
def add_distance_to_df(df,city,Day=None):
    # Release Coordinates
    release_L_loc1 = (52.233343, -0.437888)
    release_R_loc1 = (51.9201216, 4.5237450)
    release_R_loc2 = (51.9203931, 4.5224917)
    release_R_loc3 = (51.921028, 4.523775) 
    release_U_loc1 = (52.0874256, 5.1647191) #? (maybe 52.08850892,5.16532755 and 52.08860602,5.16401192)
    release_U_loc2 = (52.0885635, 5.1644029) #?
    release_T_loc1 = (43.655007, -79.325254)
    release_T_loc2 = (43.782970, -79.46952)
    
    R_dict = {
        1: release_R_loc1,
        2: release_R_loc2,
        3: release_R_loc3
        }
    U_dict = {
        1: release_U_loc1,
        2: release_U_loc2
        }
    
    df.reset_index(inplace=True,drop=False)
    
    
    if (city == 'Rotterdam'):
        distances = []
        for i in range(len(df)):
            release_loc = R_dict[df.loc[i, 'Loc']]
            x = geodesic(release_loc, (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])).meters
            distances.append(x)
        df['Distance_to_source'] = distances
        
    elif (city == 'Utrecht'):
        distances = []
        for i in range(len(df)):
            release_loc = U_dict[df.loc[i, 'Loc']]
            x = geodesic(release_loc, (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])).meters
            distances.append(x)
        df['Distance_to_source'] = distances
        
    elif city == 'Toronto':
        distances = []
        if Day == 1:
            release_loc = release_T_loc1
        elif Day == 2:
            release_loc = release_T_loc2
        else: print('Toronto: wrong day')         
        for i in range(len(df)):
            x = geodesic(release_loc, (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])).meters
            distances.append(x)           
        df['Distance_to_source'] = distances
        
    elif city == 'London':
        distances = []
        release_loc = release_L_loc1
        for i in range(len(df)):
            x = geodesic(release_loc, (df.loc[i, 'Latitude'], df.loc[i, 'Longitude'])).meters
            distances.append(x)           
        df['Distance_to_source'] = distances
                
    else: print('Wrong city')
    
    df.set_index('Datetime', inplace=True)
    return df
        

def combine_columns(row):
    return (row['Loc'], row['Release_rate'])

        
        
        
        
        
        
        