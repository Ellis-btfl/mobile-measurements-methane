# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:05:52 2025

@author: rober
"""

import pandas as pd
import numpy as np

def filter_peaks_with_ethane(DSO_peaks, Ethane_peaks, time_window=10):
    """
    Filters DSO_peaks to retain only those associated with Ethane peaks.
    
    Parameters:
    - DSO_peaks: DataFrame of methane peaks
    - Ethane_peaks: DataFrame of ethane peaks
    - time_window: time tolerance in seconds to consider peaks as associated
    
    Returns:
    - filtered_peaks: DataFrame of CH₄ peaks associated with C₂H₆ peaks
    """
    # Convert indices to datetime if needed
    DSO_peaks.index = pd.to_datetime(DSO_peaks.index)
    Ethane_peaks.index = pd.to_datetime(Ethane_peaks.index)

    associated_indices = []

    for ch4_time in DSO_peaks.index:
        # Check if any ethane peak is within the time window
        if any(abs((ethane_time - ch4_time).total_seconds()) <= time_window for ethane_time in Ethane_peaks.index):
            associated_indices.append(ch4_time)

    filtered_peaks = DSO_peaks.loc[associated_indices]
    return filtered_peaks