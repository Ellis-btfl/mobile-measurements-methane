# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 10:11:42 2026

@author: rober
"""

#%% Librareis and Toolbox
import sys
sys.path.append(r'C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\code')
import pandas as pd
import numpy as np
import os
import pickle 
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import folium
from datetime import timedelta
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy import stats


#%% Time-clustering supprting function and Area-Qauntification method


def aggregate_temporal_subclusters(
    gdf,
    cluster_col="cluster",
    smoothed_col="CH4_smoothed",
    area_col="Peak_Area_ppmm",
    time_window_seconds=5
    ):
    """
    Aggregate all peaks that are closer than time_window_seconds in time (default = 5seconds). Keep the highest peak as maximumum peak and sum all areas of the aggregated subpeaks as new peak area.

    Parameters:
        gdf : GeoDataFrame with Datetime index
        cluster_col : space clusterc col
        smoothed_col : smoothed peak col 
        area_col : peak area col
        time_window_seconds : minimum timedistance between peaks to not be merged 

    Returns:
        GeoDataFrame :
        - one row per cluster in time
        - aggregated area 
        - other points marked with area = NaN
    """

    # Copy for safety
    gdf = gdf.copy()

    # sottocluster temp col
    gdf["temp_subcluster"] = -1
    subcluster_id = 0

    # cluster space
    for cl in sorted(gdf[cluster_col].unique()):
        if cl == -1:
            # Ignore "noise"
            continue

        subset = gdf[gdf[cluster_col] == cl].sort_index()

        current_group_start = None

        for idx, row in subset.iterrows():
            if current_group_start is None:
                # New timecluster
                current_group_start = idx
                gdf.loc[idx, "temp_subcluster"] = subcluster_id
            else:
                # Check distance
                if (idx - current_group_start) <= timedelta(seconds=time_window_seconds):
                    # Same timecluster
                    gdf.loc[idx, "temp_subcluster"] = subcluster_id
                else:
                    # New timecluster
                    subcluster_id += 1
                    current_group_start = idx
                    gdf.loc[idx, "temp_subcluster"] = subcluster_id

        subcluster_id += 1  # prepara ID per il prossimo cluster spaziale


    # Aggregate areas and save peakmax
    gdf["area_aggregated"] = None
    gdf["is_representative"] = False

    for (sp_cl, temp_cl), group in gdf.groupby([cluster_col, "temp_subcluster"]):

        if temp_cl == -1:
            continue
        
        

        # Select peakmax to keep
        rep_idx = group[smoothed_col].idxmax()

        # Sum areas in subcluster
        total_area = group[area_col].sum()

        # Add only representative value 
        gdf.loc[rep_idx, "area_aggregated"] = total_area
        gdf.loc[rep_idx, "is_representative"] = True

    return gdf


def area_quantification_method_eq12(peak_cluster):
    
    site_lv_valid_areas = peak_cluster['area_aggregated'].values
    site_lv_valid_areas = pd.to_numeric(site_lv_valid_areas, errors='coerce')
    site_lv_valid_areas = site_lv_valid_areas[~np.isnan(site_lv_valid_areas)]
    n_valid_areas = len(site_lv_valid_areas)
    if n_valid_areas < 1:
        return np.nan, n_valid_areas
    geom_mean = np.mean(np.log(site_lv_valid_areas))
    
    LI_rate_area = np.exp((1.292 * geom_mean) - 2.377)
    
    return LI_rate_area, n_valid_areas

def calculate_weighted_centroid(group):
    weights = group['area_aggregated']
    # calculation of weighter x, y
    weighted_x = np.average(group['x'], weights=weights)
    weighted_y = np.average(group['y'], weights=weights)
    
    # common values for all points of a cluster
    return pd.Series({
        'x': weighted_x,
        'y': weighted_y,
        'rE_Area_slpm': group['rE_Area_slpm'].iloc[0],
        'N_Peaks': group['N_Peaks'].iloc[0],
        'Total_Area_Cluster': weights.sum(),
        # 'n_days': group['n_days'].iloc[0]
    })


def single_peak_quantification_method(CH4_max):
    """
    Calcule le rE pour un seul pic basé sur son aire individuelle (en ppm·m).
    """
    if pd.isna(CH4_max) or CH4_max <= 0:
        return np.nan
    
    # Application directe de la formule de régression log-normale
    log_height = np.log(CH4_max)
    LI_rate_height = np.exp((log_height + 0.988)/0.817)
    
    return LI_rate_height

def height_quantification_method_weller(peak_cluster, ch4_col='peak max'):
    """
    Calcule le débit d'émission (rE) basé sur la moyenne des hauteurs de pics (max excess CH4)
    d'un cluster selon la formule log-log de Weller et al. (2019).
    """
    # 1. Extraction des hauteurs de tous les pics représentatifs du cluster
    ch4_peaks = peak_cluster[ch4_col].dropna().values
    
    # Sécurité : vérifier qu'il y a des valeurs valides et strictement positives
    ch4_peaks = ch4_peaks[ch4_peaks > 0]
    n_peaks = len(ch4_peaks)
    
    if n_peaks < 1:
        return np.nan
    
    # 2. Moyenne arithmétique des logarithmes (équivalent au ln de la moyenne géométrique)
    mean_log_height = np.mean(np.log(ch4_peaks))
    
    # 3. Application de l'équation de régression (Eq. 4 de Weller et al. 2019)
    # ln(max_excess) = -0.988 + 0.817 * ln(emission_rate)
    # => ln(emission_rate) = (ln(max_excess) + 0.988) / 0.817
    log_emission_rate = (mean_log_height + 0.988) / 0.817
    
    # 4. Conversion en débit (passage à l'exponentielle)
    LI_rate_height = np.exp(log_emission_rate)
    
    return LI_rate_height

#%% Read-in peaks from cities

path_to_csv_folder = r'C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\results'
csv_name = 'MySurveys_Amsterdam.csv'

path_survey_ids = r'C:/Users/33ell/OneDrive/Documents/4A GEN/SIRD/standardised data/metadata/analyzer_specs.xlsx'
survey_ids = pd.read_excel(path_survey_ids)

# 1. Lecture du gros CSV unifié avec son double index [Nom de la campagne, Date]
df_all_city_surveys_raw = pd.read_csv(
    os.path.join(path_to_csv_folder, csv_name), 
    sep=';', 
    index_col=[0, 1], 
    parse_dates=[1]
)

# 2. transform the first index (nsurvey name) in a 'survey' colomn
df_all_city_surveys = df_all_city_surveys_raw.reset_index(level=0)
df_all_city_surveys.rename(columns={'level_0': 'survey'}, inplace=True)

# "Utrecht260220.csv" become just "Utrecht260220"
df_all_city_surveys['survey'] = df_all_city_surveys['survey'].str.replace('.csv', '', regex=False)

survey_of_choice = "Utrecht_2026"
print(f"You choose survey {survey_of_choice}")

# 4. be sure name are matching
survey_list_of_choice = survey_ids[survey_ids["survey_id"]==survey_of_choice]["survey_name"].unique()
survey_list_of_choice = [name.replace('.csv', '').replace('.pkl', '').replace('.pickle', '') for name in survey_list_of_choice]

# 5. Filter the chosen city
df_city_survey = df_all_city_surveys[df_all_city_surveys["survey"].isin(survey_list_of_choice)].copy()



#%% Run all peaks and cluster


gdf = gpd.GeoDataFrame(
    df_city_survey,
    geometry=[Point(xy) for xy in zip(df_city_survey.Longitude, df_city_survey.Latitude)],
    crs="EPSG:4326"
).to_crs("EPSG:28992")

# metric coordonnate
gdf["x"] = gdf.geometry.x
gdf["y"] = gdf.geometry.y


coords = gdf[["x", "y"]].values

# eps = max radius in Meters per cluster

db = DBSCAN(eps=50, min_samples=2).fit(coords) # 50 meter cluster

gdf["cluster"] = db.labels_

m = folium.Map(
    location=[df_city_survey.Latitude.mean(),
              df_city_survey.Longitude.mean()],
    zoom_start = 12
)

for _, row in gdf.iterrows():
    color = "red" if row["cluster"] == -1 else f"#{row['cluster']:02x}00aa"
    folium.CircleMarker(
        location=[row.Latitude, row.Longitude],
        radius=4,
        color=color,
        fill=True,
        fill_opacity=0.8
        ).add_to(m)
       


#%% Merge temporale di picchi troppo ravvicinati

time_aggr_window = 5 # seconds
gdf_agg = aggregate_temporal_subclusters(gdf, "cluster", "Smoothed_CH4", "Peak_Area_ppmm", time_window_seconds=time_aggr_window)

# # #%% miminimum of 2 different days filter

# # 1. Date only extraction
# gdf_agg['Date_Only'] = gdf_agg.index.date

# # 2. Number of unique day per cluster (except noise)
# cluster_counts = gdf_agg[gdf_agg['cluster'] != -1].groupby('cluster')['Date_Only'].nunique()
# gdf_agg['n_days'] = gdf_agg['cluster'].map(cluster_counts)

# # List of clusters to delete
# clusters_to_delete = cluster_counts[cluster_counts < 2].index

# # 3. Put all deleted clusters in noise (clustyer -1)
# if not clusters_to_delete.empty:
#     gdf_agg.loc[gdf_agg['cluster'].isin(clusters_to_delete), 'cluster'] = -1

# print(f"Clusters supprimés car vus sur un seul jour : {list(clusters_to_delete)}")

#%% Graphical oputput
# Scegli la colormap
cmap = cm.get_cmap("viridis")

# Estrai le aree valide (dei punti rappresentativi)
areas = gdf_agg.loc[gdf_agg["is_representative"], "area_aggregated"].values

# Evita problemi con cluster vuoti
if len(areas) == 0:
    raise ValueError("Nessun picco rappresentativo trovato per generare la colormap.")

# Normalizzazione 0-1 sulle aree
norm = colors.Normalize(vmin=np.min(areas), vmax=np.max(areas))

# Colore fisso per il noise
noise_color = "#aaaaaa" # gray


m2 = folium.Map(
    location=[df_city_survey.Latitude.mean(),
              df_city_survey.Longitude.mean()],
    zoom_start = 12
)


for idx, row in gdf_agg.iterrows():

    if row["cluster"] == -1:
        color = noise_color
    else:
        if row["is_representative"]:
            val = row["area_aggregated"]
        else:
            val = row["Peak_Area_ppmm"]
        
        if np.isnan(val):
            color = '#cccccc' # fallback for no-area 
        else:
            color = colors.to_hex(cmap(norm(val)))

    opac = 0.8 if row["is_representative"] else 0.2

    popup_text = (
        f"<b>Datetime:</b> {idx}<br>"
        f"<b>Area:</b> {row['area_aggregated']}"
    )

    folium.CircleMarker(
        location=[row.Latitude, row.Longitude],
        radius=4,
        color=color,
        fill=True,
        fill_opacity=opac,
        popup=folium.Popup(popup_text, max_width=250)
    ).add_to(m2)

    
    
# Save the clustered map
clustered_map_path = os.path.join(r'C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\results\cities\maps', survey_of_choice+"_Amsterdam.html")
m2.save(clustered_map_path)


#%%% Emission rate quantification

gdf_agg[["rE_Area_slpm", "N_Peaks", "rE_Height_Peak_max", "rE_Area_Peak_max", "rE_10%"]] = np.nan, np.nan, np.nan, np.nan, np.nan
mask_quality = (gdf["peak max"] > (0.05 * gdf["bg_aeris"]))
gdf_high_quality = gdf[mask_quality].copy()


for space_cluster in gdf_agg["cluster"].unique():
    
    gdf_cluster = gdf_agg[gdf_agg["cluster"] == space_cluster].copy()
    print('lenght',len(gdf_cluster))
    gdf_cluster = gdf_cluster[(gdf_cluster['is_representative']) & (gdf_cluster["Peak_Length_m"] > 0)].copy()
    print('lenght',len(gdf_cluster))
    
    if gdf_cluster.empty:
        print(f"Cluster {space_cluster} ignoré (aucun pic représentatif valide).")
        continue
    
    peak_idxs = gdf_cluster.index
    
    rE_estimate, n_transects = area_quantification_method_eq12(gdf_cluster)
    
    # quantification based on the peak of maximum enhancement (with his height and his area alone)  
    idx_max_ch4 = gdf_cluster['peak max'].idxmax()
    row_max_ch4 = gdf_cluster.loc[idx_max_ch4]
    CH4_max = row_max_ch4['peak max']
    rEAPm = row_max_ch4['rE_Peak_slpm']
    rEHPm = single_peak_quantification_method(CH4_max)
    rE_geometric_mean_peak_max = height_quantification_method_weller(gdf_cluster, ch4_col='peak max')
    
    gdf_cluster_hq = gdf_high_quality[gdf_high_quality["cluster"] == space_cluster].copy()
    gdf_cluster_hq = aggregate_temporal_subclusters(gdf_cluster_hq, "cluster", "Smoothed_CH4", "Peak_Area_ppmm", time_window_seconds=time_aggr_window)
    rE_quality, n_transects_bis = area_quantification_method_eq12(gdf_cluster_hq)
    if n_transects_bis < 2:
        rE_quality = np.nan
       
    gdf_agg.loc[peak_idxs, "rE_Area_slpm"] = rE_estimate
    gdf_agg.loc[peak_idxs, "N_Peaks"] = n_transects
    gdf_agg.loc[peak_idxs, "rE_Area_Peak_max"] = rEAPm
    gdf_agg.loc[peak_idxs, "rE_Height_Peak_max"] = rEHPm
    gdf_agg.loc[peak_idxs, "rE_5%"] = rE_quality
    gdf_agg.loc[peak_idxs, "rE_Height_geometric_mean"] = rE_geometric_mean_peak_max
    gdf_agg.loc[peak_idxs, "N_5%"] = n_transects_bis
    
    print('cluster: ', space_cluster, 'N: ', n_transects, 'LR: ', rE_estimate)
#%% Display Emission Distribution

rE_data_dt = (
    gdf_agg.query("cluster != -1 and Peak_Length_m > 0 and is_representative")
    .groupby("cluster")[["Datetime_UTC", "rE_Area_slpm"]]
    .first()
)

rE_data = rE_data_dt["rE_Area_slpm"]

tetas = stats.lognorm.fit(rE_data)

plt.figure()
plt.hist(rE_data, bins=20)
plt.xlabel("rE estimate (slpm)")
plt.ylabel("Count (-)")

plt.figure()
plt.plot(np.linspace(min(rE_data), max(rE_data), 1000), 
         stats.lognorm.pdf(np.linspace(min(rE_data), max(rE_data), 1000), *tetas),
         label="pdf")

plt.xlabel("rE estimate (slpm)")
plt.ylabel("Probability Density")

#%% Export for QGIS

gdf_agg_repr = gdf_agg[(gdf_agg['is_representative']) & (gdf_agg['cluster'] != -1)].copy()

gdf_agg_repr["area_aggregated"] = pd.to_numeric(gdf_agg_repr["area_aggregated"], errors="coerce")
gdf_agg_repr["rE_Area_slpm"] = pd.to_numeric(gdf_agg_repr["rE_Area_slpm"], errors="coerce")
gdf_agg_repr["N_Peaks"] = pd.to_numeric(gdf_agg_repr["N_Peaks"], errors="coerce")
gdf_agg_repr["rE_Area_Peak_max"] = pd.to_numeric(gdf_agg_repr["rE_Area_Peak_max"], errors="coerce")
gdf_agg_repr["rE_Height_Peak_max"] = pd.to_numeric(gdf_agg_repr["rE_Height_Peak_max"], errors="coerce")
gdf_agg_repr["rE_5%"] = pd.to_numeric(gdf_agg_repr["rE_5%"], errors="coerce")
gdf_agg_repr["rE_Height_geometric_mean"] = pd.to_numeric(gdf_agg_repr["rE_Height_geometric_mean"], errors="coerce")
gdf_agg_repr["N_5%"] = pd.to_numeric(gdf_agg_repr["N_5%"], errors="coerce")


# Barycenter

# filtering of noises and be sure areas are numbers
df_valid_clusters = gdf_agg[gdf_agg['cluster'] != -1].copy()
df_valid_clusters['area_aggregated'] = pd.to_numeric(df_valid_clusters['area_aggregated'], errors='coerce')

# deleting lines with NaN or 0
df_valid_clusters = df_valid_clusters.dropna(subset=['area_aggregated'])
df_valid_clusters = df_valid_clusters[df_valid_clusters['area_aggregated'] > 0]


# Application of the calculation for each cluster
cluster_barycenters = df_valid_clusters.groupby('cluster').apply(calculate_weighted_centroid, include_groups=False).reset_index()
gdf_barycenters = gpd.GeoDataFrame(cluster_barycenters,geometry=gpd.points_from_xy(cluster_barycenters.x, cluster_barycenters.y),crs="EPSG:28992")


qgis_filename = os.path.join(r'C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\results\cities\maps', survey_of_choice+"_Amsterdam.gpkg")
gdf_agg_repr.to_file(qgis_filename, layer="CH4_clusters", driver="GPKG", index=True)
gdf_barycenters.to_file(qgis_filename, layer="CH4_cluster_centroids", driver="GPKG", index=False)
 
#%% Display Emission Distribution Max > 10% bg
rE_data_dt_newmax = (
    gdf_agg.query("cluster != -1 and Peak_Length_m > 0 and is_representative and Peakmax_between_bases_raw > 0.2")
    .groupby("cluster")[["Datetime_UTC", "rE_Area_slpm"]]
    .first()
)

rE_data_newmax = rE_data_dt_newmax["rE_Area_slpm"]

if len(rE_data_newmax) > 0:
    tetas_newmax = stats.lognorm.fit(rE_data_newmax)

    plt.figure()
    plt.hist(rE_data_newmax, bins=20)
    plt.xlabel("rE estimate (slpm)")
    plt.ylabel("Count (-)")
    plt.title("Histogramme rE (Peakmax > 0.2)")

    plt.figure()
    x_axis = np.linspace(min(rE_data_newmax), max(rE_data_newmax), 1000)
    plt.plot(x_axis, stats.lognorm.pdf(x_axis, *tetas_newmax), label="pdf")
    plt.xlabel("rE estimate (slpm)")
    plt.ylabel("Probability Density")
    plt.title("PDF Log-normale rE (Peakmax > 0.2)")
    plt.show()
else:
    print("Info : Aucun pic ne satisfait la condition Peakmax_between_bases_raw > 0.2 pour ce survol.")
