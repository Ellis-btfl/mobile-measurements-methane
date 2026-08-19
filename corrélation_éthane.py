import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from datetime import timedelta, date
import glob
import os

# --- 1. CONFIGURATION DES DOSSIERS ET PARAMÈTRES ---
# gpkg_path1 = "C:/Users/33ell/OneDrive/Documents/4A GEN/SIRD/results/cities/maps/Utrecht_2026_bike_2jours.gpkg"
gpkg_path1 = "C:/Users/33ell/OneDrive/Documents/4A GEN/SIRD/results/cities/maps/Utrecht_2026_Amsterdam.gpkg"
# gpkg_path3 = "C:/Users/33ell/OneDrive/Documents/4A GEN/SIRD/results/cities/maps/Utrecht_2026_van.gpkg"
dossier_aeris = r"C:\Users\33ell\OneDrive\Documents\4A GEN\SIRD\raw data\bike\ts les txt\bike"

FENETRE_SECONDES = 30  # Fenêtre de temps autour du pic (+/- 30s)

# --- 2. DICTIONNAIRE DES DÉCALAGES PAR DATE ---
DECALAGES_JOURNALIERS = {
    date(2026, 2, 20): pd.Timedelta(hours=2, seconds=35),
    date(2026, 3, 4):  pd.Timedelta(hours=2, seconds=35),
    date(2026, 3, 9):  pd.Timedelta(hours=2, seconds=35),
    date(2026, 3, 23): pd.Timedelta(hours=0, seconds=-47),
    date(2026, 3, 24): pd.Timedelta(hours=0, seconds=-43),
    date(2026, 3, 26): pd.Timedelta(hours=0, seconds=-43),
    date(2026, 4, 2):  pd.Timedelta(hours=0, seconds=-43),
    date(2026, 4, 10): pd.Timedelta(hours=0, seconds=-51),
    date(2026, 4, 14): pd.Timedelta(hours=0, seconds=-51),
    date(2026, 4, 17): pd.Timedelta(hours=0, seconds=-43),
    date(2026, 6, 17): pd.Timedelta(hours=0, seconds=-7),
    date(2026, 6, 18): pd.Timedelta(hours=0, seconds=-8),
    date(2026, 6, 23): pd.Timedelta(hours=0, seconds=-7),
    date(2026, 6, 29): pd.Timedelta(hours=0, seconds=-7),
    date(2026, 6, 30): pd.Timedelta(hours=0, seconds=-7),
    date(2026, 7, 2):  pd.Timedelta(hours=0, seconds=-7),
}

# --- 3. CHARGEMENT ET CORRECTION INDIVIDUELLE DES LOGS AERIS ---
fichiers_txt = glob.glob(os.path.join(dossier_aeris, "*.txt"))
print(f"Trouvé {len(fichiers_txt)} fichiers .txt d'Aeris.")

liste_df_aeris = []
for f in fichiers_txt:
    try:
        df_temp = pd.read_csv(f, sep=",", skipinitialspace=True)
        if 'Time Stamp' in df_temp.columns and 'CH4 (ppm)' in df_temp.columns:
            # Conversion en datetime pour ce fichier spécifique
            df_temp['datetime'] = pd.to_datetime(df_temp['Time Stamp'])
            
            # Extraction de la date du fichier (basée sur la première ligne)
            date_du_fichier = df_temp['datetime'].dt.date.iloc[0]
            
            # Application du décalage propre à cette date
            if date_du_fichier in DECALAGES_JOURNALIERS:
                decalage = DECALAGES_JOURNALIERS[date_du_fichier]
                df_temp['datetime'] = df_temp['datetime'] - decalage
                print(f" -> {os.path.basename(f)} : Date {date_du_fichier} recalée de {decalage}")
            else:
                print(f" ⚠️ {os.path.basename(f)} : Aucune correction configurée pour le {date_du_fichier}")
                
            liste_df_aeris.append(df_temp)
    except Exception as e:
        print(f" -> Erreur lors de la lecture de {os.path.basename(f)} : {e}")

if not liste_df_aeris:
    raise ValueError("Aucun fichier Aeris valide n'a pu être chargé. Vérifiez le dossier.")

print("Fusion de toutes les données Aeris...")
aeris_df = pd.concat(liste_df_aeris, ignore_index=True)

# Conversion des unités et tri chronologique global
aeris_df['CH4_ppb'] = aeris_df['CH4 (ppm)'] * 1000.0  # Conversion ppm -> ppb
aeris_df['C2H6_ppb'] = aeris_df['C2H6 (ppb)']
aeris_df = aeris_df.sort_values('datetime').reset_index(drop=True)

# --- 4. CHARGEMENT DES PICS (GPKG) ---
print("Chargement du fichier de pics GPKG...")
gdf1 = gpd.read_file(gpkg_path1, layer='CH4_clusters')
# gdf2 = gpd.read_file(gpkg_path2, layer='CH4_clusters')
# gdf3 = gpd.read_file(gpkg_path3, layer='CH4_clusters')
# pics_gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2, gdf3], ignore_index=True), crs=gdf1.crs)
pics_gdf = gdf1

# Détection automatique de la colonne temporelle
noms_possibles_temps = ['datetime_utc', 'time stamp', 'timestamp', 'time_stamp', 'date', 'time', 'datetime', 'date_time']
colonne_temps_trouvee = None

for col in pics_gdf.columns:
    if col.lower() in noms_possibles_temps:
        colonne_temps_trouvee = col
        break

if colonne_temps_trouvee is not None:
    print(f" -> Colonne temporelle détectée dans le GPKG : '{colonne_temps_trouvee}'")
    pics_gdf['datetime'] = pd.to_datetime(pics_gdf[colonne_temps_trouvee])
else:
    print("\n[ERREUR] Impossible de trouver la colonne de temps.")
    print("Colonnes disponibles :", pics_gdf.columns.tolist())
    raise KeyError("Veuillez vérifier le nom de la colonne temporelle dans le GPKG.")

# --- 5. ANALYSE STATISTIQUE BINAIRE PAR COEFFICIENT DE PEARSON ---
results = []
print("Analyse des corrélations en cours...")

for idx, pic in pics_gdf.iterrows():
    t_pic = pic['datetime']
    
    t_start = t_pic - timedelta(seconds=FENETRE_SECONDES)
    t_end = t_pic + timedelta(seconds=FENETRE_SECONDES)
    
    zone_plume = aeris_df[(aeris_df['datetime'] >= t_start) & (aeris_df['datetime'] <= t_end)]
    
    # Calcul des hauteurs de pics (Delta)
    if len(zone_plume) > 5:
        delta_CH4 = zone_plume['CH4 (ppm)'].max() - zone_plume['CH4 (ppm)'].min()
        delta_C2H6 = zone_plume['C2H6 (ppb)'].max() - zone_plume['C2H6 (ppb)'].min()
    else:
        delta_CH4 = np.nan
        delta_C2H6 = np.nan

    # Validation du pic de méthane (Seuil adouci à 0.005 ppm soit 5 ppb pour inclure les petits pics)
    if len(zone_plume) > 5 and delta_CH4 > 0.04:
        x = zone_plume['CH4_ppb'].values
        y = zone_plume['C2H6_ppb'].values
        
        # Sécurité pour éviter la division par zéro si la variance est nulle
        if np.std(x) > 0 and np.std(y) > 0:
            r_coeff, _ = pearsonr(x, y)
            
            if r_coeff >= 0.6:
                statut = "Pics corrélés (R >= 0.6)"
            else:
                statut = "Pics non corrélés (R < 0.6)"
        else:
            r_coeff = np.nan
            statut = "Variance nulle (signal plat)"
            
        results.append({
            'pic_id': idx,
            'time': t_pic,
            'R_correlation': r_coeff,
            'delta_CH4_ppm': delta_CH4,
            'delta_C2H6_ppb': delta_C2H6,
            'statut_correlation': statut,
            'geometry': pic['geometry']
        })
    else:
        # Zone sans pic significatif
        results.append({
            'pic_id': idx,
            'time': t_pic,
            'R_correlation': np.nan,
            'delta_CH4_ppm': delta_CH4,
            'delta_C2H6_ppb': delta_C2H6,
            'statut_correlation': "Zone de bruit / Pas de pic CH4 franc",
            'geometry': pic['geometry']
        })

# --- 6. EXPORT DES RÉSULTATS ---
output_gdf = gpd.GeoDataFrame(results, geometry='geometry', crs=pics_gdf.crs)

chemin_sortie = os.path.join(dossier_aeris, "pics_methane_analyses_multi_jours_AMS.gpkg")
output_gdf.to_file(chemin_sortie, driver="GPKG")

print("\n=== ANALYSE TERMINÉE ===")
print(f"Le fichier a été créé avec succès à l'emplacement suivant :\n{chemin_sortie}")
print(output_gdf[['time', 'delta_CH4_ppm', 'delta_C2H6_ppb', 'R_correlation', 'statut_correlation']].head(10))