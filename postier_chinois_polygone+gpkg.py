import os
import geopandas as gpd
import gpxpy.gpx
import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon, box

# ==========================================
# 1. POLYGONE ET FICHIERS SOURCE
# ==========================================
# Coordonnées (longitude, latitude, altitude) depuis geojson.io
zone_points_3d = [
          [
            5.124176,
            52.081548,
            3.39096
          ],
          [
            5.125493,
            52.081471,
            5.83633
          ],
          [
            5.126676,
            52.08314,
            9.72171
          ],
          [
            5.128578,
            52.084674,
            6.8027
          ],
          [
            5.129758,
            52.085337,
            9.55018
          ],
          [
            5.128536,
            52.08793,
            8.97719
          ],
          [
            5.129416,
            52.089336,
            9.35422
          ],
          [
            5.127957,
            52.090127,
            7.86903
          ],
          [
            5.127088,
            52.092282,
            9.49899
          ],
          [
            5.128203,
            52.093541,
            5.27923
          ],
          [
            5.126498,
            52.094417,
            9.68425
          ],
          [
            5.126034,
            52.096164,
            6.3569
          ],
          [
            5.126841,
            52.097613,
            3.10178
          ],
          [
            5.126347,
            52.097883,
            5.08935
          ],
          [
            5.124405,
            52.09754,
            6.67687
          ],
          [
            5.122702,
            52.097002,
            7.44947
          ],
          [
            5.121006,
            52.097292,
            7.7992
          ],
          [
            5.120384,
            52.097134,
            8.79599
          ],
          [
            5.119762,
            52.096317,
            9.93488
          ],
          [
            5.116449,
            52.096551,
            10.46598
          ],
          [
            5.115912,
            52.096396,
            10.18833
          ],
          [
            5.114565,
            52.096415,
            7.40384
          ],
          [
            5.114447,
            52.096033,
            7.62167
          ],
          [
            5.114211,
            52.095585,
            12.88449
          ],
          [
            5.114127,
            52.094643,
            10.72984
          ],
          [
            5.11431,
            52.093741,
            14.26114
          ],
          [
            5.114449,
            52.093075,
            11.17161
          ],
          [
            5.113245,
            52.092665,
            11.47279
          ],
          [
            5.113653,
            52.091373,
            16.98518
          ],
          [
            5.114707,
            52.089818,
            13.86173
          ],
          [
            5.118047,
            52.086123,
            11.99715
          ],
          [
            5.120062,
            52.083793,
            8.94967
          ],
          [
            5.121028,
            52.082,
            9.09365
          ],
          [
            5.121352,
            52.081043,
            8.79281
          ],
          [
            5.12198,
            52.080819,
            8.10656
          ],
          [
            5.123058,
            52.081132,
            5.8278
          ],
          [
            5.122999,
            52.081105,
            6.30813
          ],
          [
            5.123664,
            52.081303,
            2.82001
          ],
          [
            5.124176,
            52.081548,
            3.39096
          ]
        ]

# Isolation de (lon, lat) pour la géométrie 2D
zone_points_2d = [(pt[0], pt[1]) for pt in zone_points_3d]
polygon = Polygon(zone_points_2d)

gpkg_filename = r"C:/Users/33ell/OneDrive/Documents/4A GEN/SIRD/gps/stedin_reseau.gpkg"
output_folder = r"C:/Users/33ell/OneDrive/Documents/4A GEN/SIRD/gps"
os.makedirs(output_folder, exist_ok=True)
output_filename = os.path.join(output_folder, "Circuit_Gaz_Polygone_binnenstad.gpx")

# ==========================================
# 2. TÉLÉCHARGEMENT OSM ET CHARGEMENT DU GPKG
# ==========================================
print("1. Téléchargement du réseau OSM dans le polygone...")
G = ox.graph_from_polygon(polygon, network_type="bike", simplify=False)
G_un = nx.MultiGraph(G.to_undirected())

print(f"2. Lecture du réseau de gaz ({gpkg_filename})...")
stedin_gdf = gpd.read_file(gpkg_filename)

# Extraction de la carte OSM en GeoDataFrame
nodes, edges = ox.graph_to_gdfs(G_un, nodes=True, edges=True)

# ==========================================
# 3. INTERSECTION SPATIALE (OSM x STEDIN)
# ==========================================
print("3. Filtrage des rues longeant le réseau de gaz...")
bbox = edges.total_bounds

# Alignement du CRS / Projection géographique pour la boîte englobante
if stedin_gdf.crs != edges.crs:
    bbox_geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=edges.crs).to_crs(stedin_gdf.crs).total_bounds
    minx, miny, maxx, maxy = bbox_geom
else:
    minx, miny, maxx, maxy = bbox

# Découpage rapide du GPKG sur la bounding box élargie du polygone
marge = 0.002 if edges.crs.is_geographic else 200
stedin_local = stedin_gdf.cx[minx - marge : maxx + marge, miny - marge : maxy + marge]

# Projection en mètres (EPSG:28992 - Amersfoort / RD New pour les Pays-Bas) pour calculer une zone tampon exacte
edges_rd = edges.to_crs(epsg=28992)
stedin_rd = stedin_local.to_crs(epsg=28992)

# Zone tampon de 20m autour des tuyaux de gaz
stedin_buffered_gdf = gpd.GeoDataFrame(geometry=stedin_rd.geometry.buffer(20), crs=stedin_rd.crs)

# Détection des rues qui croisent/longent les canalisations
edges_intersected = gpd.sjoin(edges_rd, stedin_buffered_gdf, how="inner", predicate="intersects")
edges_filtered = edges.loc[edges.index.isin(edges_intersected.index)]

# Suppression des segments d'arêtes sans canalisation de gaz
edges_to_remove = set(edges.index) - set(edges_filtered.index)
for u, v, k in edges_to_remove:
    if G_un.has_edge(u, v, k):
        G_un.remove_edge(u, v, k)

# Conservation de la composante connexe principale
largest_cc = max(nx.connected_components(G_un), key=len)
G_sub = nx.MultiGraph(G_un.subgraph(largest_cc))

# ==========================================
# 4. ÉLAGAGE ITÉRATIF DES IMPASSES
# ==========================================
print("4. Élimination des impasses...")
anciennes_routes = len(G_sub.edges())

while True:
    impasses = [node for node, degree in G_sub.degree() if degree == 1]
    if not impasses:
        break
    G_sub.remove_nodes_from(impasses)

largest_cc = max(nx.connected_components(G_sub), key=len)
G_sub = nx.MultiGraph(G_sub.subgraph(largest_cc))

print(f"-> Élagage réussi ! {anciennes_routes - len(G_sub.edges())} segments d'impasses supprimés.")

# ==========================================
# 5. OPTIMISATION (POSTIER CHINOIS OPTIMAL)
# ==========================================
print("5. Calcul du couplage optimal des sommets impairs...")
odd_nodes = [v for v, d in G_sub.degree() if d % 2 == 1]

if odd_nodes:
    pair_weights = {}
    for i, u in enumerate(odd_nodes):
        for v in odd_nodes[i + 1 :]:
            try:
                length = nx.shortest_path_length(G_sub, u, v, weight="length")
                pair_weights[(u, v)] = length
            except nx.NetworkXNoPath:
                continue

    K = nx.Graph()
    for (u, v), w in pair_weights.items():
        K.add_edge(u, v, weight=w)

    matching = nx.min_weight_matching(K, weight="weight")

    for u, v in matching:
        path = nx.shortest_path(G_sub, u, v, weight="length")
        for i in range(len(path) - 1):
            u_node, v_node = path[i], path[i + 1]
            edge_data = list(G_sub[u_node][v_node].values())[0]
            G_sub.add_edge(u_node, v_node, **edge_data)

# ==========================================
# 6. DÉPART ET CIRCUIT EULÉRIEN
# ==========================================
# Sélection de l'intersection la plus proche du premier point du polygone comme départ
start_node = ox.nearest_nodes(G_sub, X=zone_points_2d[0][0], Y=zone_points_2d[0][1])

circuit = list(nx.eulerian_circuit(G_sub, source=start_node))

# ==========================================
# 7. EXPORTATION GPX
# ==========================================
print("6. Génération du fichier GPX...")
gpx = gpxpy.gpx.GPX()
track = gpxpy.gpx.GPXTrack(name="Circuit_Gaz_Polygone_binnenstad")
gpx.tracks.append(track)
segment = gpxpy.gpx.GPXTrackSegment()
track.segments.append(segment)

for u, v, *key in circuit:
    k = key[0] if key else 0
    edge_data = G_sub[u][v][k]

    if "geometry" in edge_data:
        coords = list(edge_data["geometry"].coords)
        if coords[0] != (G_sub.nodes[u]["x"], G_sub.nodes[u]["y"]):
            coords = coords[::-1]
        for x, y in coords[:-1]:
            segment.points.append(gpxpy.gpx.GPXTrackPoint(y, x))
    else:
        node = G_sub.nodes[u]
        segment.points.append(gpxpy.gpx.GPXTrackPoint(node["y"], node["x"]))

last_node = G_sub.nodes[circuit[-1][1]]
segment.points.append(gpxpy.gpx.GPXTrackPoint(last_node["y"], last_node["x"]))

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(gpx.to_xml())

print(f"\n✅ Terminé avec succès ! Trace disponible ici : {output_filename}")