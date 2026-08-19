import os
import gpxpy.gpx
import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon

# ==========================================
# 1. Polygon
# ==========================================
# I create my polygon on geojson.io and copypaste the coordinates
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
            5.115123,
            52.09691,
            5.65891
          ],
          [
            5.113685,
            52.096699,
            6.102
          ],
          [
            5.109973,
            52.09542,
            2.98607
          ],
          [
            5.110123,
            52.09488,
            4.5614
          ],
          [
            5.111904,
            52.092797,
            8.86055
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

# Extraction of(lon, lat) for Shapely, because geojson.io also give altitude
zone_points_2d = [(pt[0], pt[1]) for pt in zone_points_3d]
polygon = Polygon(zone_points_2d)

output_folder = r"C:/Users/33ell/OneDrive/Documents/4A GEN/SIRD/gps"
os.makedirs(output_folder, exist_ok=True)
output_filename = os.path.join(output_folder, "Circuit_binnenstad_ttes_rues.gpx")

# ==========================================
# 2. download the map and create the graph
# ==========================================
print("Téléchargement de la zone...")
G = ox.graph_from_polygon(polygon, network_type="bike", simplify=False)
G = nx.MultiGraph(G.to_undirected())

# Élagage itératif des impasses (degré 1)
print("Élimination des impasses...")
anciennes_routes = len(G.edges())
while True:
    impasses = [node for node, degree in G.degree() if degree == 1]
    if not impasses:
        break
    G.remove_nodes_from(impasses)

# Preservation of the largest connected component
largest_cc = max(nx.connected_components(G), key=len)
G = G.subgraph(largest_cc).copy()
print(
    f"-> Élagage réussi ! {anciennes_routes - len(G.edges())} segments d'impasses supprimés."
)

# ==========================================
# 3. OPTIMISATION (POSTIER CHINOIS OPTIMAL)
# ==========================================
print("Rendre le graphe eulérien...")
odd_nodes = [v for v, d in G.degree() if d % 2 == 1]

if odd_nodes:
    # 1. Calculating the shortest paths between ALL the odd-numbered vertices
    pair_weights = {}
    for i, u in enumerate(odd_nodes):
        for v in odd_nodes[i + 1 :]:
            try:
                length = nx.shortest_path_length(G, u, v, weight="length")
                pair_weights[(u, v)] = length
            except nx.NetworkXNoPath:
                continue

    # 2. Complete graph of odd vertices with inverse weights (for min_weight_matching)
    K = nx.Graph()
    for (u, v), w in pair_weights.items():
        K.add_edge(u, v, weight=w)

    # 3.Perfect coupling with minimum weight

    matching = nx.min_weight_matching(K, weight="weight")

    # 4. Duplicate the edges in the original graph
    for u, v in matching:
        path = nx.shortest_path(G, u, v, weight="length")
        for i in range(len(path) - 1):
            u_node, v_node = path[i], path[i + 1]
            # Récupérer les attributs de l'arête existante
            edge_data = list(G[u_node][v_node].values())[0]
            G.add_edge(u_node, v_node, **edge_data)

# ==========================================
# 4. CIRCUIT GENERATION
# ==========================================
gare_coord = (52.081604, 5.124065)  # (lat, lon)
start_node = ox.nearest_nodes(G, X=gare_coord[1], Y=gare_coord[0])

circuit = list(nx.eulerian_circuit(G, source=start_node))

# ==========================================
# 5. EXPORT TO GPX
# ==========================================
gpx = gpxpy.gpx.GPX()
track = gpxpy.gpx.GPXTrack(name="Circuit_binnenstad_ttes_rues")
gpx.tracks.append(track)
segment = gpxpy.gpx.GPXTrackSegment()
track.segments.append(segment)

for u, v, *key in circuit:
    k = key[0] if key else 0
    edge_data = G[u][v][k]

    # If OpenStreetMap has the exact geometry of the street, we use it
    if "geometry" in edge_data:
        coords = list(edge_data["geometry"].coords)
        # Check the direction of the route
        if coords[0] != (G.nodes[u]["x"], G.nodes[u]["y"]):
            coords = coords[::-1]
        for x, y in coords[:-1]:
            segment.points.append(gpxpy.gpx.GPXTrackPoint(y, x))
    else:
        node = G.nodes[u]
        segment.points.append(gpxpy.gpx.GPXTrackPoint(node["y"], node["x"]))

# Adding the very last point to bring things full circle
last_node = G.nodes[circuit[-1][1]]
segment.points.append(gpxpy.gpx.GPXTrackPoint(last_node["y"], last_node["x"]))

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(gpx.to_xml())

print(f"✅ Terminé ! Circuit généré : {output_filename}")