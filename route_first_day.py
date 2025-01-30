import pandas as pd
import numpy as np
from math import radians, cos, sin, sqrt, atan2
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
import requests
import geopandas as gpd
from shapely import wkt

# Load data files
data_file = "daily_hotspot_predictions.csv"
shapefile_path = "dataset/UTRGRID100/UTRGRID100WGS84.shp"

# Load actual hotspot data
print("Loading actual hotspot data...")
grid_data = gpd.read_file(shapefile_path)
grid_data = grid_data.to_crs(epsg=4326)

# Load predicted data but filter only actual hotspots
print("Loading and filtering actual hotspot data...")
actual_data = pd.read_csv(data_file, low_memory=False)
actual_data['datum'] = pd.to_datetime(actual_data['datum'], errors='coerce')
actual_data['geometry'] = actual_data['geometry'].apply(wkt.loads)
actual_data = gpd.GeoDataFrame(actual_data, geometry='geometry')
actual_data.set_crs(epsg=4326, inplace=True)

# Merge actual hotspot data
data = actual_data.merge(grid_data, on="UTRGRID100", how="left")

# Filter for May 18, 2017 (one-day optimization)
selected_date = "2017-05-18"
filtered_data = data[(data['datum'] == selected_date) & (data['hotspot_category'] != "<5 min")]

# Extract coordinates of actual hotspots
locations = filtered_data[['center_latitude', 'center_longitude']].dropna().values

# Haversine function to compute distances
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

# Create the distance matrix
def create_distance_matrix(locations):
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations))
    for i in range(num_locations):
        for j in range(num_locations):
            if i != j:
                distance_matrix[i][j] = haversine(
                    locations[i][0], locations[i][1],
                    locations[j][0], locations[j][1]
                )
    return distance_matrix

# Solve the TSP using OR-Tools
def solve_tsp(distance_matrix):
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node] * 1000)  # Convert to meters

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set the search parameters for optimal solution
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 30  # Allow up to 30s for optimization

    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)

    # Extract the route
    if solution:
        route = []
        index = routing.Start(0)
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))  # Return to the start
        return route
    else:
        return None

# Fetch road-based route from OSRM
def get_osrm_route(locations):
    coordinates = ";".join([f"{lon},{lat}" for lat, lon in locations])
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{coordinates}?overview=full&geometries=geojson"

    response = requests.get(osrm_url)
    if response.status_code == 200:
        route_data = response.json()
        return route_data['routes'][0]['geometry']
    else:
        print("Failed to fetch the route from OSRM.")
        return None

# Visualize the road-based route with Folium
def visualize_route_with_roads(locations, route_geojson):
    map_center = [locations[0][0], locations[0][1]]
    route_map = folium.Map(location=map_center, zoom_start=13, tiles="CartoDB Positron")  # White background

    # Add markers for all locations
    for idx, loc in enumerate(locations):
        folium.Marker(
            location=[loc[0], loc[1]],
            popup=f"Hotspot {idx + 1}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(route_map)

    # Add the road-based route as a GeoJSON line
    if route_geojson:
        folium.GeoJson(
            route_geojson,
            name="Optimal Route",
            style_function=lambda x: {"color": "red", "weight": 4, "opacity": 0.8}
        ).add_to(route_map)

    # Save the map
    route_map.save("optimal_route_actual_hotspots.html")
    print("✅ Map saved as 'optimal_route_actual_hotspots.html'. Open this file to view the map.")

# Main execution
if len(locations) > 1:
    print("Computing optimal route for actual hotspots...")
    distance_matrix = create_distance_matrix(locations)
    route = solve_tsp(distance_matrix)

    if route:
        print("Optimal Route (indices):", route)
        ordered_locations = [locations[i] for i in route]
        route_geojson = get_osrm_route(ordered_locations)
        visualize_route_with_roads(ordered_locations, route_geojson)
    else:
        print("No solution found!")
else:
    print("Not enough locations for route optimization.")
