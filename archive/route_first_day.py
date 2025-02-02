import pandas as pd
import numpy as np
from math import radians, cos, sin, sqrt, atan2
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
import requests

# Load the data
data_file = '/Users/fabianveltkamp/Desktop/daily_hotspot_predictions.csv'
data = pd.read_csv(data_file, low_memory=False)

# Ensure 'datum' is in datetime format and filter for the first day
data['datum'] = pd.to_datetime(data['datum'], errors='coerce')
filtered_data = data[data['datum'] == '2017-05-18']

# Extract relevant columns
locations = filtered_data[['center_latitude',
                           'center_longitude']].dropna().values
hotspot_categories = filtered_data['hotspot_category']

# Haversine function to compute distances


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * \
        cos(radians(lat2)) * sin(dlon / 2)**2
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
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)

    # Create the routing model
    routing = pywrapcp.RoutingModel(manager)

    # Define the cost function
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Set the search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

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
    # Convert coordinates into OSRM-compatible format
    coordinates = ";".join([f"{lon},{lat}" for lat, lon in locations])
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{coordinates}?overview=full&geometries=geojson"

    # Make a request to OSRM
    response = requests.get(osrm_url)
    if response.status_code == 200:
        route_data = response.json()
        # Return GeoJSON for the route
        return route_data['routes'][0]['geometry']
    else:
        print("Failed to fetch the route from OSRM.")
        return None

# Visualize the road-based route with Folium


def visualize_route_with_roads(locations, route_geojson):
    # Center the map at the first location
    map_center = [locations[0][0], locations[0][1]]
    route_map = folium.Map(location=map_center, zoom_start=13)

    # Add markers for all locations
    for idx, loc in enumerate(locations):
        folium.Marker(
            location=[loc[0], loc[1]],
            popup=f"Location {idx}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(route_map)

    # Add the road-based route as a GeoJSON line
    if route_geojson:
        folium.GeoJson(
            route_geojson,
            name="Road-based Route",
            style_function=lambda x: {
                "color": "red", "weight": 4, "opacity": 0.8}
        ).add_to(route_map)

    # Save the map to an HTML file
    route_map.save("road_based_route_map.html")
    print("Map saved as 'road_based_route_map.html'. Open this file in your browser to view the map.")


# Main execution
distance_matrix = create_distance_matrix(locations)
route = solve_tsp(distance_matrix)

if route:
    print("Optimal Route (indices):", route)
    print("Optimal Route (locations):")
    ordered_locations = [locations[i] for i in route]
    route_geojson = get_osrm_route(ordered_locations)
    visualize_route_with_roads(ordered_locations, route_geojson)
else:
    print("No solution found!")
