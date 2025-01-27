import pandas as pd
import numpy as np
from math import radians, cos, sin, sqrt, atan2
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import folium
import requests
from folium.plugins import TimeSliderChoropleth
import time

# Load the data
data_file = '/Users/fabianveltkamp/Desktop/daily_hotspot_predictions.csv'
data = pd.read_csv(data_file, low_memory=False)

# Ensure 'datum' is in datetime format
data['datum'] = pd.to_datetime(data['datum'], errors='coerce')

# Haversine function to compute distances


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lat2 - lon1)
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
    manager = pywrapcp.RoutingIndexManager(len(distance_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    solution = routing.SolveWithParameters(search_parameters)

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

    time.sleep(0.5)  # Add a small delay between requests
    response = requests.get(osrm_url)
    if response.status_code == 200:
        route_data = response.json()
        # Return GeoJSON for the route
        return route_data['routes'][0]['geometry']
    else:
        print(
            f"Failed to fetch the route from OSRM. Status code: {response.status_code}")
        return None

# Create the map with a slider


def create_map_with_slider(data):
    unique_dates = sorted(data['datum'].dropna().unique())  # Get unique dates
    map_center = [data['center_latitude'].mean(
    ), data['center_longitude'].mean()]
    route_map = folium.Map(location=map_center, zoom_start=13)

    # Dictionary to store routes per day
    routes_per_day = {}

    for date in unique_dates:
        day_data = data[data['datum'] == date]
        locations = day_data[['center_latitude',
                              'center_longitude']].dropna().values
        if len(locations) > 1:  # Only process if there are multiple points
            print(f"Processing date: {date}")
            distance_matrix = create_distance_matrix(locations)
            route = solve_tsp(distance_matrix)

            if route:
                ordered_locations = [locations[i] for i in route]
                route_geojson = get_osrm_route(ordered_locations)
                if route_geojson:
                    routes_per_day[date] = route_geojson

    # Add routes to the map with a slider
    geojson_collection = {
        "type": "FeatureCollection",
        "features": []
    }

    for date, geojson in routes_per_day.items():
        feature = {
            "type": "Feature",
            "geometry": geojson,
            "properties": {"time": date.strftime('%Y-%m-%d')}
        }
        geojson_collection["features"].append(feature)

    TimeSliderChoropleth(
        data=geojson_collection,
        styledict={
            str(idx): {
                "color": "red",
                "opacity": 0.8
            }
            for idx, feature in enumerate(geojson_collection["features"])
        }
    ).add_to(route_map)

    route_map.save("daily_routes_with_slider.html")
    print("Map saved as 'daily_routes_with_slider.html'. Open this file in your browser to view the map.")


# Main execution
create_map_with_slider(data)
