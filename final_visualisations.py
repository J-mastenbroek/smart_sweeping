import pandas as pd
import geopandas as gpd
import folium
from shapely import wkt

# Load data files
data_file = "daily_hotspot_predictions.csv"
shapefile_path = "dataset/UTRGRID100/UTRGRID100WGS84.shp"

# Load predicted hotspots
print("Loading predicted hotspot data...")
predicted_data = pd.read_csv(data_file, low_memory=False)
predicted_data['datum'] = pd.to_datetime(predicted_data['datum'], errors='coerce')
predicted_data['geometry'] = predicted_data['geometry'].apply(wkt.loads)
predicted_data = gpd.GeoDataFrame(predicted_data, geometry='geometry')
predicted_data.set_crs(epsg=4326, inplace=True)

# Load actual hotspot data from shapefile
print("Loading actual hotspot data...")
grid_data = gpd.read_file(shapefile_path)
grid_data = grid_data.to_crs(epsg=4326)

# Merge actual and predicted hotspots
data = predicted_data.merge(grid_data, on="UTRGRID100", how="left")

# Define hotspot categories and **better contrast color mappings**
hotspot_categories = ["5-10 min", "10-15 min", "15+ min"]  # Ignore <5 min
actual_colors = {
    "5-10 min": "#00C0FF",  # Light Blue (Better)
    "10-15 min": "#0080FF",  # Medium Blue
    "15+ min": "#0040FF"  # Dark Blue (Worst)
}
predicted_colors = {
    "5-10 min": "#FFC000",  # Light Yellow (Better)
    "10-15 min": "#FF8000",  # Orange
    "15+ min": "#FF4000"  # Dark Red (Worst)
}

# Offset distance to bring predicted & actual closer
offset = 0.0002  # Small shift

# Filter for **one day**: May 18, 2017
selected_date = "2017-05-18"
filtered_data = data[data['datum'] == selected_date]

# Create the Folium map **with a WHITE background**
map_center = [52.0907, 5.1214]
hotspot_map = folium.Map(location=map_center, zoom_start=13, tiles="CartoDB Positron")  # White background

# Add grid layer (Shapefile overlay)
folium.GeoJson(
    grid_data,
    name="Grid Overlay",
    style_function=lambda x: {
        "color": "#000000", "weight": 0.3, "fillOpacity": 0.1
    }
).add_to(hotspot_map)

# Add actual hotspots (Squares 🔳)
for _, row in filtered_data.iterrows():
    if row['hotspot_category'] in actual_colors:
        folium.RegularPolygonMarker(
            location=[row['center_latitude'] + offset, row['center_longitude']],  # Shift up
            number_of_sides=4,  # Square
            radius=8,
            color=actual_colors[row['hotspot_category']],
            fill=True,
            fill_color=actual_colors[row['hotspot_category']],
            fill_opacity=0.9,
            popup=f"Actual Hotspot: {row['hotspot_category']}"
        ).add_to(hotspot_map)

# Add predicted hotspots (Triangles 🔺)
for _, row in filtered_data.iterrows():
    if row['hotspot_category'] in predicted_colors:
        folium.RegularPolygonMarker(
            location=[row['center_latitude'] - offset, row['center_longitude']],  # Shift down
            number_of_sides=3,  # Triangle
            radius=8,
            color=predicted_colors[row['hotspot_category']],
            fill=True,
            fill_color=predicted_colors[row['hotspot_category']],
            fill_opacity=0.9,
            popup=f"Predicted Hotspot: {row['hotspot_category']}"
        ).add_to(hotspot_map)

# Add a bigger & more centered legend
legend_html = """
<div style="position: fixed; bottom: 70px; left: 100px; width: 380px; height: 230px; 
            background-color: white; z-index:9999; font-size:14px;
            border:2px solid grey; padding: 15px;">
    <b>Hotspot Categories</b><br>
    <b>Actual (Squares 🔳)</b><br>
    <i style="background: #00C0FF; width: 16px; height: 16px; display: inline-block;"></i> 5-10 min<br>
    <i style="background: #0080FF; width: 16px; height: 16px; display: inline-block;"></i> 10-15 min<br>
    <i style="background: #0040FF; width: 16px; height: 16px; display: inline-block;"></i> 15+ min<br>
    <br>
    <b>Predicted (Triangles 🔺)</b><br>
    <i style="background: #FFC000; width: 16px; height: 16px; display: inline-block;"></i> 5-10 min<br>
    <i style="background: #FF8000; width: 16px; height: 16px; display: inline-block;"></i> 10-15 min<br>
    <i style="background: #FF4000; width: 16px; height: 16px; display: inline-block;"></i> 15+ min<br>
</div>
"""
hotspot_map.get_root().html.add_child(folium.Element(legend_html))

# Save the map
hotspot_map.save("hotspot_map_actual_vs_predicted.html")
print("✅ Map saved as 'hotspot_map_actual_vs_predicted.html'. Open this file in your browser to view the map.")
