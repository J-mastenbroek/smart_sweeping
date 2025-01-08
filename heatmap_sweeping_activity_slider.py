import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import TimestampedGeoJson
import branca.colormap as cm
import json

# Load and process CSV data
csv_path = "dataset/alles_20171819_3tracties.csv"
csv_data = pd.read_csv(csv_path, delimiter=';', usecols=['tijdstip', 'actie', 'UTRGRID100', 'seconde'])
arbeid_data = csv_data[csv_data['actie'] == 'Arbeid']

# Convert 'tijdstip' to datetime and extract the date
arbeid_data['tijdstip'] = pd.to_datetime(arbeid_data['tijdstip'])
arbeid_data['date'] = arbeid_data['tijdstip'].dt.date

# Aggregate sweeping activity per day and grid, applying the 300-second threshold
arbeid_aggregated = arbeid_data.groupby(['date', 'UTRGRID100']).agg({
    'seconde': 'sum'
}).reset_index()

arbeid_aggregated = arbeid_aggregated[arbeid_aggregated['seconde'] >= 300]  # Exclude grids with less than 300 seconds

# Load shapefile data
shapefile_path = "dataset/UTRGRID100/UTRGRID100WGS84.shp"
grid_data = gpd.read_file(shapefile_path)
grid_data = grid_data.to_crs(epsg=4326)  # Convert to WGS 84 CRS

# Merge shapefile data with aggregated data
merged_data = grid_data.merge(arbeid_aggregated, left_on='UTRGRID100', right_on='UTRGRID100')

# Precompute GeoJSON features and colormap with YlOrRd
colormap = cm.linear.YlOrRd_03.scale(0, merged_data['seconde'].max()).to_step(n=10)  
colormap.caption = "Daily Sweeping Activity (seconds)"

features = []
for date in sorted(arbeid_aggregated['date'].unique()):  # Iterate by day
    daily_data = merged_data[merged_data['date'] == date]

    for _, row in daily_data.iterrows():
        # Apply color scale based on seconds value
        color = colormap(row['seconde'])

        feature = {
            "type": "Feature",
            "geometry": row['geometry'].__geo_interface__,
            "properties": {
                "time": str(date),
                "seconde": row['seconde'],
                "style": {
                    "color": color,
                    "weight": 0.5,  # Grid borders are now more visible
                    "fillOpacity": 0.8,  # Higher opacity for better grid visibility
                    "fillColor": color,  # Use the color from the colormap
                },
            },
        }
        features.append(feature)

# Save features to GeoJSON file
with open("daily_features_colored_YlOrRd.geojson", "w") as f:
    json.dump({"type": "FeatureCollection", "features": features}, f)

# Create the map with adjusted zoom level
m = folium.Map(location=[52.0907, 5.1214], zoom_start=14, tiles="CartoDB positron")

# Add TimestampedGeoJson with the new features
TimestampedGeoJson(
    {"type": "FeatureCollection", "features": features},
    period="P1D",  # Daily intervals
    auto_play=True,
    loop=True,
    max_speed=5,
    loop_button=True,
    date_options="YYYY-MM-DD",
    time_slider_drag_update=True,
).add_to(m)

# Add colormap to the map for reference
colormap.add_to(m)

# Save the map
m.save("daily_sweeping_heatmap.html")
print("Daily heatmap saved as 'daily_sweeping_heatmap.html'")
