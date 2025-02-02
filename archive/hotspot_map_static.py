import pandas as pd
import geopandas as gpd
import folium

# Import data
csv_path = "dataset/alles_20171819_3tracties.csv"
csv_data = pd.read_csv(csv_path, delimiter=';', usecols=['tijdstip', 'actie', 'UTRGRID100', 'seconde'])
arbeid_data = csv_data[csv_data['actie'] == 'Arbeid']

# Aggregate data
arbeid_data['tijdstip'] = pd.to_datetime(arbeid_data['tijdstip'], errors='coerce')
arbeid_aggregated = arbeid_data.groupby(['UTRGRID100']).agg({'seconde': 'sum'}).reset_index()

# Normalize sweeping time (0 to 1 scale)
arbeid_aggregated['normalized_seconde'] = arbeid_aggregated['seconde'] / arbeid_aggregated['seconde'].max()

# Hotspot categories based on normalized values
arbeid_aggregated['hotspot_category'] = pd.cut(
    arbeid_aggregated['normalized_seconde'],
    bins=[0, 0.2, 0.5, 0.8, 1.0],
    labels=["Low (<20%)", "Moderate (20-50%)", "High (50-80%)", "Very High (>80%)"]
)

# Load and merge shapefiles
shapefile_path = "dataset/UTRGRID100/UTRGRID100WGS84.shp"
grid_data = gpd.read_file(shapefile_path)
grid_data = grid_data.to_crs(epsg=4326)
merged_data = grid_data.merge(arbeid_aggregated, left_on='UTRGRID100', right_on='UTRGRID100')

# Hotspot color mapping
color_mapping = {
    "Low (<20%)": "#f7f7f7",
    "Moderate (20-50%)": "yellow",
    "High (50-80%)": "orange",
    "Very High (>80%)": "red"
}

# Create map
m = folium.Map(location=[52.0907, 5.1214], zoom_start=14, tiles="CartoDB positron")

# Add normalized hotspots to map
for _, row in merged_data.iterrows():
    color = color_mapping[row['hotspot_category']]
    folium.GeoJson(
        data=row['geometry'].__geo_interface__,
        style_function=lambda _: {
            "color": color,
            "weight": 0.5,
            "fillOpacity": 0.8,
            "fillColor": color
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["UTRGRID100", "hotspot_category"],  # Match fields with aliases
            aliases=["Grid ID", "Hotspot Category"]   # Ensure same length
        )
    ).add_to(m)

# Custom legend
legend_html = """
<div style="position: fixed;
            bottom: 50px; left: 50px; width: 200px; height: 150px; 
            background-color: white; z-index:9999; font-size:14px;
            border:2px solid grey; padding: 10px;">
    <b>Hotspot Categories</b><br>
    <i style="background: #f7f7f7; width: 10px; height: 10px; display: inline-block;"></i> Low (<20%)<br>
    <i style="background: yellow; width: 10px; height: 10px; display: inline-block;"></i> Moderate (20-50%)<br>
    <i style="background: orange; width: 10px; height: 10px; display: inline-block;"></i> High (50-80%)<br>
    <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> Very High (>80%)<br>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Save the map
m.save("normalized_hotspot_map.html")
print("Static normalized hotspot map saved as 'normalized_hotspot_map.html'")
