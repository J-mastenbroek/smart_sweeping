import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import TimestampedGeoJson
import json

# import 
csv_path = "dataset/alles_20171819_3tracties.csv"
csv_data = pd.read_csv(csv_path, delimiter=';', usecols=['tijdstip', 'actie', 'UTRGRID100', 'seconde'])
arbeid_data = csv_data[csv_data['actie'] == 'Arbeid']

# aggregate
arbeid_data['tijdstip'] = pd.to_datetime(arbeid_data['tijdstip'])
arbeid_data['date'] = arbeid_data['tijdstip'].dt.date
arbeid_aggregated = arbeid_data.groupby(['date', 'UTRGRID100']).agg({
    'seconde': 'sum'
}).reset_index()

# tresholds at 5+, 10+ and 15+ minutes
arbeid_aggregated['hotspot_category'] = pd.cut(
    arbeid_aggregated['seconde'] / 60,  
    bins=[0, 5, 10, 15, float('inf')],
    labels=["<5 min", "5-10 min", "10-15 min", "15+ min"]
)
arbeid_aggregated = arbeid_aggregated[arbeid_aggregated['seconde'] >= 5 * 60]

# load and merge shapefiles
shapefile_path = "dataset/UTRGRID100/UTRGRID100WGS84.shp"
grid_data = gpd.read_file(shapefile_path)
grid_data = grid_data.to_crs(epsg=4326)
merged_data = grid_data.merge(arbeid_aggregated, left_on='UTRGRID100', right_on='UTRGRID100')

# hotspot color mapping
color_mapping = {
    "<5 min": "#f7f7f7",  
    "5-10 min": "yellow",
    "10-15 min": "orange",
    "15+ min": "red"
}

features = []
for date in sorted(arbeid_aggregated['date'].unique()):
    daily_data = merged_data[merged_data['date'] == date]
    
    for _, row in daily_data.iterrows():
        color = color_mapping[row['hotspot_category']]
        feature = {
            "type": "Feature",
            "geometry": row['geometry'].__geo_interface__,
            "properties": {
                "times": [str(date)],
                "seconde": row['seconde'],
                "hotspot_category": row['hotspot_category'],
                "style": {
                    "color": color,
                    "weight": 0.5,
                    "fillOpacity": 0.8,
                    "fillColor": color
                },
            },
        }
        features.append(feature)

m = folium.Map(location=[52.0907, 5.1214], zoom_start=14, tiles="CartoDB positron")
TimestampedGeoJson(
    {"type": "FeatureCollection", "features": features},
    period="P1D",
    auto_play=True,
    loop=True,
    max_speed=5,
    loop_button=True,
    date_options="YYYY-MM-DD",
    time_slider_drag_update=True,
    duration='P1D'
).add_to(m)

# custom legend
legend_html = """
<div style="position: fixed;
            bottom: 50px; left: 50px; width: 150px; height: 120px; 
            background-color: white; z-index:9999; font-size:14px;
            border:2px solid grey; padding: 10px;">
    <b>Hotspot Categories</b><br>
    <i style="background: yellow; width: 10px; height: 10px; display: inline-block;"></i> 5-10 min<br>
    <i style="background: orange; width: 10px; height: 10px; display: inline-block;"></i> 10-15 min<br>
    <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> 15+ min<br>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

m.save("hotspot_heatmap.html")
print("Hotspot heatmap saved as 'hotspot_heatmap.html'")
