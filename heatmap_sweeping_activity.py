import pandas as pd
import geopandas as gpd
import folium
from folium import Choropleth

# load the csv files and filter on sweeping
csv_path = "dataset/alles_20171819_3tracties.csv"
csv_data = pd.read_csv(csv_path, delimiter=';')
arbeid_data = csv_data[csv_data['actie'] == 'Arbeid']

# aggregate for sweeping time spent in the grids 
arbeid_aggregated = arbeid_data.groupby('UTRGRID100').agg({
    'seconde': 'sum' 
}).reset_index()

print("\nAggregated Data for Sweeping (Arbeid):")
print(arbeid_aggregated.head())

# load shapefiles
shapefile_path = "dataset/UTRGRID100/UTRGRID100WGS84.shp"
grid_data = gpd.read_file(shapefile_path)

# merge sweeping time with shape
merged_data = grid_data.merge(arbeid_aggregated, left_on='UTRGRID100', right_on='UTRGRID100')

# folium map creation
geojson_data = merged_data.to_crs(epsg=4326)  
geojson_path = "merged_data.geojson"
geojson_data.to_file(geojson_path, driver="GeoJSON")
m = folium.Map(location=[52.0907, 5.1214], zoom_start=12, tiles="CartoDB positron")

# map parameters
Choropleth(
    geo_data=geojson_path,
    name='Sweeping Activity',
    data=arbeid_aggregated,
    columns=['UTRGRID100', 'seconde'],
    key_on='feature.properties.UTRGRID100',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Sweeping Activity (seconds)'
).add_to(m)

folium.LayerControl().add_to(m)
m.save("sweeping_heatmap.html")
print("Heatmap saved as 'sweeping_heatmap.html'")
