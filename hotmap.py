import pandas as pd
import folium
from folium.plugins import HeatMap

# Load the CSV file with proper delimiter
data_file = '/Users/fabianveltkamp/Desktop/alles_20171819_3tracties.csv'
data = pd.read_csv(data_file, delimiter=';', low_memory=False)

# Replace commas with dots in lat and long columns, then convert them to float
data['lat'] = data['lat'].str.replace(',', '.').astype(float)
data['long'] = data['long'].str.replace(',', '.').astype(float)

# Filter data for 'Arbeid'
arbeid_data = data[data['actie'] == 'Arbeid']

# Group by lat/long and sum up 'seconde'
hotspot_data = arbeid_data.groupby(['lat', 'long']).agg(
    {'seconde': 'sum'}).reset_index()

# Normalize the 'seconde' column for heatmap intensity
hotspot_data['seconde_normalized'] = hotspot_data['seconde'] / \
    hotspot_data['seconde'].max()

# Create a heatmap
m = folium.Map(location=[52.0907, 5.1214], zoom_start=12)
heat_data = hotspot_data[['lat', 'long', 'seconde_normalized']].values.tolist()
HeatMap(heat_data, radius=10).add_to(m)

# Save the map
m.save('hotspot_map.html')
print("Hotspot map created: 'hotspot_map.html'")
