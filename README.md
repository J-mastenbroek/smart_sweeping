# Smart Sweeping

## The Challenge

The objective is to enhance the efficiency of street sweeping in the Gemeente Utrecht by addressing the following:

1. **Hotspot Map**: Create a map highlighting areas with concentrated sweeping activity.
2. **Predictive Model**: Forecast hotspots for the next day or period.
3. **Optimal Pathing**: Identify the shortest driving routes to predicted hotspots.
4. **Efficiency Evaluation**: Calculate and assess the efficiency improvements achieved.

---

## Dataset Overview

The dataset contains operational data for **three street sweepers** in Utrecht with IDs:

- **4271**
- **8284**
- **8288**

### Data Characteristics:

- **Daily Records**: 86,400 rows per day per street sweeper (one row per second).
- **Filtered Dataset**: A cleaned and combined dataset with **23,473,356 rows** and **10 columns** is available for analysis.

### Key Columns:

1. **Tractie**: Street sweeper ID.
2. **Actie**: Activity type (either _arbeid_ – working or _rijden_ – driving).
3. **Lat/Long**: Geographical coordinates of the sweeper's location.
4. **Sec**: Time spent (in seconds) for specific actions.
5. **Grid100**: A 100m x 100m grid reference. Shapefiles for this grid are provided in the `UTRGRID100` folder for spatial analysis.

---
