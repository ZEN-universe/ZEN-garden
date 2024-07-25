import geopandas as gpd
from shapely.geometry import Polygon
import os
DIRECTORY = '../zen_garden/postprocess/results/'

def make_bbox(long0, lat0, long1, lat1):
    """
    Function to create a bounding box polygon.

    Args:
        long0 (float): Starting longitude.
        lat0 (float): Starting latitude.
        long1 (float): Ending longitude.
        lat1 (float): Ending latitude.

    Returns:
        Polygon: Bounding box polygon object.
    """
    return Polygon([[long0, lat0],
                    [long1, lat0],
                    [long1, lat1],
                    [long0, lat1]])

def create_US():
    """
    Function to create a geospatial dataset representing countries in Europe (Faraway islands of France).

    Returns:
        GeoDataFrame: Geospatial dataset representing European countries.
    """
    # Define the bounding box to exclude far away Islands of France
    bbox = make_bbox(-125, 0, -50, 90)
    bbox_gdf = gpd.GeoDataFrame(index=[0], geometry=[bbox])

    # Read the geospatial dataset of European countries
    us_gdf = gpd.read_file(os.path.join(DIRECTORY, 'States_shapefile.geojson'))

    return us_gdf


def create_county_US():
    """
    Function to create a geospatial dataset representing countries in Europe (Faraway islands of France).

    Returns:
        GeoDataFrame: Geospatial dataset representing European countries.
    """

    # Read the geospatial dataset of European countries
    #show current directory
    # Construct the path to the state shapefile
    #state_shapefile_path = '../zen_garden/postprocess/results/cb_2023_us_county_20m/cb_2023_us_county_20m.shp'
    state_shapefile_path = 'cb_2023_us_county_20m/cb_2023_us_county_20m.shp'

    # Read the shapefile
    us_counties = gpd.read_file(state_shapefile_path)
    us_counties['county_code'] = us_counties['STUSPS'] + '_' + us_counties['NAME'].str[:2].str.upper() + us_counties['COUNTYFP']
    us_counties.rename(columns={'STUSPS':'state'}, inplace=True)

    return us_counties