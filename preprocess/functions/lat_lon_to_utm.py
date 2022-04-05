import pandas as pd
import geopandas as gpd
import requests

def convertLatLon(df, out_crs='EPSG:32632'):
    """
    - performs final post-processing on the entire grid to extract the lat and
    lon values for each grid point and convert to x,y in meters as given by the
    given output CRS
    :param df: dataframe of the entire world grid data
    :param out_crs: desired output CRS,
                    e.g. EPSG:3035 (optimized for Europe) or EPSG:32632 (optimized for Switzerland)
    :return df: dataframe of the entire world grid data with lat,lon coordinates converted to x,y in meters
    """
    gdf = gpd.GeoDataFrame(df, crs='EPSG:4326', geometry=gpd.points_from_xy(df.lon, df.lat))
    gdf = gdf.to_crs(crs=out_crs)
    gdf['x[m]'] = gdf['geometry'].x
    gdf['y[m]'] = gdf['geometry'].y
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    return df

def convertLV95(x, y):
    """
    converts the Swiss coordinate systm LV95 to WGS84 coordinates with the REST API from swisstopo
    """

    base_url = 'http://geodesy.geo.admin.ch/reframe/lv95towgs84'
    url = f'{base_url}?easting={x}&northing={y}&format=json'
    r = requests.get(url=url)
    x_wgs84 = r.json()['easting']
    y_wgs84 = r.json()['northing']
    return x_wgs84, y_wgs84

