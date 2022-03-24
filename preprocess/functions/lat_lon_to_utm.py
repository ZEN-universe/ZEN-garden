import pandas as pd
import geopandas as gpd

def convertLatLon(df, out_crs):
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
