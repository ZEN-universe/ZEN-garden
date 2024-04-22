import os
import pandas as pd
import chardet
import geopandas as gpd
from shapely.geometry import Point
import numpy as np

def generate_existing_capacities(file_demand, lifetime, technology):
    capacity_existing = pd.read_csv(file_demand)

    capacity_existing['year_construction'] = np.nan
    capacity_existing['capacity_existing'] = capacity_existing['demand'] / lifetime

    new_rows = []

    for index, row in capacity_existing.iterrows():
        for year in range(lifetime):
            new_row = row.copy()
            new_row['year_construction'] = 2023 - year
            new_rows.append(new_row)

    expanded_df = pd.DataFrame(new_rows)

    print(expanded_df.head())
    expanded_df = expanded_df.drop(['demand'], axis=1)
    expanded_df.to_csv(f'../data/hard_to_abate/set_technologies/set_conversion_technologies/{technology}/capacity_existing.csv', index=False)

    return expanded_df

def generate_existing_capacity_ASU(file_demand, lifetime, technology):
    capacity_existing = pd.read_csv(file_demand)

    capacity_existing['year_construction'] = np.nan
    capacity_existing['capacity_existing'] = capacity_existing['demand'] * 0.823 / lifetime

    new_rows = []

    for index, row in capacity_existing.iterrows():
        for year in range(lifetime):
            new_row = row.copy()
            new_row['year_construction'] = 2023 - year
            new_rows.append(new_row)

    expanded_df = pd.DataFrame(new_rows)

    print(expanded_df.head())
    expanded_df = expanded_df.drop(['demand'], axis=1)
    expanded_df.to_csv(f'../data/hard_to_abate/set_technologies/set_conversion_technologies/{technology}/capacity_existing.csv', index=False)

    return expanded_df

def generate_existing_capacities_steel(file_demand, lifetime_BF_BOF, lifetime_EAF, lifetime_DRI):
    capacity_existing = pd.read_csv(file_demand)

    capacity_existing['year_construction'] = np.nan
    capacity_existing_BF_BOF = capacity_existing.copy()
    capacity_existing_BF_BOF['capacity_existing'] = capacity_existing_BF_BOF['demand'] * 0.7 / lifetime_BF_BOF

    new_rows = []

    for index, row in capacity_existing_BF_BOF.iterrows():
        for year in range(lifetime_BF_BOF):
            new_row = row.copy()
            new_row['year_construction'] = 2023 - year
            new_rows.append(new_row)

    expanded_df_BF_BOF = pd.DataFrame(new_rows)
    expanded_df_BF_BOF = expanded_df_BF_BOF.drop(['demand'], axis=1)
    print(expanded_df_BF_BOF.head())
    expanded_df_BF_BOF.to_csv(
        f'../data/hard_to_abate/set_technologies/set_conversion_technologies/BF_BOF/capacity_existing.csv',
        index=False)

    capacity_existing_EAF = capacity_existing.copy()
    capacity_existing_EAF['capacity_existing'] = capacity_existing_EAF['demand'] * 0.3 / lifetime_EAF

    new_rows = []

    for index, row in capacity_existing_EAF.iterrows():
        for year in range(lifetime_EAF):
            new_row = row.copy()
            new_row['year_construction'] = 2023 - year
            new_rows.append(new_row)

    expanded_df_EAF = pd.DataFrame(new_rows)
    expanded_df_EAF = expanded_df_EAF.drop(['demand'], axis=1)
    print(expanded_df_EAF.head())
    expanded_df_EAF.to_csv(
        f'../data/hard_to_abate/set_technologies/set_conversion_technologies/EAF/capacity_existing.csv',
        index=False)

    capacity_existing_DRI = capacity_existing.copy()
    capacity_existing_DRI['capacity_existing'] = capacity_existing_EAF['demand'] * 0.22 / lifetime_EAF

    new_rows = []

    for index, row in capacity_existing_DRI.iterrows():
        for year in range(lifetime_DRI):
            new_row = row.copy()
            new_row['year_construction'] = 2023 - year
            new_rows.append(new_row)

    expanded_df_DRI = pd.DataFrame(new_rows)
    expanded_df_DRI = expanded_df_DRI.drop(['demand'], axis=1)
    print(expanded_df_DRI.head())
    expanded_df_DRI.to_csv(
        f'../data/hard_to_abate/set_technologies/set_conversion_technologies/DRI/capacity_existing.csv',
        index=False)



def adjust_year(year, lifetime):
    if year == 0:
        return year
    while year + lifetime <= 2024:
        year += lifetime
    return year

def existing_capacities_hydrogen(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(100000))
        encoding = result['encoding']

    existing_capa = pd.read_csv(file_path, delimiter=';', encoding=encoding)
    existing_capa_filtered = existing_capa[['eigl_text_lon', 'eigl_text_lat', 'start_year', 'eigl_process', 'capacity_mwe']].copy()

    existing_capa_filtered['geometry'] = [Point(xy) for xy in zip(existing_capa_filtered.eigl_text_lon, existing_capa_filtered.eigl_text_lat)]
    geo_df = gpd.GeoDataFrame(existing_capa_filtered, geometry='geometry')
    geo_df.set_crs(epsg=4326, inplace=True)

    nuts2_regions = gpd.read_file('nuts_data/NUTS_RG_20M_2021_4326.shp')
    nuts2_regions = nuts2_regions.to_crs(epsg=4326)

    joined = gpd.sjoin(geo_df, nuts2_regions, how="inner", predicate='intersects')

    filtered_df = joined[joined['LEVL_CODE'] == 2]
    filtered_df = filtered_df[['NUTS_ID', 'start_year', 'eigl_process', 'capacity_mwe']]

    filtered_df['start_year'] = filtered_df['start_year'].fillna('0')
    filtered_df['start_year'] = pd.to_numeric(filtered_df['start_year'], errors='coerce')
    filtered_df = filtered_df[filtered_df['start_year'] <= 2024]
    filtered_df['start_year'] = filtered_df['start_year'].astype(int)

    filtered_df['capacity_mwe'] = filtered_df['capacity_mwe'] / 1000 # in GW

    filtered_df = filtered_df.rename(columns={'NUTS_ID': 'node', 'start_year': 'year_construction', 'capacity_mwe': 'capacity_existing'})

    electrolysis_capacity = filtered_df[filtered_df['eigl_process'].str.contains('electrolysis', case=False, na=False)]
    electrolysis_capacity = electrolysis_capacity.drop('eigl_process', axis=1)
    electrolysis_capacity['year_construction'] = electrolysis_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=10))
    electrolysis_capacity = electrolysis_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})
    electrolysis_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/electrolysis/capacity_existing.csv', index=False)

    SMR_capacity = filtered_df[filtered_df['eigl_process'].str.contains('other or unknown', case=False, na=False)]
    SMR_capacity = SMR_capacity.drop('eigl_process', axis=1)
    #SMR_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/SMR/capacity_existing.csv', index=False)

    SMR_CCS_capacity = filtered_df[filtered_df['eigl_process'].str.contains('CCS', case=False, na=False)]
    SMR_CCS_capacity = SMR_CCS_capacity.drop('eigl_process', axis=1)
    SMR_CCS_capacity['year_construction'] = SMR_CCS_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=25))
    SMR_CCS_capacity = SMR_CCS_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})
    SMR_CCS_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/SMR_CCS/capacity_existing.csv', index=False)

def existing_capacities_carbon_capture(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(100000))
        encoding = result['encoding']

    existing_capa = pd.read_csv(file_path, delimiter=';', encoding=encoding)
    existing_capa_filtered = existing_capa[['eigl_text_lon', 'eigl_text_lat', 'start_date', 'capture_type', 'planned_capture_rate_kt', 'emission_type']].copy()

    existing_capa_filtered['geometry'] = [Point(xy) for xy in zip(existing_capa_filtered.eigl_text_lon, existing_capa_filtered.eigl_text_lat)]
    geo_df = gpd.GeoDataFrame(existing_capa_filtered, geometry='geometry')
    geo_df.set_crs(epsg=4326, inplace=True)

    nuts2_regions = gpd.read_file('nuts_data/NUTS_RG_20M_2021_4326.shp')
    nuts2_regions = nuts2_regions.to_crs(epsg=4326)

    joined = gpd.sjoin(geo_df, nuts2_regions, how="inner", predicate='intersects')

    filtered_df = joined[joined['LEVL_CODE'] == 2]
    filtered_df = filtered_df[['NUTS_ID', 'start_date', 'capture_type', 'planned_capture_rate_kt', 'emission_type']].copy()

    filtered_df['start_date'] = filtered_df['start_date'].fillna('0')
    filtered_df['start_date'] = pd.to_numeric(filtered_df['start_date'], errors='coerce')
    filtered_df = filtered_df[filtered_df['start_date'] <= 2024]
    filtered_df['start_date'] = filtered_df['start_date'].astype(int)

    filtered_df['planned_capture_rate_kt'] = filtered_df['planned_capture_rate_kt'] / 8760 # in kt/h

    filtered_df = filtered_df.rename(columns={'NUTS_ID': 'node', 'start_date': 'year_construction', 'planned_capture_rate_kt': 'capacity_existing'})

    DAC_capacity = filtered_df[filtered_df['capture_type'].str.contains('DAC', case=False, na=False)]
    DAC_capacity = DAC_capacity.drop(['capture_type', 'emission_type'], axis=1)
    DAC_capacity['year_construction'] = DAC_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=25))
    DAC_capacity = DAC_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})
    DAC_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/DAC/capacity_existing.csv', index=False)

    BF_BOF_CCS_capacity = filtered_df[filtered_df['emission_type'].str.contains('steel', case=False, na=False)]
    BF_BOF_CCS_capacity = BF_BOF_CCS_capacity.drop(['emission_type', 'capture_type'], axis=1)
    BF_BOF_CCS_capacity['year_construction'] = BF_BOF_CCS_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=25))
    BF_BOF_CCS_capacity = BF_BOF_CCS_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})
    BF_BOF_CCS_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/set_retrofitting_technologies/BF_BOF_CCS/capacity_existing.csv', index=False)

def existing_capacities_steel(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(100000))
        encoding = result['encoding']

    existing_capa = pd.read_csv(file_path, delimiter=';', encoding=encoding)
    existing_capa_filtered = existing_capa[['eigl_text_lon', 'eigl_text_lat', 'year', 'primary_production_type', 'capacity']].copy()

    existing_capa_filtered['geometry'] = [Point(xy) for xy in zip(existing_capa_filtered.eigl_text_lon, existing_capa_filtered.eigl_text_lat)]
    geo_df = gpd.GeoDataFrame(existing_capa_filtered, geometry='geometry')
    geo_df.set_crs(epsg=4326, inplace=True)

    nuts2_regions = gpd.read_file('nuts_data/NUTS_RG_20M_2021_4326.shp')
    nuts2_regions = nuts2_regions.to_crs(epsg=4326)

    joined = gpd.sjoin(geo_df, nuts2_regions, how="inner", predicate='intersects')

    filtered_df = joined[joined['LEVL_CODE'] == 2]
    filtered_df = filtered_df[['NUTS_ID', 'year', 'primary_production_type', 'capacity']]

    filtered_df['year'] = filtered_df['year'].fillna('0')
    filtered_df['year'] = pd.to_numeric(filtered_df['year'], errors='coerce')
    filtered_df = filtered_df[filtered_df['year'] <= 2024]
    filtered_df['year'] = filtered_df['year'].astype(int)
    print(filtered_df)

    filtered_df['capacity'] = filtered_df['capacity'] * (1000/8760) # in kt/h

    filtered_df = filtered_df.rename(columns={'NUTS_ID': 'node', 'year': 'year_construction', 'capacity': 'capacity_existing'})

    BF_BOF_capacity = filtered_df[filtered_df['primary_production_type'].str.contains('BF', case=False, na=False)]
    BF_BOF_capacity = BF_BOF_capacity.drop('primary_production_type', axis=1)
    BF_BOF_capacity['year_construction'] = BF_BOF_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=40))
    BF_BOF_capacity = BF_BOF_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})
    BF_BOF_capacity = BF_BOF_capacity[BF_BOF_capacity['year_construction'] != 0.0]
    BF_BOF_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/BF_BOF/capacity_existing.csv', index=False)

    demand_steel = pd.read_csv("../data/hard_to_abate/set_carriers/steel/demand.csv")
    steel_merged = pd.merge(demand_steel, BF_BOF_capacity, how='left', on='node')
    print('capacity')
    print(steel_merged['capacity_existing'].sum())
    print('demand')
    print(steel_merged['demand'].sum())
    print('percentage')
    print(steel_merged['capacity_existing'].sum() / steel_merged['demand'].sum())
    print(steel_merged)

    DRI_capacity = filtered_df[filtered_df['primary_production_type'].str.contains('DRI', case=False, na=False)]
    DRI_capacity = DRI_capacity.drop('primary_production_type', axis=1)
    DRI_capacity['year_construction'] = DRI_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=25))
    DRI_capacity = DRI_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})
    DRI_capacity = DRI_capacity[DRI_capacity['year_construction'] != 0.0]
    DRI_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/DRI/capacity_existing.csv', index=False)

    EAF_capacity = filtered_df[filtered_df['primary_production_type'].str.contains('EAF', case=False, na=False)]
    EAF_capacity = EAF_capacity.drop('primary_production_type', axis=1)
    EAF_capacity['year_construction'] = EAF_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=25))
    EAF_capacity = EAF_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})
    EAF_capacity = EAF_capacity[EAF_capacity['year_construction'] != 0.0]

    demand_steel = pd.read_csv("../data/hard_to_abate/set_carriers/steel/demand.csv")
    steel_merged = pd.merge(demand_steel, EAF_capacity, how='left', on='node')
    print('capacity')
    print(steel_merged['capacity_existing'].sum())
    print('demand')
    print(steel_merged['demand'].sum())
    print('percentage')
    print(steel_merged['capacity_existing'].sum()/steel_merged['demand'].sum())

    print(steel_merged)
    EAF_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/EAF/capacity_existing.csv', index=False)

def existing_capacities_cement(file_path):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(100000))
        encoding = result['encoding']

    existing_capa = pd.read_csv(file_path, delimiter=';', encoding=encoding)
    existing_capa_filtered = existing_capa[['eigl_text_lon', 'eigl_text_lat', 'year', 'capacity']].copy()

    existing_capa_filtered['geometry'] = [Point(xy) for xy in zip(existing_capa_filtered.eigl_text_lon, existing_capa_filtered.eigl_text_lat)]
    geo_df = gpd.GeoDataFrame(existing_capa_filtered, geometry='geometry')
    geo_df.set_crs(epsg=4326, inplace=True)

    nuts2_regions = gpd.read_file('nuts_data/NUTS_RG_20M_2021_4326.shp')
    nuts2_regions = nuts2_regions.to_crs(epsg=4326)

    joined = gpd.sjoin(geo_df, nuts2_regions, how="inner", predicate='intersects')

    filtered_df = joined[joined['LEVL_CODE'] == 2]
    filtered_df = filtered_df[['NUTS_ID', 'year', 'capacity']]

    filtered_df['year'] = filtered_df['year'].fillna('0')
    filtered_df['year'] = pd.to_numeric(filtered_df['year'], errors='coerce')
    filtered_df = filtered_df[filtered_df['year'] <= 2024]
    filtered_df['year'] = filtered_df['year'].astype(int)

    filtered_df['capacity'] = filtered_df['capacity'] * (1000/8760) # in kt/h

    filtered_df = filtered_df.rename(columns={'NUTS_ID': 'node', 'year': 'year_construction', 'capacity': 'capacity_existing'})

    cement_plant_capacity = filtered_df.copy()
    cement_plant_capacity['year_construction'] = cement_plant_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=25))
    cement_plant_capacity = cement_plant_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})

    demand_cement = pd.read_csv("../data/hard_to_abate/set_carriers/cement/demand.csv")
    merged_df = pd.merge(cement_plant_capacity, demand_cement, on='node', how='left')
    merged_df.loc[(merged_df['year_construction'] > 0.0) & (merged_df['capacity_existing'] == 0.0) & (~merged_df['demand'].isna()), 'capacity_existing'] = merged_df['demand']
    merged_df = merged_df[['node', 'year_construction', 'capacity_existing']]
    print(merged_df)
    cement_plant_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/cement_plant/capacity_existing.csv', index=False)

def existing_capacities_methanol(file_path, sheet_name):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(100000))
        encoding = result['encoding']

    existing_capa = pd.read_excel(file_path, sheet_name)
    existing_capa_filtered = existing_capa[['longitude', 'latitude', 'petrochemical', 'year', 'capacity']].copy()
    existing_capa_filtered = existing_capa_filtered[existing_capa_filtered['petrochemical'] == 'Methanol']

    existing_capa_filtered['geometry'] = [Point(xy) for xy in zip(existing_capa_filtered.longitude, existing_capa_filtered.latitude)]
    geo_df = gpd.GeoDataFrame(existing_capa_filtered, geometry='geometry')
    geo_df.set_crs(epsg=4326, inplace=True)

    nuts2_regions = gpd.read_file('nuts_data/NUTS_RG_20M_2021_4326.shp')
    nuts2_regions = nuts2_regions.to_crs(epsg=4326)

    joined = gpd.sjoin(geo_df, nuts2_regions, how="inner", predicate='intersects')

    filtered_df = joined[joined['LEVL_CODE'] == 2]
    filtered_df = filtered_df[['NUTS_ID', 'year', 'capacity']]

    filtered_df['year'] = filtered_df['year'].fillna('0')
    filtered_df['year'] = pd.to_numeric(filtered_df['year'], errors='coerce')
    filtered_df = filtered_df[filtered_df['year'] <= 2024]
    filtered_df['year'] = filtered_df['year'].astype(int)

    filtered_df['capacity'] = filtered_df['capacity'] /8760 # in kt/h

    filtered_df = filtered_df.rename(columns={'NUTS_ID': 'node', 'year': 'year_construction', 'capacity': 'capacity_existing'})

    methanol_synthesis_capacity = filtered_df.copy()
    methanol_synthesis_capacity['year_construction'] = methanol_synthesis_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=25))
    methanol_synthesis_capacity = methanol_synthesis_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})

    demand_methanol = pd.read_csv("../data/hard_to_abate/set_carriers/methanol/demand.csv")
    merged_df = pd.merge(methanol_synthesis_capacity, demand_methanol, on='node', how='left')
    merged_df = merged_df[['node', 'year_construction', 'capacity_existing']]
    print(merged_df)
    methanol_synthesis_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/methanol_synthesis/capacity_existing.csv', index=False)

def existing_capacities_ammonia(file_path, sheet_name):
    with open(file_path, 'rb') as file:
        result = chardet.detect(file.read(100000))
        encoding = result['encoding']

    existing_capa = pd.read_excel(file_path, sheet_name)
    existing_capa_filtered = existing_capa[['longitude', 'latitude', 'petrochemical', 'year', 'capacity']].copy()
    existing_capa_filtered = existing_capa_filtered[existing_capa_filtered['petrochemical'] == 'Ammonia']

    existing_capa_filtered['geometry'] = [Point(xy) for xy in zip(existing_capa_filtered.longitude, existing_capa_filtered.latitude)]
    geo_df = gpd.GeoDataFrame(existing_capa_filtered, geometry='geometry')
    geo_df.set_crs(epsg=4326, inplace=True)

    nuts2_regions = gpd.read_file('nuts_data/NUTS_RG_20M_2021_4326.shp')
    nuts2_regions = nuts2_regions.to_crs(epsg=4326)

    joined = gpd.sjoin(geo_df, nuts2_regions, how="inner", predicate='intersects')

    filtered_df = joined[joined['LEVL_CODE'] == 2]
    filtered_df = filtered_df[['NUTS_ID', 'year', 'capacity']]

    filtered_df['year'] = filtered_df['year'].fillna('0')
    filtered_df['year'] = pd.to_numeric(filtered_df['year'], errors='coerce')
    filtered_df = filtered_df[filtered_df['year'] <= 2024]
    filtered_df['year'] = filtered_df['year'].astype(int)

    filtered_df['capacity'] = filtered_df['capacity'] /8760 # in kt/h

    filtered_df = filtered_df.rename(columns={'NUTS_ID': 'node', 'year': 'year_construction', 'capacity': 'capacity_existing'})

    haber_bosch_capacity = filtered_df.copy()
    haber_bosch_capacity['year_construction'] = haber_bosch_capacity['year_construction'].apply(lambda year: adjust_year(year, lifetime=30))
    haber_bosch_capacity = haber_bosch_capacity.groupby(['node', 'year_construction'], as_index=False).agg({'capacity_existing': 'sum'})

    demand_ammonia = pd.read_csv("../data/hard_to_abate/set_carriers/cement/demand.csv")
    merged_df = pd.merge(haber_bosch_capacity, demand_ammonia, on='node', how='left')
    merged_df = merged_df[['node', 'year_construction', 'capacity_existing']]
    print(merged_df)
    haber_bosch_capacity.to_csv('../data/hard_to_abate/set_technologies/set_conversion_technologies/haber_bosch/capacity_existing.csv', index=False)
    return haber_bosch_capacity

def compare_capa_to_demand(demand_file, capacity_file, capacity_file_steel):
    demand = pd.read_csv(demand_file)
    capacity = pd.read_csv(capacity_file)
    #capacity_steel = pd.read_csv(capacity_file_steel)
    #merged_steel_df = pd.merge(capacity_steel, capacity, on='node', how='outer', suffixes=('_BF', '_EAF'))
    #merged_steel_df['capacity_existing_EAF'] = merged_steel_df['capacity_existing_EAF'].fillna(0)
    #merged_steel_df['capacity_existing_BF'] = merged_steel_df['capacity_existing_BF'].fillna(0)
    #merged_steel_df['capacity_existing'] = merged_steel_df['capacity_existing_BF'] + merged_steel_df['capacity_existing_EAF']
    #merged_steel_df = merged_steel_df[['node', 'capacity_existing']]
    #merged_df = pd.merge(demand, merged_steel_df, on='node', how='left', suffixes=('_demand', '_capacity'))
    merged_df = pd.merge(demand, capacity, on='node', how='left', suffixes=('_demand', '_capacity'))
    merged_df = merged_df[['node', 'demand', 'capacity_existing']]
    merged_df['difference'] = merged_df['demand'] - merged_df['capacity_existing']
    merged_df.to_csv("existing_capacities_input/diff_cement.csv", index=False)
    print(merged_df)

if __name__ == "__main__":

    industries = ['ammonia', 'methanol', 'oil_products', 'cement', #'steel'
                   ]

    industry_data = {
        'ammonia': {
            'lifetime': 30,
            'technology': 'haber_bosch'
        },
        'steel': {
            'lifetime': 40,
            'technology': 'BF_BOF'
        },
        'methanol': {
            'lifetime': 25,
            'technology': 'methanol_synthesis'
        },
        'oil_products': {
            'lifetime': 30,
            'technology': 'refinery'
        },
        'cement': {
            'lifetime': 25,
            'technology': 'cement_plant'
        }
    }

    for industry in industries:
        data = industry_data[industry]
        lifetime = data['lifetime']
        technology = data['technology']
        file_demand = f"../data/hard_to_abate/set_carriers/{industry}/demand.csv"
        #generate_existing_capacities(file_demand, lifetime, technology)

    file_demand = "../data/hard_to_abate/set_carriers/ammonia/demand.csv"
    lifetime = 30
    technology = 'ASU'
    #generate_existing_capacity_ASU(file_demand, lifetime, technology)

    file_demand = "../data/hard_to_abate/set_carriers/steel/demand.csv"
    lifetime_BF_BOF = 40
    lifetime_EAF = 25
    lifetime_DRI = 25
    generate_existing_capacities_steel(file_demand, lifetime_BF_BOF, lifetime_EAF, lifetime_DRI)

    file_path_hydrogen = "existing_capacities_input/existing_capacity_hydrogen.csv"
    #existing_capacities_hydrogen(file_path_hydrogen)

    file_path_carbon_capture = "existing_capacities_input/existing_capacity_carbon_capture.csv"
    #existing_capacities_carbon_capture(file_path_carbon_capture)

    file_path_steel = "existing_capacities_input/existing_capacity_steel.csv"
    #existing_capacities_steel(file_path_steel)

    file_path_cement = "existing_capacities_input/existing_capacity_cement.csv"
    #existing_capacities_cement(file_path_cement)

    file_path_methanol = "existing_capacities_input/existing_capacity_petrochemicals.xlsx"
    sheet_name_methanol = "SFI_ALD_Petrochemicals_EUNA"
    #existing_capacities_methanol(file_path_methanol, sheet_name_methanol)

    file_path_ammonia = "existing_capacities_input/existing_capacity_petrochemicals.xlsx"
    sheet_name_ammonia = "SFI_ALD_Petrochemicals_EUNA"
    #existing_capacities_ammonia(file_path_ammonia, sheet_name_ammonia)

    industry = "cement"
    demand_file = f"../data/hard_to_abate/set_carriers/{industry}/demand.csv"
    capacity_file = "../data/hard_to_abate/set_technologies/set_conversion_technologies/cement_plant/capacity_existing.csv"
    capacity_file_steel = "../data/hard_to_abate/set_technologies/set_conversion_technologies/BF_BOF/capacity_existing.csv"
    #compare_capa_to_demand(demand_file, capacity_file, capacity_file_steel)

