"""
:Title:        ZEN-GARDEN results_JS5
:Created:      March-2024
:Authors:      Jara Spate (jspaete@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich
"""
import os
import itertools
import pandas as pd
import math
import numpy as np
import re
from pprint import pprint

import zen_garden.postprocess.results.gdf_US_JS as gdf_US_JS
from zen_garden.postprocess.results.unit_handling_JS import EnergySystemUnits
from zen_garden.postprocess.results.results_JS2 import Results
#from zen_garden.postprocess.results.results import Results as Results2
from zen_garden.postprocess.results.folder_structur_JS import create_folder, get_folder_path
import zen_garden.postprocess.results.plot_results as plot_results
import zen_garden.postprocess.results.state_mapping_JS as state_mapping_JS



def create_dict(column_name, unit, dict_info, dict_type, capacity_type=None, carriers=False):
    """
    Create a dictionary containing the filename, title, ylabel, and unit for the given data.

    :param column_name: The column name for the data.
    :param unit: The unit for the data.
    :param dict_info: The dictionary containing the information for the plot.
    :param dict_type: The dictionary type.
    :param capacity_type: The capacity type for the data (optional).
    :param carriers: Boolean indicating whether the data is for carriers.
    :return: The updated dictionary containing the filename, title, ylabel, and unit for the given data.
    """

    # Determine the filename, title, and ylabel based on the conditions
    if carriers:
        filename = f"{dict_type}]"
        title = f"{dict_type} {unit}"
        ylabel = f"{column_name} {unit}"
    elif capacity_type:
        filename = f"{dict_type}_{column_name}_{capacity_type}"
        title = f"{dict_type} {column_name} {capacity_type} {unit}"
        ylabel = f"{column_name} {capacity_type} {unit}"
    else:
        filename = f"{dict_type}_{column_name}"
        title = f"{dict_type} {column_name} {unit}"
        ylabel = f"{column_name} {unit}"

    # Ensure dict_info is not None
    if dict_info is None:
        dict_info = {}

    # Add the information to the dictionary
    dict_info[dict_type] = {
        'filename': filename,
        'title': title,
        'ylabel': ylabel,
        'unit': unit
    }

    return dict_info

def create_dict2(column_name, unit, dict_info, dict_type, capacity_type=None, carriers=False):
    """
    Create a dictionary containing the filename, title, ylabel, and unit for the given data.

    :param column_name: The column name for the data.
    :param unit: The unit for the data.
    :param dict_info: The dictionary containing the information for the plot.
    :param dict_type: The dictionary type.
    :param capacity_type: The capacity type for the data (optional).
    :param carriers: Boolean indicating whether the data is for carriers.
    :return: The updated dictionary containing the filename, title, ylabel, and unit for the given data.
    """

    # Determine the filename, title, and ylabel based on the conditions
    if carriers:
        filename = f"{dict_type}]"
        title = f"{dict_type}"
        ylabel = f"{column_name}"
    elif capacity_type:
        filename = f"{dict_type}_{column_name}_{capacity_type}"
        title = f"{dict_type} {column_name} {capacity_type}"
        ylabel = f"{column_name} {capacity_type} {unit}"
    else:
        filename = f"{dict_type}_{column_name}"
        title = f"{dict_type} {column_name}"
        ylabel = f"{column_name}"

    # Ensure dict_info is not None
    if dict_info is None:
        dict_info = {}

    # Add the information to the dictionary
    dict_info[dict_type] = {
        'filename': filename,
        'title': title,
        'ylabel': ylabel,
        'unit': unit
    }

    return dict_info

def handle_transport(res, component, carrier, energy_system_dir, scenario):
    """
    Handles transport technology data and creates a DataFrame.

    :param res: Results object containing the system results.
    :param component: Component type ('flow_transport_in' or 'flow_transport_out').
    :param carrier: Carrier for the transport technology.
    :param energy_system_dir: Path to the energy system directory.
    :param scenario: Scenario for filtering the data.
    :return: DataFrame containing transport technology data.
    """
    current_dir = os.getcwd()
    # Load the edges from the energy system directory
    filename_edges = 'set_edges.csv'
    #edges_path = os.path.join(filename_edges)
    edges_path = os.path.join(energy_system_dir, filename_edges)
    edges = pd.read_csv(edges_path)

    # Create the DataFrame for the transport technology
    df_transport_in_nodes = pd.DataFrame()

    # Iterate over the nodes and create the DataFrame
    for node in res.results[scenario]["system"]['set_nodes']:
        if node in edges['node_from'].values:
            # Get the full time series for the transport technology at the node and scenario
            full_ts_node = res.get_full_ts("flow_transport", node=node, scenario=scenario)

            # Edit the carrier flows for the transport technology such that in and out flows are separated
            if component == "flow_transport_in":
                df_transport_node = res.edit_carrier_flows(full_ts_node, node, carrier, "in", scenario)
            elif component == "flow_transport_out":
                df_transport_node = res.edit_carrier_flows(full_ts_node, node, carrier, "out", scenario)
            else:
                raise ValueError("Invalid component type. Must be 'flow_transport_in' or 'flow_transport_out'.")

            # Set the index name to 'technology'
            df_transport_node.index.name = "technology"
            if df_transport_node.empty:
                continue

            # Insert the carrier and node columns
            df_transport_node.insert(1, 'carrier', carrier)
            df_transport_node.insert(2, 'node', node)

            # Concatenate the current DataFrame with the previous ones
            df_transport_in_nodes = pd.concat([df_transport_in_nodes, df_transport_node], axis=0)

    return df_transport_in_nodes




def create_df_components(res, carrier, scenario):
    '''
    Create a DataFrame containing all components for a given scenario.
    param res: Results object containing the system results.
    param energy_system_dir: Path to the energy system directory.
    param scenario: Scenario for filtering the data.
    return: DataFrame containing all components for a given scenario.
    '''

    component_mapping_carrier = {
            'water': ["flow_storage_charge", "flow_storage_discharge", "flow_conversion_input", "flow_conversion_output"],
            'electricity': ["flow_import","flow_storage_charge", "flow_storage_discharge" , "flow_conversion_input", "flow_conversion_output"],
            'diesel': ["flow_import", "flow_conversion_input", "flow_conversion_output"],
            'irrigation_water': ["flow_conversion_input", "flow_conversion_output", "demand"],
            'blue_water': ["flow_import", "flow_conversion_input", "flow_conversion_output"]

        }
    components = component_mapping_carrier[carrier]
    print("\n")
    print ("Carrier: ", carrier)
    print ("Components: ", components)

    # Create an empty DataFrame to store the results
    df_full_components = pd.DataFrame()
    # Iterate over the components and create the DataFrame
    for component in components:
        #handle transport technology as in and out flows are separated
        df_component = res.get_full_ts(component, scenario=scenario)

        # Reset the index to ensure proper indexing
        df_component.reset_index(inplace=True)
        df_component.insert(0, 'variable', component)

        if 'technology' not in df_component.columns:
            # Add technology column at the first position if it is not present, treat it as an index
            df_component.insert(1, 'technology', 'no_technology')
        if 'carrier' not in df_component.columns:
            # Add carrier column with default value as NaN
            df_component['carrier'] = pd.NA

        # Now set the values for the 'carrier' column based on the 'technology' column
        df_component.loc[df_component['technology'] == 'battery', 'carrier'] = 'electricity'
        df_component.loc[df_component['technology'] == 'water_storage', 'carrier'] = 'water'



        # Concatenate the current dataframe with the previous ones
        df_full_components = pd.concat([df_full_components, df_component], axis=0)

    # Reset index to ensure proper indexing
    df_full_components.reset_index(drop=True, inplace=True)
    df_full_components.set_index(['variable', 'technology', 'carrier', 'node'], inplace=True)

    return df_full_components

def create_df_components_sum(df_full_components):
    '''
    Create a DataFrame containing the sum of all components.

    variable: The variable name like 'flow_import','flow_transport_out'
    technology: The technology name like 'PV', 'battery'
    carrier: The carrier name like 'electricity', 'water'

    param df_full_components: DataFrame containing all components for a given scenario.
    return: DataFrame containing the sum of all components.
    '''
    # Group the DataFrame by the variable, technology, and carrier and sum the values
    df_full_components_sum = df_full_components.groupby(["variable", "technology", "carrier"]).sum()
    return df_full_components_sum




def create_map(directory, df, dict_tech_type, column_name, county=True, default_vmax=None):
    """
    Create a map for the given data using the US map.

    :param directory: The directory where the map will be saved.
    :param df: The DataFrame containing the data.
    :param dict_tech_type: The dictionary containing the information for the plot.
    :param column_name: The column name containing the data.
    :param default_vmax: The default vmax for the colorbar.
    """
    if county:
        us_gdf = gdf_US_JS.create_county_US()
        df_map = df.groupby(["county_code"])[column_name].sum().reset_index()
    else:
        us_gdf = gdf_US_JS.create_US()
        df_map = df.groupby(["State_Code"])[column_name].sum().reset_index()
    plot_results.plot_map_data(directory, df_map, us_gdf, column_name, dict_tech_type, default_vmax=default_vmax)



def create_variable_name(row):
    """
    Create the variable name for the DataFrame.

    :param row: The row of the DataFrame.
    :return: The variable name for the DataFrame.
    """
    if row['technology'] == 'no_technology':
        return f"{row['variable']}, {row['carrier']}"
    else:
        return f"{row['variable']}, {row['technology']}, {row['carrier']}"



def create_df_map_carrier(res, carrier, energy_system_dir, scenario, county=True):
    """
    Create the DataFrame for the carrier.

    :param res: Results object containing the system results.
    :param carrier: The carrier for the DataFrame.
    :param energy_system_dir: Path to the energy system directory.
    :param scenario: Scenario for filtering the data.
    :return: DataFrame containing the carrier data.
    """
    # Create the DataFrame and restructure it
    df_components_map = create_df_components(res, carrier, energy_system_dir, scenario)
    df_components_map.reset_index(inplace=True)
    df_components_map['variable_name'] = df_components_map.apply(create_variable_name, axis=1)

    # Filter and process DataFrame for the specific carrier
    df_carrier = df_components_map[df_components_map['carrier'] == carrier]
    df_carrier.drop(columns=['technology', 'carrier'], inplace=True)
    df_carrier.set_index(['variable_name', 'node', 'variable'], inplace=True)

    summed_values = df_carrier.sum(axis=1)  # Sum the values
    summed_values_df = summed_values.to_frame()  # Convert the Series to a DataFrame

    # Reset the index and rename the columns
    summed_values_df.rename(columns={0: "summed_values"}, inplace=True)
    summed_values_df.reset_index(inplace=True)
    if county:
        summed_values_df.rename(columns={'node': "county_code"}, inplace=True)
    else:
        summed_values_df.rename(columns={'node': "State_Code"}, inplace=True)

    return summed_values_df




def create_maps_capacity(res, directory, energy_system_dir, scenario, county=True, default_vmax=None):
    """
    Create the maps for the capacity of the technologies.

    :param res: Results object containing the system results.
    :param directory: The directory where the map will be saved.
    :param energy_system_dir: Path to the energy system directory.
    :param scenario: Scenario for filtering the data.
    :return: None
    """
    # Get the full time series for the capacity addition
    df_capacity_full = res.get_full_ts("capacity_addition", scenario=scenario)
    df_capacity_full.reset_index(inplace=True)
    if county:
        df_capacity_full.rename(columns={0: 'capacity', 'location': 'county_code'}, inplace=True)
    else:
        df_capacity_full.rename(columns={0: 'capacity', 'location': 'State_Code'}, inplace=True)



    # Define the tech_types and remove the 'power_line' technology
    tech_types = df_capacity_full['technology'].unique()
    tech_types = [tech for tech in tech_types if tech != 'power_line']

    # Define the filename and boolean variables
    filename_default = 'default_values_capacities.csv'
    bool_default_vmax = False
    if scenario:
        path_default_vmax = os.path.join(os.path.dirname(directory), filename_default)
    else:
        path_default_vmax = os.path.join(directory, filename_default)
        default_vmax = None

    # Check if the default vmax is available, if yes load it; otherwise, create the DataFrame and save it
    if os.path.exists(path_default_vmax):
        default_vmax_df = pd.read_csv(path_default_vmax)
        bool_default_vmax = True

    # Create an instance of EnergySystemUnits
    energy_units = EnergySystemUnits(energy_system_dir)
    # Call the create_unit_dictionary_capacity method on the instance
    unit_dictionary = energy_units.create_unit_dictionary_capacity()

    # Create an empty dictionary to store the information for the tech_types
    tech_type_info = {}

    # Iterate over the tech_types and create the maps
    for tech_type in tech_types:
        df_capacity_full_tech = df_capacity_full[df_capacity_full['technology'] == tech_type]
        capacity_types = df_capacity_full_tech['capacity_type'].unique()

        for capacity_type in capacity_types:
            df_map = df_capacity_full_tech[df_capacity_full_tech['capacity_type'] == capacity_type]
            unit = unit_dictionary[(tech_type, capacity_type)]
            tech_type_info = create_dict2('capacity', unit, tech_type_info, tech_type, capacity_type)

            if bool_default_vmax:
                # Filter the DataFrame to get the default_vmax value for the specified tech_type and capacity_type
                filtered_df = default_vmax_df[(default_vmax_df['tech_types'] == tech_type) &
                                            (default_vmax_df['capacity_types'] == capacity_type)]

                # Check if there are any matching rows and extract the default_vmax value
                if not filtered_df.empty:
                    default_vmax = filtered_df['default_vmax'].values[0]
                else:
                    raise ValueError(f"No matching default_vmax found for tech_type: {tech_type} and capacity_type: {capacity_type}")

            if tech_type == 'PV':
                create_map(directory, df_map, tech_type_info[tech_type], column_name='capacity', county=county, default_vmax=default_vmax)
            elif tech_type in ['battery', 'water_storage'] and capacity_type == 'power':
                continue # Skip the battery and water storage power capacity
            else:
                create_map(directory, df_map, tech_type_info[tech_type], column_name='capacity', county=county, default_vmax=default_vmax)


def create_maps_carrier(res, directory, carrier, energy_system_dir, scenario, county=True):
    """
    Create the maps for the carrier.

    :param res: Results object containing the system results.
    :param directory: The directory where the map will be saved.
    :param carrier: The carrier for the map.
    :param energy_system_dir: Path to the energy system directory.
    :param scenario: Scenario for filtering the data.
    :return: None
    """
    # Create an instance of EnergySystemUnits
    energy_units = EnergySystemUnits(energy_system_dir)

    # Call the create_unit_dictionary_carrier method on the instance
    unit_dictionary_carrier = energy_units.create_unit_dictionary_carrier()
    unit = unit_dictionary_carrier[carrier]

    summed_values_df_carrier = create_df_map_carrier(res, carrier, energy_system_dir, scenario)
    component_types = summed_values_df_carrier['variable_name'].unique()

    # Define the filename and boolean variables
    filename_components = f'default_values_{carrier}.csv'
    bool_default_vmax = False
    if scenario:
        path_default_vmax = os.path.join(os.path.dirname(directory), filename_components)
    else:
        path_default_vmax = os.path.join(directory, filename_components)



    # Check if the default vmax is available, if yes load it; otherwise, create the DataFrame and save it
    if os.path.exists(path_default_vmax):
        default_vmax_df = pd.read_csv(path_default_vmax)
        bool_default_vmax = True

    # Iterate over the component types and create the maps
    for component_type in component_types:
        summed_values_df_comp = summed_values_df_carrier[summed_values_df_carrier['variable_name'] == component_type]
        variable = summed_values_df_comp['variable'].iloc[0]

        # Create a dictionary containing the filename, title, ylabel, and unit for the given data
        component_type_info = create_dict2(variable, unit, {}, component_type, capacity_type=None, carriers=True)

        # If default vmax is available, use it; otherwise, plot the map and calculate vmax
        if bool_default_vmax:
            # Filter the DataFrame to get the default_vmax value for the specified component_type
            filtered_df = default_vmax_df[default_vmax_df['component_types'] == component_type]

            # Check if there are any matching rows and extract the default_vmax value
            if not filtered_df.empty:
                default_vmax = filtered_df['default_vmax'].values[0]
                column_name = 'summed_values'

                # Plot the map
                create_map(directory, summed_values_df_comp, component_type_info[component_type], column_name, county, default_vmax)
            else:
                raise ValueError(f"No matching default_vmax found for component_type: {component_type}")

        else:
            create_map(directory, summed_values_df_comp, component_type_info[component_type], column_name='summed_values', county=county)  # plot the map




def create_df_vmax_carrier(res, directory, carrier, energy_system_dir, scenarios):
    """
    Create the maps for the carrier.

    :param res: Results object containing the system results.
    :param directory: The directory where the map will be saved.
    :param carrier: The carrier for the map.
    :param energy_system_dir: Path to the energy system directory.
    :param scenarios: List of scenarios for filtering the data.
    :return: None
    """
    # Define the filename and boolean variables
    filename_components = f'default_values_{carrier}.csv'
    save_components_types = os.path.join(directory, filename_components)

    vmax_dfs = pd.DataFrame(columns=["component_types"])

    for scenario in scenarios:
        print("\nScenario: ", scenario)
        summed_values_df_carrier = create_df_map_carrier(res, carrier, energy_system_dir, scenario)
        component_types = summed_values_df_carrier['variable_name'].unique()
        vmax_df = pd.DataFrame(component_types, columns=["component_types"])

        # Iterate over the component types and create the maps
        for component_type in component_types:
            summed_values_df_comp = summed_values_df_carrier[summed_values_df_carrier['variable_name'] == component_type]

            vmax = summed_values_df_comp['summed_values'].max()
            vmax_df.loc[vmax_df['component_types'] == component_type, f'vmax_{scenario}'] = vmax

        # Merge vmax_df with vmax_dfs
        vmax_dfs = vmax_dfs.merge(vmax_df, on="component_types", how="outer")


    #Find the maximum vmax value for each component type
    vmax_dfs['default_vmax'] = vmax_dfs.filter(like='vmax').max(axis=1)
    # Round the maximum vmax value for each component type to the nearest thousand
    vmax_dfs['default_vmax'] = vmax_dfs['default_vmax'].apply(lambda x: round(x / (10 ** (math.floor(math.log10(x)) - 1))) * (10 ** (math.floor(math.log10(x)) - 1)) if x != 0 else 0)
    vmax_dfs.to_csv(save_components_types, index=False)



def create_df_vmax_capacity(res, directory, energy_system_dir, scenarios, county=True):
    """
    Create the maps for the capacity of the technologies.

    :param res: Results object containing the system results.
    :param directory: The directory where the map will be saved.
    :param energy_system_dir: Path to the energy system directory.
    :param scenarios: List of scenarios for filtering the data.
    :return: None
    """

    # Define the filename and boolean variables
    filename_default = 'default_values_capacities.csv'
    save_components_types = os.path.join(directory, filename_default)
    vmax_dfs = pd.DataFrame(columns=['tech_types', 'capacity_types'])


    for scenario in scenarios:
        # Get the full time series for the capacity addition
        df_capacity_full = res.get_full_ts("capacity_addition", scenario=scenario)
        df_capacity_full.reset_index(inplace=True)
        if county:
            df_capacity_full.rename(columns={0: 'capacity', 'location': 'county_code'}, inplace=True)
        else:
            df_capacity_full.rename(columns={0: 'capacity', 'location': 'State_Code'}, inplace=True)

        # Define the tech_types and remove the 'power_line' technology
        tech_types = df_capacity_full['technology'].unique()
        tech_types = [tech for tech in tech_types if tech != 'power_line']

        unique_combinations = list(itertools.product(tech_types, df_capacity_full['capacity_type'].unique()))
        vmax_df = pd.DataFrame(unique_combinations, columns=['tech_types', 'capacity_types'])

        # Iterate over the tech_types and create the maps
        for tech_type in tech_types:
            df_capacity_full_tech = df_capacity_full[df_capacity_full['technology'] == tech_type]
            capacity_types = df_capacity_full_tech['capacity_type'].unique()

            for capacity_type in capacity_types:
                df_map = df_capacity_full_tech[df_capacity_full_tech['capacity_type'] == capacity_type]

                vmax = df_map['capacity'].max()
                vmax_df.loc[(vmax_df['tech_types'] == tech_type) & (vmax_df['capacity_types'] == capacity_type), f'vmax_{scenario}'] = vmax

        # Merge vmax_df with vmax_dfs
        vmax_dfs = vmax_dfs.merge(vmax_df, on=['tech_types', 'capacity_types'], how="outer")

    # Find the maximum vmax value for each component type
    vmax_dfs['default_vmax'] = vmax_dfs.filter(like='vmax').max(axis=1)
    vmax_dfs.dropna(subset=['default_vmax'], inplace=True)
    # Round the maximum vmax value for each component type to the nearest thousand
    vmax_dfs['default_vmax'] = vmax_dfs['default_vmax'].apply(lambda x: round(x / (10 ** (math.floor(math.log10(x)) - 1))) * (10 ** (math.floor(math.log10(x)) - 1)) if x != 0 else 0)

    vmax_dfs.to_csv(save_components_types, index=False)


def compare_and_store_vmax(directory1, directory2, carrier, key_word1, key_word2):
    """
    Compare vmax values from two different directories and store the higher values.

    :param directory1: The directory where the first map is saved.
    :param directory2: The directory where the second map is saved.
    :param carrier: The carrier for the map.
    :param key_word1: Keyword describing the first dataset.
    :param key_word2: Keyword describing the second dataset.
    :return: None
    """

    # Load the resulting CSV files
    vmax_df1 = pd.read_csv(os.path.join(directory1, f'default_values_{carrier}.csv'))
    vmax_df2 = pd.read_csv(os.path.join(directory2, f'default_values_{carrier}.csv'))
    if carrier == 'capacities':
        merged_df = vmax_df1.merge(vmax_df2, on=['tech_types','capacity_types'], suffixes=(f'_{key_word1}', f'_{key_word2}'))
    else:
        # Merge the two dataframes on 'component_types'
        merged_df = vmax_df1.merge(vmax_df2, on='component_types', suffixes=(f'_{key_word1}', f'_{key_word2}'))

    # Create a new column for the maximum values
    merged_df[f'default_vmax_{key_word1}'] = merged_df[f'default_vmax_{key_word1}'].fillna(0)
    merged_df[f'default_vmax_{key_word2}'] = merged_df[f'default_vmax_{key_word2}'].fillna(0)
    merged_df['default_vmax'] = merged_df[[f'default_vmax_{key_word1}', f'default_vmax_{key_word2}']].max(axis=1)

    # Save the final merged dataframe to the first directory
    final_filename1 = f'default_values_{carrier}_{key_word1}.csv'
    final_filename2 = f'default_values_{carrier}_{key_word2}.csv'
    merged_df.to_csv(os.path.join(directory1, final_filename1), index=False)
    merged_df.to_csv(os.path.join(directory2, final_filename2), index=False)

def compare_vmax(filename1, filename2, carriers, key_word1, key_word2):
    """
    Compare the vmax of two optimization results and store the biggest vmax
    param
    """
    directory1 = os.path.join("../data/outputs", filename1)
    directory2 = os.path.join("../data/outputs", filename2)
    for carrier in carriers:
        compare_and_store_vmax(directory1, directory2, carrier, key_word1, key_word2)

def create_vmax_scenarios(filename, carriers, path_energy_system, county):
    """
    Create the vmax for the carriers and capacities for the given scenarios
    """

    # Load the results object
    directory = os.path.join("../outputs", filename)
    directory = os.path.join("../outputs", filename)
    res_basic = Results(directory)
    # Get the scenarios
    scenarios = [scenario for scenario in os.listdir(directory) if scenario.startswith('scenario_')]

    if not scenarios:
        scenarios = [None]

    # Create the dataframes for the vmax of the carriers and capacities
    for carrier in carriers:
        create_df_vmax_carrier(res_basic, directory, carrier,  path_energy_system, scenarios)
    create_df_vmax_capacity(res_basic, directory, path_energy_system, scenarios, county)

def plot_energy_balances_carriers(res_basic, node, carriers, directory, scenario=None, short=False, save_fig=False):
    '''Visualise the energy balance at the node for the carrier in year 0 for different time periods'''
    for carrier in carriers:
        data_plot = create_df_components(res_basic, carrier, scenario)
        data_plot_masked = data_plot.mask((data_plot < 0) & (data_plot >= -0.001), 0)
        if carrier == 'electricity':
            data_plot_2 = create_df_components(res_basic, 'diesel', scenario)
            data_plot = pd.concat([data_plot, data_plot_2], axis=0)
        plot_results.plot_energy_balance_JS2(data_plot_masked, node, carrier, 0, directory, scenario, short, save_fig)



def filter_capacities_state(res_basic, folder, df_tech_cap):
    """
    Filter the capacities for each state and only the technology and capacity type given in the capacities_boxplot.csv
    """
    # Get the full time series of the capacity addition
    df = res_basic.get_full_ts("capacity_addition")
    df.reset_index(inplace=True)
    # Drop unnecessary columns early
    df = df.drop(columns=["location"]).reset_index()

    # Rename the columns
    df = df.rename(columns={"level_0": "scenario", 0: "capacity"})

    # Sum the capacities for each scenario, technology, and capacity type
    group_columns = ['scenario', 'technology', 'capacity_type'] if 'scenario' in df.columns else ['folder', 'technology', 'capacity_type']
    df_sum = df.groupby(group_columns, as_index=False).agg({'capacity': 'sum'})

    # Add the folder name at the first position
    df_sum.insert(0, 'folder', folder)

    # Filter the technology and capacity type given in the capacities_boxplot.csv
    filtered_df = df_sum.merge(df_tech_cap, on=['technology', 'capacity_type'], how='inner')
    filtered_df['tech_type'] = filtered_df['technology'] + ', ' + filtered_df['capacity_type']

    # Pivot the DataFrame
    pivot_df = filtered_df.pivot(index=['folder', 'scenario'], columns='tech_type', values='capacity').reset_index()
    # Get the scenarios
    # scenarios = [scenario for scenario in os.listdir(directory) if scenario.startswith('scenario_')]

    df_co2_dict = res_basic.get_df('carbon_emissions_cumulative')
    df_cost_dict = res_basic.get_df('net_present_cost')

    # Convert the dictionary to a DataFrame
    df_co2 = pd.DataFrame.from_dict(df_co2_dict)
    df_cost = pd.DataFrame.from_dict(df_cost_dict)
    if 'carbon_emissions_cumulative' in df_co2.columns:
        df_co2.rename(columns={'carbon_emissions_cumulative': 'scenario_cost_optimal'}, inplace=True)
        df_cost.rename(columns={'net_present_cost': 'scenario_cost_optimal'}, inplace=True)
    # Drop the 'year' column
    if 'year' in df_co2.columns:
        df_co2.drop('year', axis=1, inplace=True)
    if 'year' in df_cost.columns:
        df_cost.drop('year', axis=1, inplace=True)

    # Reshape the DataFrame
    df_co2_2 = df_co2.melt(var_name="scenario", value_name="carbon_emissions_cumulative")
    df_cost_2 = df_cost.melt(var_name="scenario", value_name="net_present_cost")

    # Ensure the scenario column in capacities_df is a string
    pivot_df['scenario'] = pivot_df['scenario'].astype(str)

    # Merge capacities_df with df_co2 on 'scenario'
    merged_df = pivot_df.merge(df_co2_2, on='scenario', how='left')
    merged_df = merged_df.merge(df_cost_2, on='scenario', how='left')


    return merged_df




def create_maps_scenarios(filename, carriers, scenarios=None, county=True):
    """
    Create the maps for the different scenarios
    :param filename: Filename of the results
    :param scenarios: List of scenarios
    :param carriers: List of carriers
    :return: None
    """
    # Path to the energy system and load the results
    directory = os.path.join("../data/outputs", filename)
    # Split the filenam to only get the name of the last folder
    filename_energy_system = filename.split('/')[-1]

    path_energy_system = os.path.join('../data/input', filename_energy_system, "energy_system")
    res_basic = Results(directory)
    # Get the scenarios
    if scenarios is None:
        scenarios = [scenario for scenario in os.listdir(directory) if scenario.startswith('scenario_')]

    if not scenarios:
        scenarios = [None]


    # Create the maps for the different scenarios
    for scenario in scenarios:
        if scenario is not None:
            dir_scenario = os.path.join(directory, scenario)
        else:
            dir_scenario = directory
        for carrier in carriers:
            create_maps_carrier(res_basic, dir_scenario, carrier, path_energy_system, scenario, county)
        create_maps_capacity(res_basic, dir_scenario, path_energy_system, scenario, county)


def determine_time_steps(folder):
    match = re.search(r'_(\d+)$', folder)
    return int(match.group(1)) if match else None


def get_df_imports(res_basic,folder):
    # Get the full time series of the capacity addition
    df_flow_import = res_basic.get_full_ts("flow_import")

    # Perform the sum along the columns between 0 and 8759
    df_flow_import['flow_import'] = df_flow_import.loc[:, 0:8759].sum(axis=1).to_frame()

    # Get the summed values of the carriers
    df_flow_import_sum = df_flow_import[[ 'flow_import']]
    df_flow_import_sum.reset_index(inplace=True)
    df_flow_import_sum.rename(columns={'level_0': 'scenario'}, inplace=True)

    # Filter for the carriers 'diesel' and 'electricity'
    df_filtered = df_flow_import_sum[(df_flow_import_sum['carrier'] == 'electricity') | (df_flow_import_sum['carrier'] == 'diesel')]
    if 'scenario' not in df_filtered.columns:
        df_filtered['scenario'] = 'no_scenario'
    df_grouped = df_filtered.groupby(['scenario', 'carrier']).sum()
    df_grouped = df_grouped.drop(columns=['node'])
    df_grouped.reset_index(inplace=True)
    # Add at the first position a new column with the folder name
    df_grouped.insert(0, 'folder', folder)

        # Pivot the DataFrame
    pivot_df = df_grouped.pivot(index=['folder', 'scenario'], columns='carrier', values='flow_import').reset_index()
    # Get the scenarios
    # scenarios = [scenario for scenario in os.listdir(directory) if scenario.startswith('scenario_')]

    df_co2_dict = res_basic.get_df('carbon_emissions_cumulative')
    df_cost_dict = res_basic.get_df('net_present_cost')

    # Convert the dictionary to a DataFrame
    df_co2 = pd.DataFrame.from_dict(df_co2_dict)
    df_cost = pd.DataFrame.from_dict(df_cost_dict)
    if 'carbon_emissions_cumulative' in df_co2.columns:
        df_co2.rename(columns={'carbon_emissions_cumulative': 'no_scenario'}, inplace=True)
        df_cost.rename(columns={'net_present_cost': 'no_scenario'}, inplace=True)
    # Drop the 'year' column
    if 'year' in df_co2.columns:
        df_co2.drop('year', axis=1, inplace=True)
    if 'year' in df_cost.columns:
        df_cost.drop('year', axis=1, inplace=True)

    # Reshape the DataFrame
    df_co2_2 = df_co2.melt(var_name="scenario", value_name="carbon_emissions_cumulative")
    df_cost_2 = df_cost.melt(var_name="scenario", value_name="net_present_cost")

    # Ensure the scenario column in capacities_df is a string
    pivot_df['scenario'] = pivot_df['scenario'].astype(str)

    # Merge capacities_df with df_co2 on 'scenario'
    merged_df = pivot_df.merge(df_co2_2, on='scenario', how='left')
    merged_df = merged_df.merge(df_cost_2, on='scenario', how='left')
    print(merged_df)
    return merged_df



def get_cost_co2(res, folder, directory):
    """
    Get the cost and CO2 dataframes for the scenarios
    """

    # Initialize an empty dataframe to concatenate into
    combined_dfs = pd.DataFrame()
    # Get the scenarios
    scenarios = [scenario for scenario in os.listdir(directory) if scenario.startswith('scenario_')]
    if not scenarios:
        scenarios = [None]

    for scenario in scenarios:
        df_cost = res.get_df('net_present_cost', scenario=scenario)
        df_co2 = res.get_df('carbon_emissions_cumulative', scenario=scenario)

        # Concatenate cost and CO2 dataframes for the current scenario
        df_cost_co2 = pd.concat([df_cost, df_co2], axis=1)

        # Add the scenario name to the dataframe at the first position
        df_cost_co2.insert(0, 'scenario', scenario)

        # Concatenate the combined dataframe into the overall dataframe
        combined_dfs = pd.concat([combined_dfs, df_cost_co2], axis=0)

    combined_dfs.reset_index(inplace=True)
    if 'folder' not in combined_dfs.columns:
        combined_dfs.insert(0, 'folder', folder)
    # Add at the first position a new column with the folder name

    return combined_dfs


def get_costs(res_basic, folder, directory):
    costs = ["cost_capex_total", "cost_opex_total", "cost_carbon_emissions_total", "cost_carrier_total", "carbon_emissions_cumulative"]
    scenarios_cost_dfs = pd.DataFrame()

    # Get the scenarios
    scenarios = [scenario for scenario in os.listdir(directory) if scenario.startswith('scenario_')]
    if not scenarios:
        scenarios = [None]

    for scenario in scenarios:
        cost_dfs = pd.DataFrame()  # Initialize an empty DataFrame
        for cost in costs:
            df_cost = res_basic.get_df(cost, scenario=scenario).to_frame()
            # Reset index to ensure proper concatenation
            df_cost.reset_index(drop=True, inplace=True)
            # Concatenate with existing data
            cost_dfs = pd.concat([cost_dfs, df_cost], axis=1)
        # Add scenario column to the DataFrame
        cost_dfs["scenario"] = scenario
        # Append to the final DataFrame
        scenarios_cost_dfs = pd.concat([scenarios_cost_dfs, cost_dfs], ignore_index=True)

    # Add at the first position a new column with the folder name
    scenarios_cost_dfs.insert(0, 'folder', folder)
    return scenarios_cost_dfs


def get_info_system(res, scenario, print_system=False, print_nodes=False):
    system = Results2.get_system(res, scenario_name=scenario)
    if print_system:
        print("Complete informations about this system:")
        pprint(system)
    if print_nodes:
        print("Nodes in the system:")
        pprint(system.set_nodes)

    print("Carriers in the system:")
    pprint(system.set_carriers)

    print("\nTechnologies in the system:")
    pprint(system.set_technologies)

    print("\nTotal hours per year:")
    pprint(system.total_hours_per_year)

    print("\nAggregated time steps per year:")
    pprint(system.aggregated_time_steps_per_year)

    print("\nConduct time series aggregation setting:")
    pprint(system.conduct_time_series_aggregation)

    print("\nConduct scenario analysis setting:")
    pprint(system.conduct_scenario_analysis)

def filter_boxplot_no_parent_folder(folder, output_path, specific_scenario_name, filter_component, df_tech_cap=None):
    dfs = pd.DataFrame()
    is_scenario = True

    # Function to determine filter_df based on filter_component
    def get_filter_df(res_basic, folder, directory):
        if filter_component == 'flow_import':
            return get_df_imports(res_basic, folder)
        elif filter_component == 'cost_co2':
            return get_cost_co2(res_basic, folder, directory)
        elif filter_component == 'costs':
            return get_costs(res_basic, folder, directory)
        elif filter_component == 'capacities':
            if df_tech_cap is None:
                raise ValueError("The dataframe df_tech_cap is needed for the component capacities")
            return filter_capacities_state(res_basic, folder, df_tech_cap)


    directory = os.path.join(output_path, folder)
    res_basic = Results(directory)

    filter_df = get_filter_df(res_basic, folder, directory)



    # Check if there is a column named scenario
    if 'scenario' not in filter_df.columns or filter_df['scenario'].isna().all():
        filter_df['scenario'] = 'no_scenario'
        is_scenario = False



    # Concatenate the filtered dataframe to the main dataframe
    dfs = pd.concat([dfs, filter_df], axis=0)
    # Define the scenario_name based on specific_scenario_name
    def create_scenario_name(row):
        if specific_scenario_name == 'folder':
            if not is_scenario:
                return row['folder']
            else:
                scenario = row['scenario'].replace('scenario_', '').replace(',no_grid', '').replace('co2', ' co2')
                folder = row['folder'].replace('county_CA_1206_', '').replace('_no_grid_tradeoffs', '')
                return f"{scenario}, {folder}, {row['scenario']}"
        elif specific_scenario_name == 'analysis_110724':
            return 'analysis_110724'
        else:
            return row['scenario']

    # Apply the scenario_name creation logic and update scenario_name and scenario columns
    dfs['scenario_name'] = dfs.apply(create_scenario_name, axis=1)
    dfs['scenario'] = dfs.apply(lambda row: row['scenario'].replace('scenario_', ''), axis=1)


    # Drop the 'year' column if it exists
    if 'year' in dfs.columns:
        dfs.drop('year', axis=1, inplace=True)

    # Reorder the columns
    column_order = ['folder', 'scenario_name', 'scenario']
    dfs = dfs[column_order + [col for col in dfs.columns if col not in column_order]]

    return dfs

def filter_boxplot(parent_folder, folders, specific_scenario_name, filter_component, df_tech_cap=None):
    dfs = pd.DataFrame()
    is_scenario = True

    # Function to determine filter_df based on filter_component
    def get_filter_df(res_basic, folder, directory):
        if filter_component == 'flow_import':
            return get_df_imports(res_basic, folder)
        elif filter_component == 'cost_co2':
            return get_cost_co2(res_basic, folder, directory)
        elif filter_component == 'costs':
            return get_costs(res_basic, folder, directory)
        elif filter_component == 'capacities':
            if df_tech_cap is None:
                raise ValueError("The dataframe df_tech_cap is needed for the component capacities")
            return filter_capacities_state(res_basic, folder, df_tech_cap)

    for folder in folders:
        directory = os.path.join("../../../outputs", parent_folder)
        directory = os.path.join("../../../outputs", parent_folder)
        res_basic = Results(directory)

        filter_df = get_filter_df(res_basic, folder, directory)

        # Add the time_steps column
        filter_df['time_steps'] = determine_time_steps(folder)

        # Check if there is a column named scenario
        if 'scenario' not in filter_df.columns:
            filter_df['scenario'] = ''
            is_scenario = False

        # Create the 'grid' column based on the condition
        filter_df['grid'] = filter_df['folder'].apply(lambda x: 'w/o grid' if 'no_grid' in x else 'with grid')

        # Concatenate the filtered dataframe to the main dataframe
        dfs = pd.concat([dfs, filter_df], axis=0)

    # Define the scenario_name based on specific_scenario_name
    def create_scenario_name(row):
        if specific_scenario_name == 'time_series_anlaysis':
            return f"{row['time_steps']},{row['grid']},{row['scenario']}"
        elif specific_scenario_name == 'folder':
            if not is_scenario:
                return row['folder']
            else:
                scenario = row['scenario'].replace('scenario_', '').replace(',no_grid', '').replace('co2', ' co2')
                folder = row['folder'].replace('county_CA_1206_', '').replace('_no_grid_tradeoffs', '')
                return f"{scenario}, {folder}, {row['scenario']}"
        elif specific_scenario_name == 'analysis_1806':
            scenario = row['scenario'].replace('scenario_', '').replace(',no_grid', '').replace('co2', ' co2')
            return f"{row['time_steps']}, {row['grid']}"
        else:
            return row['scenario']

    # Apply the scenario_name creation logic and update scenario_name and scenario columns
    dfs['scenario_name'] = dfs.apply(create_scenario_name, axis=1)
    dfs['scenario'] = dfs.apply(lambda row: row['scenario'].replace('scenario_', ''), axis=1)


    # Drop the 'year' column if it exists
    if 'year' in dfs.columns:
        dfs.drop('year', axis=1, inplace=True)

    # Reorder the columns
    column_order = ['folder', 'scenario_name', 'scenario']
    dfs = dfs[column_order + [col for col in dfs.columns if col not in column_order]]

    return dfs


def filter_pareto_group(df, custom_order, pareto_group, delete_empty_scenario=True):
    # Convert the 'scenario_name' column to a categorical type with the specified order
    df['scenario'] = pd.Categorical(df['scenario'], categories=custom_order, ordered=True)


    df['pareto_group'] = df[pareto_group]

    # if delete_empty_scenario:
    #     #df = df[df['scenario'].notna() & (df['scenario'] != '') & (df['scenario'] != '100') & (df['scenario'] != '95')]
    #     #df = df[(df['scenario'] != '100')]


    # Sort the DataFrame by the 'scenario_name' column
    df = df.sort_values(by=['scenario'])

    return df



def get_co2cost(res_basic, scenario, scenario_name):
    varaiables = ['carbon_emissions_cumulative', 'cost_capex_total', 'cost_opex_total', 'cost_carrier_total', 'cost_carbon_emissions_total', 'carbon_emissions_carrier_total', 'carbon_emissions_technology_total', 'net_present_cost']

    df_variables = pd.DataFrame()
    for variable in varaiables:
        df = res_basic.get_df(variable, scenario)
        df_variables = pd.concat([df_variables, df], axis=1)
    df_variables_T = df_variables.T
    df_variables_T.reset_index(inplace=True)
    df_variables_T.rename(columns={'index': 'variable'}, inplace=True)
    df_variables_T.rename(columns={0: scenario_name}, inplace=True)
    print(df_variables_T.head())
    return df_variables_T


def get_df_variables(res_basic, scenario, scenario_name):

    dict_tech_carrier = {'battery': 'electricity', 'water_storage': 'water', 'PV': 'electricity', 'diesel_WP': 'diesel', 'el_WP': 'electricity', 'irrigation_sys': 'water'}

    var2 = ['capacity_existing', 'capacity_addition', 'flow_import']

    df_var2 = pd.DataFrame()

    for variable in var2:
        if variable == 'flow_import':
            df = res_basic.get_full_ts(variable)
            df_sum = df.sum(axis=1).to_frame()
            df_sum.reset_index(inplace=True)
            if scenario:
                df_sum= df_sum[df_sum['level_0'] == scenario]
                df_sum.drop(columns='level_0', inplace=True)
            # Ensure the 'carrier' column is available and correctly handled
            if 'carrier' in df_sum.columns:
                df_grouped = df_sum[['carrier', 0]].groupby(['carrier']).sum()
                df_grouped['technology'] = 'none'
                df_grouped['capacity_type'] = 'none'
                df_grouped['variable'] = variable
                df_grouped.reset_index(inplace=True)
                df_grouped.rename(columns={'index': 'carrier'}, inplace=True)
                df_grouped.rename(columns={0: scenario_name}, inplace=True)
        else:
            df = res_basic.get_df(variable, scenario).to_frame()
            df.reset_index(inplace=True)
            df_grouped = df[['technology', 'capacity_type', variable]].groupby(['technology', 'capacity_type']).sum()
            df_grouped['carrier'] = df_grouped.index.get_level_values('technology').map(dict_tech_carrier)
            df_grouped['variable'] = variable
            df_grouped.rename(columns={variable: scenario_name}, inplace=True)
            df_grouped.reset_index(inplace=True)

        if not df_grouped.empty:
            # df_grouped = df_grouped[(df_grouped.T != 0).any()]
            # df_grouped = df_grouped.dropna()
            df_var2 = pd.concat([df_var2, df_grouped], axis=0)
    columns_order = ['carrier', 'technology', 'capacity_type', 'variable', scenario_name]
    df_var2 = df_var2[columns_order]
    df_var2.reset_index(inplace=True)
    df_var2.drop(columns='index', inplace=True)
    print(df_var2.head())
    return df_var2



def import_flow_data(folder, output_path, scenarios, column_name, filter_carriers, list_folders=None):
    # Create and prepare the US counties data
    us_counties = gdf_US_JS.create_county_US()
    us_counties.rename(columns={'county_code': 'node'}, inplace=True)


    combined_data = pd.DataFrame()
    if list_folders is None:
        directory = os.path.join(output_path, folder)
        res = Results(directory)
        df = res.get_full_ts(column_name)

        # Sum across columns and reset index
        df_sum = df.sum(axis=1).to_frame()
        df_sum.reset_index(inplace=True)

        # Filter and concatenate data for each scenario
        df_combined = pd.concat([df_sum[df_sum['level_0'] == scenario] for scenario in scenarios])

        # Pivot the DataFrame
        df_pivot = df_combined.pivot_table(index=['carrier', 'node'], columns='level_0', values=0)

        # Flatten the MultiIndex columns
        df_pivot.columns = [f'{column_name}_{col}' for col in df_pivot.columns]
        df_pivot.reset_index(inplace=True)

        # Combine all data
        combined_data = pd.concat([combined_data, df_pivot], axis=0)
    else:
        for folder_temp in list_folders:
            directory = os.path.join(output_path, folder_temp)
            res = Results(directory)
            df = res.get_full_ts(column_name)

            # Sum across columns and reset index
            df_sum = df.sum(axis=1).to_frame()
            df_sum.reset_index(inplace=True)

            # Filter and concatenate data for each scenario
            df_combined = pd.concat([df_sum[df_sum['level_0'] == scenario] for scenario in scenarios])

            # Pivot the DataFrame
            df_pivot = df_combined.pivot_table(index=['carrier', 'node'], columns='level_0', values=0)

            # Flatten the MultiIndex columns
            df_pivot.columns = [f'{column_name}_{col}' for col in df_pivot.columns]
            df_pivot.reset_index(inplace=True)

            # Combine all data
            combined_data = pd.concat([combined_data, df_pivot], axis=0)

    # Merge with US counties data to include state information
    merged_data = pd.merge(combined_data, us_counties[['node', 'state']], on='node', how='left')
    merged_data.reset_index(inplace=True)

    # Group by state and carrier, summing the values
    columns = [col for col in merged_data.columns if col != 'node']
    grouped_data = merged_data[columns].groupby(['state', 'carrier']).sum()
    grouped_data.reset_index(inplace=True)
    grouped_data.drop(columns=['index'], inplace=True)

    # Filter for specific carriers
    filtered_grouped_data = grouped_data[grouped_data['carrier'].isin(filter_carriers)]

    # Map state abbreviations to full names
    filtered_grouped_data = state_mapping_JS.reverse_mapping(filtered_grouped_data, 'state', 'state_full', 'full_to_abbr')


    return grouped_data, filtered_grouped_data


def merge_data_folders(parent_folder, output_path, column_name, scenarios):
    """
    Merge data from multiple folders within the parent folder.

    Parameters:
    parent_folder (str): The parent directory containing subfolders with data.
    column_name (str): The name of the column to be extracted and merged.
    scenarios (list of str): List of scenario names to process.

    Returns:
    DataFrame: A DataFrame containing the merged data from all folders.
    """
    # List all subfolders in the parent folder, excluding certain files and folders
    subfolders = [folder for folder in os.listdir(os.path.join(output_path, parent_folder))
                  if folder != 'Figures' and not folder.endswith('.csv') and not folder.endswith('.png')]

    # Initialize an empty DataFrame to hold all merged data
    merged_data = pd.DataFrame()

    # If there are folders, which named 'scenario_' we will use them
    if 'scenario_' in subfolders:
        directory = os.path.join(output_path, parent_folder)
        results = Results(directory)
        scenario_data = pd.DataFrame()
        for scenario in scenarios:


            scenario_df = results.get_df(column_name, scenario=scenario).to_frame()
            scenario_df.rename(columns={column_name: f'{column_name}_{scenario}'}, inplace=True)
            scenario_data = pd.concat([scenario_data, scenario_df], axis=1)

        merged_data = pd.concat([merged_data, scenario_data], axis=0)
    else:
        for subfolder in subfolders:
            directory = os.path.join(output_path, parent_folder, subfolder)
            results = Results(directory)
            scenario_data = pd.DataFrame()

            for scenario in scenarios:
                scenario_df = results.get_df(column_name, scenario=scenario).to_frame()
                scenario_df.rename(columns={column_name: f'{column_name}_{scenario}'}, inplace=True)
                scenario_data = pd.concat([scenario_data, scenario_df], axis=1)

            merged_data = pd.concat([merged_data, scenario_data], axis=0)

    return merged_data


def aggregate_data(df_merged, us_counties):
    """
    Aggregate the merged data by state, technology, and capacity type.

    Parameters:
    df_merged (DataFrame): The merged data DataFrame.
    us_counties (DataFrame): DataFrame containing county information including 'node' and 'state'.

    Returns:
    DataFrame: A DataFrame containing the aggregated data filtered by specific technology and capacity types.
    """
    # Reset index and merge with county information
    df_merged_reset = df_merged.reset_index()
    df_merged_reset.rename(columns={'location': 'node'}, inplace=True)
    df_merged_reset = pd.merge(df_merged_reset, us_counties[['node', 'state']], on='node', how='left')

    # Group by state, technology, and capacity type, then sum the capacity additions
    df_grouped = df_merged_reset.groupby(['state', 'technology', 'capacity_type']).sum().reset_index()

    # Define a dictionary for filtering specific technology and capacity type combinations
    tech_cap_filter = {'water_storage': 'energy', 'battery': 'energy', 'PV': 'power', 'diesel_WP': 'power', 'el_WP': 'power'}
    tech_cap_tuples = list(tech_cap_filter.items())

    # Filter the DataFrame based on the defined technology and capacity type combinations
    filtered_df = df_grouped[df_grouped[['technology', 'capacity_type']].apply(tuple, axis=1).isin(tech_cap_tuples)]

    return filtered_df

def aggregate_to_states(data_folder, result_path, target_column, scenario_list, folders_list=None):
    """
    Aggregates data from counties to states based on provided scenarios and merges with US county geometries.

    Args:
        data_folder (str): Path to the parent folder containing scenario data.
        result_path (str): Path to save the output results.
        target_column (str): The column name to be aggregated.
        scenario_list (list): List of scenarios to be processed.

    Returns:
        DataFrame: Aggregated data at the state level with relevant columns.
    """
    # Load US counties geometries and rename 'county_code' to 'node'
    us_counties = gdf_US_JS.create_county_US()
    us_counties.rename(columns={'county_code': 'node'}, inplace=True)

    if folders_list is None:
        # Merge data from multiple scenarios
        merged_data = merge_data_folders(data_folder, result_path, target_column, scenario_list)
        merged_data_dfs = merged_data.copy()
    else:
        merged_data_dfs = pd.DataFrame()
        for folder in folders_list:
            merged_data = merge_data_folders(folder, result_path, target_column, scenario_list)
            merged_data_dfs = pd.concat([merged_data_dfs, merged_data], axis=0)
    # Aggregate data based on US counties
    aggregated_data = aggregate_data(merged_data_dfs, us_counties)

    # Create a new column combining technology and capacity type
    aggregated_data['tech_cap'] = aggregated_data['technology'] + ', ' + aggregated_data['capacity_type']

    # Reverse map state abbreviations to full state names
    state_mapping_JS.reverse_mapping(aggregated_data, 'state', 'state_full', 'full_to_abbr')

    # Drop unnecessary columns
    aggregated_data.drop(columns=['technology', 'capacity_type'], inplace=True)

    return aggregated_data


def plot_pareto_front(folder, output_path, units, specific_scenario_name, custom_order, area, pareto_group, list_folders=None, save_fig=True):
    """
    Plots Pareto fronts for various cost and capacity components against CO2 emissions.

    Args:
        folder (str): Folder where the data is located.
        output_path (str): Path to save the output figures.
        specific_scenario_name (str): Name of the specific scenario being analyzed.
        custom_order (list): Custom order for Pareto front filtering.
        area (str): The area being analyzed.
        pareto_group (str): The group used for Pareto front filtering.
        save_fig (bool): Flag to save the figures. Defaults to True.

    Returns:
        None
    """



    # Define parameters for the first set of plots
    cost_components = [
       ('cost_co2', f'Net Present Cost vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'net_present_cost', 'Net Present Cost'),
       ('costs', f'CAPEX vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'cost_capex_total', 'Capex'),
       ('costs', f'OPEX vs. CO2 Emissions', 'cost_opex_total', 'Opex'),
       ('costs', f'Cost Carrier vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'cost_carrier_total', 'Cost Carrier'),
       ('costs', f'Cost Carbon Emissions vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'cost_carbon_emissions_total', 'Cost Carbon Emissions')
    ]

    # Generate the first set of plots
    for filter_component, title, y_axis, y_axis_label in cost_components:
        if list_folders is None:
            df = filter_boxplot_no_parent_folder(folder, output_path, specific_scenario_name, filter_component=filter_component)
            pareto_df = filter_pareto_group(df, custom_order, pareto_group)
            result_df = pareto_df.copy()
        else:
            pareto_dfs = pd.DataFrame()
            for folder_temp in list_folders:
                df = filter_boxplot_no_parent_folder(folder_temp, output_path, specific_scenario_name, filter_component=filter_component)
                pareto_df = filter_pareto_group(df, custom_order, pareto_group)


                pareto_dfs = pd.concat([pareto_dfs, pareto_df])

        # Group by 'scenario' and sum up the relevant columns
        result_df = pareto_dfs.groupby('scenario', as_index=False).agg({
            'folder': 'first',
            'scenario_name': 'first',
            'pareto_group': 'first',
            y_axis: 'sum',
            'carbon_emissions_cumulative': 'sum'
        })
        plot_results.plot_pareto_front(
            result_df, folder, output_path, title=title, unit_co2=units['co2'],
            unit_y_axis=units['cost'], y_axis=y_axis, y_axis_label=y_axis_label, save_fig=save_fig
        )

    # Define parameters for the capacities plots
    capacity_components = [
        ('water_storage, energy', f'Water Storage Capacity vs. $\\mathrm{{CO_2}}$ Emissions in {area}', units['water_energy'], 'Installed Capacity'),
        ('battery, energy', f'Battery Capacity vs. $\\mathrm{{CO_2}}$ Emissions in {area}', units['energy'], 'Installed Capacity'),
        ('PV, power', f'PV Capacity vs. $\\mathrm{{CO_2}}$ Emissions in {area}', units['power'], 'Installed Capacity')
    ]

    # Read the capacities boxplot data
    df_tech_cap = pd.read_csv("capacities_boxplot_JS.csv")

    # Generate the capacities plots
    if list_folders is None:
        capacities_df = filter_boxplot_no_parent_folder(folder, output_path, filter_component='capacities', df_tech_cap=df_tech_cap, specific_scenario_name=specific_scenario_name)
        capacities_pareto_df = filter_pareto_group(capacities_df, custom_order, pareto_group)
        capacities_pareto_dfs = capacities_pareto_df.copy()
    else:
        capacities_pareto_dfs = pd.DataFrame()
        for folder_temp in list_folders:
            capacities_df = filter_boxplot_no_parent_folder(folder_temp, output_path, filter_component='capacities', df_tech_cap=df_tech_cap, specific_scenario_name=specific_scenario_name)
            capacities_pareto_df = filter_pareto_group(capacities_df, custom_order, pareto_group)
            capacities_pareto_dfs = pd.concat([capacities_pareto_dfs, capacities_pareto_df])

        columns_first = ['folder', 'scenario_name', 'pareto_group']
        columns_sum = [col for col in capacities_pareto_dfs.columns if col not in columns_first + ['scenario']]

        # Create the aggregation dictionary
        agg_dict = {col: 'first' for col in columns_first}
        agg_dict.update({col: 'sum' for col in columns_sum})

        # Group by 'scenario' and apply the aggregation
        result_capacities_pareto_dfs = capacities_pareto_dfs.groupby('scenario', as_index=False).agg(agg_dict)

    for y_axis, title, unit_y_axis, y_axis_label in capacity_components:
        plot_results.plot_pareto_front(
            result_capacities_pareto_dfs, folder, output_path, title=title, unit_co2=units['co2'],
            unit_y_axis=unit_y_axis, y_axis=y_axis, y_axis_label=y_axis_label, save_fig=save_fig
        )

    # Define parameters for the flow import plots
    flow_import_components = [
        ('electricity', f'Imported Electricity vs. $\\mathrm{{CO_2}}$ Emissions in {area}', units['energy'], 'Import'),
        ('diesel', f'Imported Diesel vs. $\\mathrm{{CO_2}}$ Emissions in {area}', units['energy'], 'Import'),
    ]

    # Generate the flow import plots
    if list_folders is None:
        flow_import_df = filter_boxplot_no_parent_folder(folder, output_path, filter_component='flow_import', specific_scenario_name='analysis_1806')
        flow_import_pareto_df = filter_pareto_group(flow_import_df, custom_order, pareto_group)
        result_flow_import_pareto_dfs = flow_import_pareto_df.copy()
    else:
        flow_import_pareto_dfs = pd.DataFrame()
        for folder_temp in list_folders:
            flow_import_df = filter_boxplot_no_parent_folder(folder_temp, output_path, filter_component='flow_import', specific_scenario_name='analysis_1806')
            flow_import_pareto_df = filter_pareto_group(flow_import_df, custom_order, pareto_group)
            flow_import_pareto_dfs = pd.concat([flow_import_pareto_dfs, flow_import_pareto_df])
        columns_first = ['folder', 'scenario_name', 'pareto_group']
        columns_sum = [col for col in flow_import_pareto_dfs.columns if col not in columns_first + ['scenario']]
        # Create the aggregation dictionary
        agg_dict = {col: 'first' for col in columns_first}
        agg_dict.update({col: 'sum' for col in columns_sum})
        # Group by 'scenario' and apply the aggregation
        result_flow_import_pareto_dfs = flow_import_pareto_dfs.groupby('scenario', as_index=False).agg(agg_dict)

    for y_axis, title, unit_y_axis, y_axis_label in flow_import_components:
        plot_results.plot_pareto_front(
            result_flow_import_pareto_dfs, folder, output_path, title=title, unit_co2=units['co2'],
            unit_y_axis=unit_y_axis, y_axis=y_axis, y_axis_label=y_axis_label, save_fig=save_fig
        )



def plot_pareto_front2(folder, output_path, units, specific_scenario_name, custom_order, area, pareto_group, list_folders=None, save_fig=True):
    """
    Plots Pareto fronts for various cost and capacity components against CO2 emissions.

    Args:
        folder (str): Folder where the data is located.
        output_path (str): Path to save the output figures.
        specific_scenario_name (str): Name of the specific scenario being analyzed.
        custom_order (list): Custom order for Pareto front filtering.
        area (str): The area being analyzed.
        pareto_group (str): The group used for Pareto front filtering.
        save_fig (bool): Flag to save the figures. Defaults to True.

    Returns:
        None
    """



    # Define parameters for the first set of plots
    cost_components = [
       ('costs', f'CAPEX vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'cost_capex_total', 'Capex'),
       ('costs', f'OPEX vs. CO2 Emissions', 'cost_opex_total', 'Opex'),
       ('costs', f'Cost Carrier vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'cost_carrier_total', 'Cost Carrier'),
       ('costs', f'Cost Carbon Emissions vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'cost_carbon_emissions_total', 'Cost Carbon Emissions')
    ]

    # Generate the first set of plots
    for filter_component, title, y_axis, y_axis_label in cost_components:
        if list_folders is None:
            df = filter_boxplot_no_parent_folder(folder, output_path, specific_scenario_name, filter_component=filter_component)
            pareto_df = filter_pareto_group(df, custom_order, pareto_group)
            result_df = pareto_df.copy()
        else:
            pareto_dfs = pd.DataFrame()
            for folder_temp in list_folders:
                df = filter_boxplot_no_parent_folder(folder_temp, output_path, specific_scenario_name, filter_component=filter_component)
                pareto_df = filter_pareto_group(df, custom_order, pareto_group)


                pareto_dfs = pd.concat([pareto_dfs, pareto_df])

        # Group by 'scenario' and sum up the relevant columns
        result_df = pareto_dfs.groupby('scenario', as_index=False).agg({
            'folder': 'first',
            'scenario_name': 'first',
            'pareto_group': 'first',
            y_axis: 'sum',
            'carbon_emissions_cumulative': 'sum'
        })
        plot_results.plot_pareto_front(
            result_df, folder, output_path, title=title, unit_co2=units['co2'],
            unit_y_axis=units['cost'], y_axis=y_axis, y_axis_label=y_axis_label, save_fig=save_fig
        )

def prepare_data_for_stacked_cost_plot(folder, list_folders, output_path, specific_scenario_name, filter_component=None):
    """
    Prepare the data for the stacked cost plot
    """

    if list_folders is None:
        df = filter_boxplot_no_parent_folder(folder, output_path, specific_scenario_name, filter_component=filter_component)
        result_dfs = df.copy()
    else:
        dfs = pd.DataFrame()
        for folder_temp in list_folders:
            df = filter_boxplot_no_parent_folder(folder_temp, output_path, specific_scenario_name, filter_component=filter_component)
            dfs = pd.concat([dfs, df])

        columns_first = ['folder', 'scenario_name']
        columns_sum = [col for col in dfs.columns if col not in columns_first + ['scenario']]

        # Create the aggregation dictionary
        agg_dict = {col: 'first' for col in columns_first}
        agg_dict.update({col: 'sum' for col in columns_sum})

        # Group by 'scenario' and apply the aggregation
        result_dfs = dfs.groupby('scenario', as_index=False).agg(agg_dict)
    # sort the dataframe by carbon_emissions_cumulative
    result_dfs = result_dfs.sort_values(by='carbon_emissions_cumulative')
    return result_dfs



def prepare_data_for_capacity_figure(folder, output_path, list_folders, specific_scenario_name):



    # Read the capacities boxplot data
    df_tech_cap = pd.read_csv("capacities_boxplot_JS.csv")

    # Generate the capacities plots
    if list_folders is None:
        capacities_df = filter_boxplot_no_parent_folder(folder, output_path, filter_component='capacities', df_tech_cap=df_tech_cap, specific_scenario_name=specific_scenario_name)
        capacities_dfs = capacities_df.copy()
    else:
        capacities_dfs = pd.DataFrame()
        for folder_temp in list_folders:
            capacities_df = filter_boxplot_no_parent_folder(folder_temp, output_path, filter_component='capacities', df_tech_cap=df_tech_cap, specific_scenario_name=specific_scenario_name)
            capacities_dfs = pd.concat([capacities_dfs, capacities_df])

    columns_first = ['folder', 'scenario_name']
    columns_sum = [col for col in capacities_dfs.columns if col not in columns_first + ['scenario']]

    # Create the aggregation dictionary
    agg_dict = {col: 'first' for col in columns_first}
    agg_dict.update({col: 'sum' for col in columns_sum})

    # Group by 'scenario' and apply the aggregation
    result_capacities_dfs = capacities_dfs.groupby('scenario', as_index=False).agg(agg_dict)
    # sort the dataframe by carbon_emissions_cumulative
    result_capacities_dfs = result_capacities_dfs.sort_values(by='carbon_emissions_cumulative')
    return result_capacities_dfs


def prepare_data_for_flow_import_stacked(folder, list_folders, output_path, specific_scenario_name):
    # Define parameters for the flow import plots

    # Generate the flow import plots
    if list_folders is None:
        flow_import_df = filter_boxplot_no_parent_folder(folder, output_path, filter_component='flow_import', specific_scenario_name=specific_scenario_name)
        result_flow_import_dfs = flow_import_df.copy()
    else:
        flow_import_dfs = pd.DataFrame()
        for folder_temp in list_folders:
            flow_import_df = filter_boxplot_no_parent_folder(folder_temp, output_path, filter_component='flow_import', specific_scenario_name=specific_scenario_name)
            flow_import_dfs = pd.concat([flow_import_dfs, flow_import_df])
        columns_first = ['folder', 'scenario_name']
        columns_sum = [col for col in flow_import_dfs.columns if col not in columns_first + ['scenario']]
        # Create the aggregation dictionary
        agg_dict = {col: 'first' for col in columns_first}
        agg_dict.update({col: 'sum' for col in columns_sum})
        # Group by 'scenario' and apply the aggregation
        result_flow_import_dfs = flow_import_dfs.groupby('scenario', as_index=False).agg(agg_dict)
    #sort the values using carbon_emissions_cumulative
    result_flow_import_dfs.sort_values(by='carbon_emissions_cumulative', inplace=True)
    return result_flow_import_dfs




def prepare_data_for_map_plot(folder, list_folders, output_path, BAU=False, scenarios=None):
    # Process capacity and demand data from multiple folders
    df_capacities = pd.DataFrame()
    df_demands = pd.DataFrame()
    df_flow_imports = pd.DataFrame()

    for folder_temp in list_folders:
        directory = os.path.join(output_path, folder_temp)
        res = Results(directory)

        # Get capacity addition data
        df_capacity = res.get_full_ts("capacity_addition")
        df_capacity.reset_index(inplace=True)
        df_capacity.rename(columns={0: 'capacity', 'location': 'county_code', 'level_0': 'scenario'}, inplace=True)
        #insert row with 'carrier' column
        df_capacity.insert(1, 'carrier', '')
        df_capacities = pd.concat([df_capacities, df_capacity])

        # Get demand data and filter for 'irrigation_water'
        df_flow_import = res.get_full_ts('flow_import').sum(axis=1).reset_index()
        df_flow_import_filtered = df_flow_import[(df_flow_import['carrier'] == 'diesel') | (df_flow_import['carrier'] == 'electricity')]
        df_flow_import_filtered.rename(columns={0: 'flow_import', 'node': 'county_code', 'level_0': 'scenario'}, inplace=True)
        df_flow_import_filtered.insert(1, 'technology', '')
        df_flow_import_filtered.insert(2, 'capacity_type', '')
        df_flow_imports = pd.concat([df_flow_imports, df_flow_import_filtered])

        # Get demand data and filter for 'irrigation_water'
        df_demand = res.get_full_ts('demand').sum(axis=1).reset_index()
        df_demand_water = df_demand[df_demand['carrier'] == 'irrigation_water']
        df_demand_water.rename(columns={0: 'demand', 'node': 'county_code', 'level_0': 'scenario'}, inplace=True)
        df_demands = pd.concat([df_demands, df_demand_water])

    # Create US county geographical data
    us_gdf = gdf_US_JS.create_county_US()

    # Read technology capacity information
    df_tech_cap_info = pd.read_csv("capacities_boxplot_JS.csv")
    df_merge_cap_import = pd.concat([df_capacities, df_flow_imports], axis=0)
    if scenarios is None:
        if 'scenario' in df_merge_cap_import.columns:
            scenarios = df_merge_cap_import['scenario'].unique()
        elif BAU:
            df_merge_cap_import['scenario'] = 'BAU'
            df_demands['scenario'] = 'BAU'
            scenarios = ['BAU']
        else:
            scenarios = ['Cost-Optimal-Scenario']
    print(scenarios)

    # Define chunk sizes for splitting df_tech_cap_info
    chunk_sizes = [3, 2, 2]  # Define sizes for each chunk

    # Get the indices for each chunk
    start_idx = 0
    titles = ['Capacity Addition Weighted by Water Demand', 'Capacity Addition of Water Pumps Weighted by Water Demand', 'Imports of Energy Carriers Weighted by Water Demand']
    for i, chunk_size in enumerate(chunk_sizes):
        end_idx = start_idx + chunk_size
        df_tech_cap_chunk = df_tech_cap_info.iloc[start_idx:end_idx]
        start_idx = end_idx
        save_filename = f"maps_{i}.png"
        if BAU and i==0:
          continue
        #print(df_tech_cap_chunk)
        # Call plot_scenarios with the current chunk
        plot_results.plot_scenarios(scenarios, df_merge_cap_import, df_demands, df_tech_cap_chunk, us_gdf, output_path, folder, save_filename, titles[i])


def prepare_data_for_map_plot2(folder, list_folders, list_folders_BAU,output_path, BAU=False, scenarios=None):
    # Process capacity and demand data from multiple folders
    df_capacities = pd.DataFrame()
    df_demands = pd.DataFrame()
    df_flow_imports = pd.DataFrame()

    for folder_temp in list_folders:
        directory = os.path.join(output_path, folder_temp)
        res = Results(directory)

        # Get capacity addition data
        df_capacity = res.get_full_ts("capacity_addition")
        df_capacity.reset_index(inplace=True)
        df_capacity.rename(columns={0: 'capacity', 'location': 'county_code', 'level_0': 'scenario'}, inplace=True)
        #insert row with 'carrier' column
        df_capacity.insert(1, 'carrier', '')
        df_capacities = pd.concat([df_capacities, df_capacity])

        # Get demand data and filter for 'irrigation_water'
        df_flow_import = res.get_full_ts('flow_import').sum(axis=1).reset_index()
        df_flow_import_filtered = df_flow_import[(df_flow_import['carrier'] == 'diesel') | (df_flow_import['carrier'] == 'electricity')]
        df_flow_import_filtered.rename(columns={0: 'flow_import', 'node': 'county_code', 'level_0': 'scenario'}, inplace=True)
        df_flow_import_filtered.insert(1, 'technology', '')
        df_flow_import_filtered.insert(2, 'capacity_type', '')
        df_flow_imports = pd.concat([df_flow_imports, df_flow_import_filtered])

        # Get demand data and filter for 'irrigation_water'
        df_demand = res.get_full_ts('demand').sum(axis=1).reset_index()
        df_demand_water = df_demand[df_demand['carrier'] == 'irrigation_water']
        df_demand_water.rename(columns={0: 'demand', 'node': 'county_code', 'level_0': 'scenario'}, inplace=True)
        df_demands = pd.concat([df_demands, df_demand_water])

    # Create US county geographical data
    us_gdf = gdf_US_JS.create_county_US()

    # Read technology capacity information
    df_tech_cap_info = pd.read_csv("capacities_boxplot_JS.csv")
    df_merge_cap_import = pd.concat([df_capacities, df_flow_imports], axis=0)
    if scenarios is None:
        if 'scenario' in df_merge_cap_import.columns:
            scenarios = df_merge_cap_import['scenario'].unique()
        elif BAU:
            df_merge_cap_import['scenario'] = 'BAU'
            df_demands['scenario'] = 'BAU'
            scenarios = ['BAU']
        else:
            scenarios = ['Cost-Optimal-Scenario']
    print(scenarios)

    # Define chunk sizes for splitting df_tech_cap_info
    chunk_sizes = [3, 2, 2]  # Define sizes for each chunk

    # Get the indices for each chunk
    start_idx = 0
    titles = ['Capacity Addition Weighted by Water Demand', 'Capacity Addition of Water Pumps Weighted by Water Demand', 'Imports of Energy Carriers Weighted by Water Demand']
    for i, chunk_size in enumerate(chunk_sizes):
        end_idx = start_idx + chunk_size
        df_tech_cap_chunk = df_tech_cap_info.iloc[start_idx:end_idx]
        start_idx = end_idx
        save_filename = f"maps_{i}.png"
        if BAU and i==0:
          continue
        #print(df_tech_cap_chunk)
        # Call plot_scenarios with the current chunk
        plot_results.plot_scenarios(scenarios, df_merge_cap_import, df_demands, df_tech_cap_chunk, us_gdf, output_path, folder, save_filename, titles[i])



def create_co2_cost_point(output_path, list_folders):
    cumulative_point = np.zeros(2)
    for folder_temp in list_folders:
        directory = os.path.join(output_path, folder_temp)
        res = Results(directory)
        df_cost = pd.DataFrame(res.get_df('net_present_cost'))
        df_co2 = pd.DataFrame(res.get_df('carbon_emissions_cumulative'))

        # Get the point values for the current folder
        point = np.array([df_co2['carbon_emissions_cumulative'].iloc[0], df_cost['net_present_cost'].iloc[0]])

        # Add the current point values to the cumulative point
        cumulative_point += point
    print(point)
    print(cumulative_point)
    return cumulative_point


def prepare_data_for_plot_stacked_tech_car(list_folders, output_path, BAU=False):
    df_costs = pd.DataFrame()

    for folder_temp in list_folders:
        directory = os.path.join(output_path, folder_temp)
        res_basic = Results(directory)

        # Capex yearly
        df_capex_yearly = res_basic.get_full_ts("capex_yearly")
        #df_capex_yearly = df_capex_yearly[(df_capex_yearly > 0.001) | (df_capex_yearly < -0.001)]
        df_capex_yearly.reset_index(inplace=True)
        df_capex_yearly.rename(columns={0: 'capex_yearly'}, inplace=True)
        if 'level_0' in df_capex_yearly.columns:
            df_capex_yearly.rename(columns={'level_0': 'scenario'}, inplace=True)
            df_capex_yearly_sum = df_capex_yearly[['scenario', 'technology', 'capex_yearly']].groupby(['scenario', 'technology']).sum()
        else:
            df_capex_yearly_sum = df_capex_yearly[['technology','capex_yearly']].groupby(['technology']).sum()
            df_capex_yearly_sum['scenario'] = 'BAU'
        df_capex_yearly_sum = df_capex_yearly_sum.reset_index()



        # Opex
        df_opex_yearly = res_basic.get_full_ts("opex_yearly")
        #Delet the row if the data is between -0.001 and 0
        #df_opex_yearly = df_opex_yearly[(df_opex_yearly > 0.001) | (df_opex_yearly < -0.001)]
        df_opex_yearly.reset_index(inplace=True)
        df_opex_yearly.rename(columns={0: 'opex_yearly'}, inplace=True)
        if 'level_0' in df_opex_yearly.columns:
            df_opex_yearly.rename(columns={'level_0': 'scenario'}, inplace=True)
            df_opex_yearly_sum = df_opex_yearly[['scenario', 'technology', 'opex_yearly']].groupby(['scenario', 'technology']).sum()
        else:
            df_opex_yearly_sum = df_opex_yearly[['technology', 'opex_yearly']].groupby(['technology']).sum()
            df_opex_yearly_sum['scenario'] = 'BAU'
        df_opex_yearly_sum = df_opex_yearly_sum.reset_index()

        # Capex + Opex
        df_cost_technologies = pd.merge(df_capex_yearly_sum, df_opex_yearly_sum, on=['technology', 'scenario'], how='outer')
        df_cost_technologies['cost_technology'] = df_cost_technologies['capex_yearly'] + df_cost_technologies['opex_yearly']
        df_cost_technologies['technology/carrier'] = df_cost_technologies['technology']
        df_cost_technologies['cost'] = df_cost_technologies['cost_technology']



        # Cost Carrier
        data = res_basic.get_full_ts('cost_carrier')
        data = data[(data > 0.001) | (data < -0.001)]
        data_sum_node = data.sum(axis=1).reset_index()
        #Filter for blue_water, electricity and diesel
        data_sum_node = data_sum_node[data_sum_node['carrier'].isin(['blue_water', 'electricity', 'diesel'])]
        data_sum_node.rename(columns={0: 'cost_carrier'}, inplace=True)
        if 'level_0' in data_sum_node.columns:
            data_sum_node.rename(columns={'level_0': 'scenario'}, inplace=True)
            df_cost_carrier = data_sum_node[['scenario', 'carrier', 'cost_carrier']].groupby(['scenario', 'carrier']).sum().reset_index()
        else:
            df_cost_carrier = data_sum_node[['carrier', 'cost_carrier']].groupby(['carrier']).sum().reset_index()
            df_cost_carrier['scenario'] = 'BAU'

        df_cost_carrier['technology/carrier'] = df_cost_carrier['carrier']
        df_cost_carrier['cost']= df_cost_carrier['cost_carrier']

        df_cost = pd.concat([df_cost_technologies[['scenario','technology/carrier','cost']], df_cost_carrier[['scenario','technology/carrier','cost']]], axis=0)

        # Cost Carbon emission cumulative
        res_basic.get_df("cost_carbon_emissions_total")
        df_cost_co2_dict = res_basic.get_df('cost_carbon_emissions_total')
        df_cost_co2 = pd.DataFrame(df_cost_co2_dict).T
        df_cost_co2.reset_index(inplace=True)
        df_cost_co2.rename(columns={'index': 'scenario',0:'cost'}, inplace=True)
        df_cost_co2['technology/carrier'] = 'cost CO2'


        # Carbon emission cumulative
        df_co2_dict = res_basic.get_df('carbon_emissions_cumulative')
        df_co2 = pd.DataFrame(df_co2_dict).T
        df_co2.reset_index(inplace=True)
        df_co2.rename(columns={'index': 'scenario',0:'co2_emissions'}, inplace=True)

        # Net present Cost
        df_net_cost_dict = res_basic.get_df('net_present_cost')
        df_net_cost= pd.DataFrame(df_net_cost_dict).T
        df_net_cost.reset_index(inplace=True)
        df_net_cost.rename(columns={'index': 'scenario',0:'net_present_cost'}, inplace=True)
        if BAU:
            df_cost_co2['scenario'] = df_cost_co2['scenario'].replace('cost_carbon_emissions_total', 'BAU')
            df_net_cost['scenario'] = df_net_cost['scenario'].replace('net_present_cost', 'BAU')
            df_co2['scenario'] = df_co2['scenario'].replace('carbon_emissions_cumulative', 'BAU')


        df_cost = pd.concat([df_cost, df_cost_co2], axis=0)
        df_cost = pd.merge(df_cost, df_net_cost, on='scenario', how='outer')
        df_cost = pd.merge(df_cost, df_co2, on='scenario', how='outer')


        # Change the name of the technology/carrier in the df_cost
        df_tech_car = pd.read_csv('stacked_plot.csv')
        short_name_rename_dict = dict(zip(df_tech_car['short_name'], df_tech_car['full_name']))
        #df_cost_filter = df_cost[df_cost['technology/carrier'].isin(short_name_rename_dict.keys())]
        df_cost_filter = df_cost.copy()
        df_cost_filter['technology/carrier'] = df_cost_filter['technology/carrier'].replace(short_name_rename_dict)


        # Add the data to the final dataframe so cost_new = cost_old + cost_new for each technology/carrier
        df_costs = pd.concat([df_costs, df_cost_filter], axis=0)
        # Concatenate the data

    # Group by 'scenario' and 'technology/carrier' and sum the 'cost' and 'co2_emissions'
    df_costs_final = df_costs.groupby(['scenario', 'technology/carrier']).sum().reset_index()
    print(df_costs_final.head(10))
    return  df_costs_final

def prepare_data_for_map_plot_capacities_renewables(list_folders, output_path):
    # Process capacity and demand data from multiple folders
    df_capacities = pd.DataFrame()
    df_demands = pd.DataFrame()

    for folder_temp in list_folders:
        directory = os.path.join(output_path, folder_temp)
        res = Results(directory)

        # Get capacity addition data
        df_capacity = res.get_full_ts("capacity_addition")
        df_capacity.reset_index(inplace=True)
        df_capacity.rename(columns={0: 'capacity', 'location': 'county_code', 'level_0': 'scenario'}, inplace=True)
        #insert row with 'carrier' column
        df_capacity.insert(1, 'carrier', '')
        df_capacities = pd.concat([df_capacities, df_capacity])

        # Get demand data and filter for 'irrigation_water'
        df_demand = res.get_full_ts('demand').sum(axis=1).reset_index()
        df_demand_water = df_demand[df_demand['carrier'] == 'irrigation_water']
        df_demand_water.rename(columns={0: 'demand', 'node': 'county_code', 'level_0': 'scenario'}, inplace=True)
        df_demands = pd.concat([df_demands, df_demand_water])

    return df_capacities, df_demands

def process_folders(folder_list, output_path, scenario_name):
    df_demands = pd.DataFrame()
    df_flow_imports = pd.DataFrame()

    for folder_temp in folder_list:
        directory = os.path.join(output_path, folder_temp)
        res = Results(directory)

        # Process flow import data
        df_flow_import = res.get_full_ts('flow_import').sum(axis=1).reset_index()
        df_flow_import_filtered = df_flow_import[(df_flow_import['carrier'] == 'diesel') | (df_flow_import['carrier'] == 'electricity')]
        df_flow_import_filtered.rename(columns={0: 'flow_import', 'node': 'county_code', 'level_0': 'scenario'}, inplace=True)
        df_flow_import_filtered.insert(1, 'technology', '')
        df_flow_import_filtered.insert(2, 'capacity_type', '')
        df_flow_import_filtered['scenario'] = scenario_name
        df_flow_imports = pd.concat([df_flow_imports, df_flow_import_filtered])

        # Process demand data
        df_demand = res.get_full_ts('demand').sum(axis=1).reset_index()
        df_demand_water = df_demand[df_demand['carrier'] == 'irrigation_water']
        df_demand_water.rename(columns={0: 'demand', 'node': 'county_code', 'level_0': 'scenario'}, inplace=True)
        df_demand_water['scenario'] = scenario_name
        df_demands = pd.concat([df_demands, df_demand_water])

    return df_demands, df_flow_imports


def prepare_data_water_pumps_map(list_folders, output_path):
    # Process capacity and demand data from multiple folders
    df_capacities = pd.DataFrame()
    df_demands = pd.DataFrame()

    for folder_temp in list_folders:
        directory = os.path.join(output_path, folder_temp)
        res = Results(directory)

        # Get capacity addition data
        df_capacity = res.get_full_ts("capacity_addition")
        df_capacity.reset_index(inplace=True)
        df_capacity.rename(columns={0: 'capacity', 'location': 'county_code', 'level_0': 'scenario'}, inplace=True)
        #insert row with 'carrier' column
        df_capacity.insert(1, 'carrier', '')
        df_capacities = pd.concat([df_capacities, df_capacity])

        # Get demand data and filter for 'irrigation_water'
        df_demand = res.get_full_ts('demand').sum(axis=1).reset_index()
        df_demand_water = df_demand[df_demand['carrier'] == 'irrigation_water']
        df_demand_water.rename(columns={0: 'demand', 'node': 'county_code', 'level_0': 'scenario'}, inplace=True)
        df_demands = pd.concat([df_demands, df_demand_water])
    return df_capacities, df_demands


def plot_all_maps(list_folders, list_folders_BAU, output_path, folder):
    us_gdf = gdf_US_JS.create_county_US()
    keys = ['capacity', 'import', 'water_pumps']
    df_tech_cap_info_files = ['capacities_renewable_JS.csv', 'imports.csv', 'water_pump.csv']
    save_filenames = ['maps_capacity_renewables.png', 'maps_flow_import.png', 'maps_water_pumps.png']
    scenarios_list = [
        ['scenario_0', 'scenario_'],
        ['cost-optimal', 'BAU'],
        ['scenario_0', 'scenario_']
    ]

    # Initialize an empty dictionary
    data_dict = {}

    # Populate the dictionary
    for i, key in enumerate(keys):
        data_dict[key] = {
            'save_filename': save_filenames[i],
            'df_tech_cap_info': pd.read_csv(df_tech_cap_info_files[i]),  # Load DataFrame from CSV file
            'scenarios': scenarios_list[i]
        }
    key = 'capacity'
    df_capacities, df_demands = prepare_data_for_map_plot_capacities_renewables(list_folders, output_path)
    df_tech_cap_info = data_dict[key]['df_tech_cap_info']
    scenarios = data_dict[key]['scenarios']
    print(scenarios)
    save_filename = data_dict[key]['save_filename']

    plot_results.plot_map_capacities_JS(df_capacities, df_demands, df_tech_cap_info, us_gdf, output_path, folder, save_filename, scenarios)

    key = 'import'
    df_tech_cap_info = data_dict[key]['df_tech_cap_info']
    scenarios = data_dict[key]['scenarios']
    save_filename = data_dict[key]['save_filename']

    # Process data for both sets of folders
    df_demands, df_flow_imports = process_folders(list_folders, output_path, 'cost-optimal')
    df_demands_BAU, df_flow_imports_BAU = process_folders(list_folders_BAU, output_path, 'BAU')

    # Change scenario names if necessary
    df_demands['scenario'] = df_demands['scenario'].replace('scenario_', 'cost-optimal')
    df_flow_imports['scenario'] = df_flow_imports['scenario'].replace('scenario_', 'cost-optimal')

    df_merge_import = pd.concat([df_flow_imports_BAU, df_flow_imports], axis=0)
    df_merge_demands = pd.concat([df_demands_BAU, df_demands], axis=0)


    plot_results.plot_map_imports(df_merge_import, df_merge_demands, df_tech_cap_info, us_gdf, output_path, folder, save_filename, scenarios)


    key = 'water_pumps'
    df_tech_cap_info = data_dict[key]['df_tech_cap_info']
    scenarios = data_dict[key]['scenarios']
    save_filename = data_dict[key]['save_filename']

    df_capacities, df_demands = prepare_data_water_pumps_map(list_folders, output_path)

    plot_results.plot_map_water_pumps(df_capacities, df_demands, df_tech_cap_info, us_gdf, output_path, folder, save_filename, scenarios)
