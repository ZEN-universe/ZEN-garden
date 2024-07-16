import pandas as pd
import os
from zen_garden.postprocess.results.results_JS2 import Results
from zen_garden.postprocess.results import results_JS5 as results_JS
from zen_garden.postprocess.results import plot_results


def get_co2_emissions(res_basic, folder, output_path):
    """
    Retrieve and print the cumulative carbon emissions data.

    Args:
        res_basic (Results): An instance of the Results class containing the data.

    Returns:
        None
    """
    # Get the DataFrame containing cumulative carbon emissions
    df_co2_dict = res_basic.get_df('carbon_emissions_cumulative')
    # Print the carbon emissions data
    print(df_co2_dict)

    # Assuming df_co2_dict is a dictionary with appropriate data structure for a DataFrame
    df_co2 = pd.DataFrame(df_co2_dict)
    save_folder = os.path.join(output_path, folder, 'CSV-files')
    #create folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_file = os.path.join(save_folder, 'carbon_emissions_cumulative.csv')
    # Save the DataFrame to a CSV file
    df_co2.to_csv(save_file, index=False)


# #################################################################
def plot_energy_balance(res_basic, node, scenarios, directory, save_fig=True):
    """
    Plot energy balances for specified carriers and scenarios.

    Args:
        res_basic (Results): An instance of the Results class containing the data.
        node (str): The node for which the energy balance will be plotted.
        scenarios (list or None): List of scenarios to be plotted. If None, defaults to all scenarios in the directory.
        directory (str): The directory containing the scenario data.
        save_fig (bool): Flag to save the figures. Defaults to True.

    Returns:
        None
    """
    # Define the carriers to be plotted
    carriers = ['water', 'electricity']

    # If scenarios are not provided, get all scenarios starting with 'scenario_0'
#    if scenarios is None:
#        scenarios = [scenario for scenario in os.listdir(directory) if scenario.startswith('scenario_0')]

    # Plot energy balances for each scenario
    for scenario in scenarios:
        results_JS.plot_energy_balances_carriers(res_basic, node, carriers, directory, scenario=scenario, save_fig=save_fig)

def print_import_cost_carrier(res_basic, folder, output_path):
    """
    Process cost carrier and flow import data, summarize them, and save the results to CSV files.

    Parameters:
    res_basic: Object containing the methods to get time series data.
    output_path: Path where the CSV files will be saved.
    """

    # Get and summarize cost carrier data
    df_cost_carrier = res_basic.get_full_ts('cost_carrier')
    df_cost_carrier_sum = df_cost_carrier.sum(axis=1).reset_index()
    df_cost_carrier_sum.rename(columns={0: 'cost_carrier'}, inplace=True)
    if 'level_0' in df_cost_carrier_sum.columns:
        df_cost_carrier_sum.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_cost_carrier_sum_sum = df_cost_carrier_sum[['scenario', 'carrier', 'cost_carrier']].groupby(['scenario', 'carrier']).sum().reset_index()
    else:
        df_cost_carrier_sum_sum = df_cost_carrier_sum[['carrier', 'cost_carrier']].groupby(['carrier']).sum().reset_index()

    print('Costs per carrier')
    print(df_cost_carrier_sum_sum)

    # Get and summarize flow import data
    df_import = res_basic.get_full_ts('flow_import')
    df_import_sum = df_import.sum(axis=1).reset_index()
    df_import_sum.rename(columns={0: 'flow_import'}, inplace=True)
    if 'level_0' in df_import_sum.columns:
        df_import_sum.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_import_sum_sum = df_import_sum[['scenario', 'carrier', 'flow_import']].groupby(['scenario', 'carrier']).sum().reset_index()
    else:
        df_import_sum_sum = df_import_sum[['carrier', 'flow_import']].groupby(['carrier']).sum().reset_index()

    print('Import per carrier')
    print(df_import_sum_sum)

    # Create folder if it does not exist
    save_folder = os.path.join(output_path, folder, 'CSV-files')
    os.makedirs(save_folder, exist_ok=True)

    # Save the summarized data to CSV files
    df_import_sum_sum.to_csv(os.path.join(save_folder, 'import.csv'), index=False)
    df_cost_carrier_sum_sum.to_csv(os.path.join(save_folder, 'carrier.csv'), index=False)
    
        # Get and summarize flow import data
    df_demand = res_basic.get_full_ts('demand')
    df_demand_sum = df_demand.sum(axis=1).reset_index()
    df_demand_sum.rename(columns={0: 'demand'}, inplace=True)
    df_demand_sum_filtered = df_demand_sum[df_demand_sum['carrier'] == 'irrigation_water']
    if 'level_0' in df_demand_sum.columns:
        df_demand_sum.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_demand_sum_sum = df_demand_sum[['scenario', 'carrier', 'demand']].groupby(['scenario', 'carrier']).sum().reset_index()
        df_demand_sum_sum_filtered = df_demand_sum_sum[df_demand_sum_sum['carrier'] == 'irrigation_water']

    else:
        df_demand_sum_sum = df_demand_sum[['carrier', 'demand']].groupby(['carrier']).sum().reset_index()
        df_demand_sum_sum_filtered = df_demand_sum_sum[df_demand_sum_sum['carrier'] == 'irrigation_water']

    print('Demand per carrier')
    print(df_demand_sum_sum_filtered)

    # Create folder if it does not exist
    save_folder = os.path.join(output_path, folder, 'CSV-files')
    os.makedirs(save_folder, exist_ok=True)

    # Save the summarized data to CSV files
    df_demand_sum_sum_filtered.to_csv(os.path.join(save_folder, 'demand.csv'), index=False)
    df_demand_sum_filtered.to_csv(os.path.join(save_folder, 'demand_long.csv'), index=False)
    
    
    
    
    # Get and summarize flow import data
    df_flow_conversion_output = res_basic.get_full_ts('flow_conversion_output')
    df_flow_conversion_output_sum = df_flow_conversion_output.sum(axis=1).reset_index()
    df_flow_conversion_output_sum.rename(columns={0: 'flow_conversion_output'}, inplace=True)
    if 'level_0' in df_flow_conversion_output_sum.columns:
        df_flow_conversion_output_sum.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_flow_conversion_output_sum_sum = df_flow_conversion_output_sum[['scenario','technology', 'carrier', 'flow_conversion_output']].groupby(['scenario', 'technology', 'carrier']).sum().reset_index()

    else:
        df_flow_conversion_output_sum_sum = df_flow_conversion_output_sum[['technology','carrier', 'flow_conversion_output']].groupby(['technology', 'carrier']).sum().reset_index()

    print('Flow Conversion Output per carrier')
    print(df_flow_conversion_output_sum_sum)

    # Create folder if it does not exist
    save_folder = os.path.join(output_path, folder, 'CSV-files')
    os.makedirs(save_folder, exist_ok=True)

    # Save the summarized data to CSV files
    df_flow_conversion_output_sum_sum.to_csv(os.path.join(save_folder, 'flow_conversion_output.csv'), index=False)
    df_flow_conversion_output_sum.to_csv(os.path.join(save_folder, 'flow_conversion_output_long.csv'), index=False)



    # Get and summarize flow import data
    df_flow_conversion_input = res_basic.get_full_ts('flow_conversion_input')
    df_flow_conversion_input_sum = df_flow_conversion_input.sum(axis=1).reset_index()
    df_flow_conversion_input_sum.rename(columns={0: 'flow_conversion_input'}, inplace=True)
    if 'level_0' in df_flow_conversion_input_sum.columns:
        df_flow_conversion_input_sum.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_flow_conversion_input_sum_sum = df_flow_conversion_input_sum[['scenario','technology', 'carrier', 'flow_conversion_input']].groupby(['scenario', 'technology', 'carrier']).sum().reset_index()

    else:
        df_flow_conversion_input_sum_sum = df_flow_conversion_input_sum[['technology','carrier', 'flow_conversion_input']].groupby(['technology', 'carrier']).sum().reset_index()

    print('Flow Conversion Input per carrier')
    print(df_flow_conversion_input_sum_sum)

    # Create folder if it does not exist
    save_folder = os.path.join(output_path, folder, 'CSV-files')
    os.makedirs(save_folder, exist_ok=True)

    # Save the summarized data to CSV files
    df_flow_conversion_input_sum_sum.to_csv(os.path.join(save_folder, 'flow_conversion_input.csv'), index=False)
    df_flow_conversion_input_sum.to_csv(os.path.join(save_folder, 'flow_conversion_input_long.csv'), index=False)
####################################################################
####################################################################



def main():
    folder = "county_1607_5_large_unit"
    output_path = "../../../outputs/"
    directory = os.path.join(output_path, folder)
    res_basic = Results(directory)

    #get_co2_emissions(res_basic, folder, output_path)
    print_import_cost_carrier(res_basic, folder, output_path)

#    node = 'MS_BE009'
#    scenarios = ['scenario_','scenario_0']
#
#    plot_energy_balance(res_basic, node, scenarios, directory, save_fig=True)
#
#    #################################################################################
#    list_folders = ["county_1607_5_large_unit"]
#    area = 'United States'
#    custom_order = ['','0']
#    pareto_group = 'scenario_name'
#    specific_scenario_name = 'analysis_110724'
#    # Define unit conversions
#    units = {
#        'co2': 'tons',
#        'cost': '1000000*USD',
#        'water_energy': '1000*meter ** 3',
#        'energy': 'MWh',
#        'power': 'MW'
#    }
#    results_JS.plot_pareto_front(folder, output_path, units, specific_scenario_name, custom_order, area, pareto_group, list_folders=list_folders, save_fig=True)
#
#    #################################################################################
#
#    #scenarios = ['scenario_0','scenario_25','scenario_75','scenario_100']
#
#
#    df_caps_all_filtered = results_JS.aggregate_to_states(folder, output_path, 'capacity_addition', scenarios, list_folders)
#    plot_results.plot_boxplot_capacities_states(scenarios, df_caps_all_filtered, output_path, folder)
#    df_group, df_group_filtered = results_JS.import_flow_data(folder, output_path, scenarios, column_name='flow_import', filter_carriers= ['electricity', 'diesel'], list_folders= list_folders)
#    plot_results.plot_boxplot_energy_states(scenarios, df_group_filtered, folder, output_path, target_column='flow_import')
#    df_group_demand, df_group_filtered_demand = results_JS.import_flow_data(folder, output_path, scenarios, 'demand', filter_carriers=['irrigation_water'], list_folders= list_folders)
#    plot_results.plot_boxplot_energy_states(scenarios, df_group_filtered_demand, folder, output_path, 'demand')

if __name__ == "__main__":
    main()
