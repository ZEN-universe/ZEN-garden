import pandas as pd
import os
from zen_garden.postprocess.results.results_JS2 import Results
from zen_garden.postprocess.results import results_JS5 as results_JS
from zen_garden.postprocess.results import plot_results


def get_co2_emissions(res_basic):
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
    if scenarios is None:
        scenarios = [scenario for scenario in os.listdir(directory) if scenario.startswith('scenario_0')]

    # Plot energy balances for each scenario
    for scenario in scenarios:
        results_JS.plot_energy_balances_carriers(res_basic, node, carriers, directory, scenario=scenario, save_fig=save_fig)


####################################################################



def main():
    folder = "county_1107_288_1"
    output_path = "../../../outputs/"
    directory = os.path.join(output_path, folder)
    res_basic = Results(directory)

    #get_co2_emissions(res_basic)

    node = 'NE_BO015'
    scenarios = ['scenario_','scenario_0','scenario_25','scenario_75','scenario_100']

    # plot_energy_balance(res_basic, node, scenarios, directory, save_fig=True)

    #################################################################################
    list_folders = ["county_1107_288_1", "county_1107_288_2", "county_1107_288_3", "county_1107_288_4", "county_1107_288_5", "county_1107_288_6", "county_1107_288_7"]
    area = 'United States'
    custom_order = ['','100','75','25','0']
    pareto_group = 'scenario_name'
    specific_scenario_name = 'analysis_110724'
        # Define unit conversions
    units = {
        'co2': 'tons',
        'cost': '1000000*USD',
        'water_energy': '1000*meter ** 3',
        'energy': 'MWh',
        'power': 'MW'
    }
    #results_JS.plot_pareto_front(folder, output_path, units, specific_scenario_name, custom_order, area, pareto_group, list_folders=list_folders, save_fig=True)

    #################################################################################

    scenarios = ['scenario_0','scenario_25','scenario_75','scenario_100']


    df_caps_all_filtered = results_JS.aggregate_to_states(folder, output_path, 'capacity_addition', scenarios, list_folders)
    plot_results.plot_boxplot_capacities_states(scenarios, df_caps_all_filtered, output_path, folder)
    df_group, df_group_filtered = results_JS.import_flow_data(folder, output_path, scenarios, column_name='flow_import', filter_carriers= ['electricity', 'diesel'], list_folders= list_folders)
    plot_results.plot_boxplot_energy_states(scenarios, df_group_filtered, folder, output_path, target_column='flow_import')
    df_group_demand, df_group_filtered_demand = results_JS.import_flow_data(folder, output_path, scenarios, 'demand', filter_carriers=['irrigation_water'], list_folders= list_folders)
    plot_results.plot_boxplot_energy_states(scenarios, df_group_filtered_demand, folder, output_path, 'demand')

if __name__ == "__main__":
    main()
