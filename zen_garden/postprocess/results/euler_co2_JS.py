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


    # Create output path
    save_folder = os.path.join(output_path, folder, 'CSV-files')
    #create folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    # Net present cost
    df_co2_dict = res_basic.get_df('net_present_cost')
    # Print the carbon emissions data
    print(df_co2_dict)
    df_co2 = pd.DataFrame(df_co2_dict)
    # Save file
    save_file = os.path.join(save_folder, 'net_present_cost.csv')
    df_co2.to_csv(save_file, index=False)

    # Carbon emission cumulative
    df_co2_dict = res_basic.get_df('carbon_emissions_cumulative')
    # Print the carbon emissions data
    print(df_co2_dict)
    df_co2 = pd.DataFrame(df_co2_dict)
    # Save file
    save_file = os.path.join(save_folder, 'co2_cumulative.csv')
    df_co2.to_csv(save_file, index=False)

    #Carbon emission annual
    df_co2_annual_dict = res_basic.get_df('carbon_emissions_annual')
    print(df_co2_annual_dict)
    df_co2_annual = pd.DataFrame(df_co2_annual_dict)
    save_file = os.path.join(save_folder, 'carbon_emissions_annual.csv')
    df_co2_annual.to_csv(save_file, index=True)

    #cost carbon emissions
    df_cost_co2_dict = res_basic.get_df('cost_carbon_emissions_total')
    print(df_cost_co2_dict)
    df_cost_co2 = pd.DataFrame(df_cost_co2_dict)
    save_file = os.path.join(save_folder, 'cost_co2_total.csv')
    df_cost_co2.to_csv(save_file, index=True)

    # Capex tot
    df_capex_tot = res_basic.get_full_ts("cost_capex_total")
    save_file = os.path.join(save_folder, 'capex_total.csv')
    df_capex_tot.to_csv(save_file, index=True)


    # Capex
    df_capex = res_basic.get_full_ts("cost_capex")
    df_capex.reset_index(inplace=True)
    df_capex.rename(columns={0: 'cost_capex'}, inplace=True)
    if 'level_0' in df_capex.columns:
        df_capex.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_capex_sum = df_capex[['scenario', 'technology', 'capacity_type','cost_capex']].groupby(['scenario', 'technology', 'capacity_type']).sum()
    else:
        df_capex_sum = df_capex[['technology', 'capacity_type','cost_capex']].groupby(['technology', 'capacity_type']).sum()
    save_file = os.path.join(save_folder, 'capex.csv')
    df_capex_sum.to_csv(save_file, index=True)

    # Capex yearly
    df_capex_yearly = res_basic.get_full_ts("capex_yearly")
    df_capex_yearly.reset_index(inplace=True)
    df_capex_yearly.rename(columns={0: 'capex_yearly'}, inplace=True)
    if 'level_0' in df_capex_yearly.columns:
        df_capex_yearly.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_capex_yearly_sum = df_capex_yearly[['scenario', 'technology', 'capacity_type','capex_yearly']].groupby(['scenario', 'technology', 'capacity_type']).sum()
    else:
        df_capex_yearly_sum = df_capex_yearly[['technology', 'capacity_type','capex_yearly']].groupby(['technology', 'capacity_type']).sum()
    save_file = os.path.join(save_folder, 'capex_yearly.csv')
    df_capex_yearly_sum.to_csv(save_file, index=True)


    # Opex tot
    df_opex_tot = res_basic.get_full_ts("cost_opex_total")
    save_file = os.path.join(save_folder, 'opex_total.csv')
    df_opex_tot.to_csv(save_file, index=True)


    # Opex
    df_opex = res_basic.get_full_ts("cost_opex")
    df_opex.reset_index(inplace=True)
    df_opex.rename(columns={0: 'cost_opex'}, inplace=True)
    if 'level_0' in df_opex.columns:
        df_opex.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_opex_sum = df_opex[['scenario', 'technology','cost_opex']].groupby(['scenario', 'technology']).sum()
    else:
        df_opex_sum = df_opex[['technology', 'cost_opex']].groupby(['technology']).sum()
    save_file = os.path.join(save_folder, 'opex.csv')
    df_opex_sum.to_csv(save_file, index=True)

    # Capex yearly
    df_opex_yearly = res_basic.get_full_ts("opex_yearly")
    df_opex_yearly.reset_index(inplace=True)
    df_opex_yearly.rename(columns={0: 'opex_yearly'}, inplace=True)
    if 'level_0' in df_opex_yearly.columns:
        df_opex_yearly.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_opex_yearly_sum = df_opex_yearly[['scenario', 'technology', 'opex_yearly']].groupby(['scenario', 'technology']).sum()
    else:
        df_opex_yearly_sum = df_opex_yearly[['technology', 'opex_yearly']].groupby(['technology']).sum()
    save_file = os.path.join(save_folder, 'opex_yearly.csv')
    df_opex_yearly_sum.to_csv(save_file, index=True)

    # Carbon emission carrier
    df_carbon_emissions_carrier = res_basic.get_full_ts('carbon_emissions_carrier')
    df_carbon_emissions_carrier_sum = df_carbon_emissions_carrier.sum(axis=1).reset_index()
    df_carbon_emissions_carrier_sum.rename(columns={0: 'carbon_emissions_carrier'}, inplace=True)
    if 'level_0' in df_carbon_emissions_carrier_sum.columns:
        df_carbon_emissions_carrier_sum.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_carbon_emissions_carrier_sum_sum = df_carbon_emissions_carrier_sum[['scenario', 'carrier', 'carbon_emissions_carrier']].groupby(['scenario', 'carrier']).sum().reset_index()
    else:
        df_carbon_emissions_carrier_sum_sum = df_carbon_emissions_carrier_sum[['carrier', 'carbon_emissions_carrier']].groupby(['carrier']).sum().reset_index()

    print('Carbon emissions per carrier')
    print(df_carbon_emissions_carrier_sum_sum)

    df_carbon_emissions_carrier_sum_sum.to_csv(os.path.join(save_folder, 'co2_carrier.csv'), index=False)
    #df_carbon_emissions_carrier_sum.to_csv(os.path.join(save_folder, 'co2_carrier_long.csv'), index=False)


    # Carbon emissions technology
    df_co2_technology = res_basic.get_full_ts('carbon_emissions_technology')
    df_co2_technology_sum = df_co2_technology.sum(axis=1).reset_index()
    df_co2_technology_sum.rename(columns={0: 'carbon_emissions_technology'}, inplace=True)
    if 'level_0' in df_co2_technology_sum.columns:
        df_co2_technology_sum.rename(columns={'level_0': 'scenario'}, inplace=True)
        print(df_co2_technology_sum.columns)
        df_co2_technology_sum_sum = df_co2_technology_sum[['scenario', 'technology', 'carbon_emissions_technology']].groupby(['scenario', 'technology']).sum().reset_index()
    else:
        print(df_co2_technology_sum.columns)
        df_co2_technology_sum_sum = df_co2_technology_sum[['technology', 'carbon_emissions_technology']].groupby(['technology']).sum().reset_index()

    print('Carbon emissions per carrier')
    print(df_co2_technology_sum_sum)

    df_co2_technology_sum_sum.to_csv(os.path.join(save_folder, 'co2_technology.csv'), index=False)
    #df_co2_technology_sum.to_csv(os.path.join(save_folder, 'co2_technology_long.csv'), index=False)



    # Get and summarize cost carrier data
    df_capacity = res_basic.get_full_ts('capacity_addition')
    df_capacity_sum = df_capacity.reset_index()
    df_capacity_sum.rename(columns={0: 'capacity_addition'}, inplace=True)
    if 'level_0' in df_capacity_sum.columns:
        df_capacity_sum.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_capacity_sum_sum = df_capacity_sum[['scenario', 'technology', 'capacity_type', 'capacity_addition']].groupby(['scenario', 'technology','capacity_type']).sum().reset_index()
    else:
        df_capacity_sum_sum = df_capacity_sum[['technology', 'capacity_type', 'capacity_addition']].groupby(['technology', 'capacity_type']).sum().reset_index()

    print('Capacity addition')
    print(df_capacity_sum_sum)

    df_capacity_sum_sum.to_csv(os.path.join(save_folder, 'capacity.csv'), index=False)
    #df_capacity_sum.to_csv(os.path.join(save_folder, 'capacity_long.csv'), index=False)


    # Charging
    df_charge = res_basic.get_full_ts('flow_storage_charge')
    df_charge_sum = df_charge.sum(axis=1).reset_index()
    df_charge_sum.rename(columns={0: 'flow_storage_charge'}, inplace=True)
    if 'level_0' in df_charge_sum.columns:
        df_charge_sum.rename(columns={'level_0': 'scenario'}, inplace=True)
        df_charge_sum_sum = df_charge_sum[['scenario', 'technology', 'flow_storage_charge']].groupby(['scenario', 'technology']).sum().reset_index()
    else:
        df_charge_sum_sum = df_charge_sum[['technology', 'flow_storage_charge']].groupby(['technology']).sum().reset_index()

    print('Charge')
    print(df_charge_sum_sum)
    df_charge_sum_sum.to_csv(os.path.join(save_folder, 'flow_storage_charge.csv'), index=False)

    # Discharging
    df_discharge = res_basic.get_full_ts('flow_storage_discharge')
    df_discharge_sum = df_discharge.sum(axis=1).reset_index()
    df_discharge_sum.rename(columns={0: 'flow_storage_discharge'}, inplace=True)
    if 'level_0' in df_discharge_sum.columns:
        df_discharge_sum.rename(columns={'level_0': 'scenario'}, inplace=True)
        print(df_discharge_sum.columns)
        df_discharge_sum_sum = df_discharge_sum[['scenario', 'technology', 'flow_storage_discharge']].groupby(['scenario', 'technology']).sum().reset_index()
    else:
        print(df_discharge_sum.columns)
        df_discharge_sum_sum = df_discharge_sum[['technology', 'flow_storage_discharge']].groupby(['technology']).sum().reset_index()

    print('Discharge')
    print(df_discharge_sum_sum)
    df_discharge_sum_sum.to_csv(os.path.join(save_folder, 'flow_storage_discharge.csv'), index=False)


# #################################################################
def plot_energy_balance(res_basic, node, scenarios, directory, short=False, save_fig=True):
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
        results_JS.plot_energy_balances_carriers(res_basic, node, carriers, directory, scenario=scenario, short=short, save_fig=save_fig)

def print_import_cost_carrier(res_basic, folder, output_path):
    """
    Process cost carrier and flow import data, summarize them, and save the results to CSV files.

    Parameters:
    res_basic: Object containing the methods to get time series data.
    output_path: Path where the CSV files will be saved.
    """
    df_cost_carrier_tot_dict = res_basic.get_df("cost_carrier_total")
    df_cost_carrier_tot = pd.DataFrame(df_cost_carrier_tot_dict)
    print(df_cost_carrier_tot)



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
    df_import_sum.to_csv(os.path.join(save_folder, 'import_long.csv'), index=False)
    df_cost_carrier_sum.to_csv(os.path.join(save_folder, 'carrier_long.csv'), index=False)

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
    #df_demand_sum_filtered.to_csv(os.path.join(save_folder, 'demand_long.csv'), index=False)




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




def main():
    #folder = 'county_1907/county_1707_1_base'
    folder = 'county_1707_2_base'
    print(folder)
    output_path = "../../../outputs/"
    directory = os.path.join(output_path, folder)
    res_basic = Results(directory)

    #get_co2_emissions(res_basic, folder, output_path)
    #print_import_cost_carrier(res_basic, folder, output_path)

    node = 'MS_BE009'
    scenarios = ['scenario_','scenario_0']

    #plot_energy_balance(res_basic, node, scenarios, directory, short=True, save_fig=True)

    ################################################################################
    list_folders = ['county_1707_1_sce','county_1707_2_sce','county_1707_3_sce','county_1707_4_sce','county_1707_5_sce','county_1707_6_sce','county_1707_7_sce','county_1707_8_sce']
    list_folders_BAU = ['county_1707_1_base','county_1707_2_base','county_1707_3_base','county_1707_4_base','county_1707_5_base','county_1707_6_base','county_1707_7_base','county_1707_8_base']
    #list_folders_BAU = ['county_1907/county_1707_1_base','county_1907/county_1707_2_base','county_1907/county_1707_6_base','county_1907/county_1707_8_base']
    #list_folders = ['county_1107/county_CA_0507_288_4','county_1107/county_CA_0507_288_5']
    area = 'United States'
    custom_order = ['','0']
    pareto_group = 'scenario_name'
    specific_scenario_name = 'analysis_110724'
    # Define unit conversions
    units = {
        'co2': 'tons',
        'cost': 'USD',
        'water_energy': '1000*meter ** 3',
        'energy': 'MWh',
        'power': 'MW'
    }
    filter_component = 'costs'

    ##### Plot figure 1
    point_BAU = results_JS.create_co2_cost_point(output_path, list_folders_BAU)

    dfs = results_JS.prepare_data_for_stacked_cost_plot(folder, list_folders, output_path, specific_scenario_name, filter_component)
    plot_results.plot_stacked_costs(dfs, output_path, folder, units, point_BAU, save_fig=True)
    dfs_BAU = results_JS.prepare_data_for_stacked_cost_plot(folder, list_folders_BAU, output_path, specific_scenario_name, filter_component)
    plot_results.plot_stacked_procentage_BAU(dfs_BAU, output_path, folder, units, save_fig=True)
    plot_results.plot_percentage_stacked_costs(dfs, output_path, folder, units, save_fig=True)

    #### Plot figure 2 (Capacities + Stacked flow import)
    result_capacities_dfs = results_JS.prepare_data_for_capacity_figure(folder, output_path, list_folders, specific_scenario_name)
    plot_results.plot_pareto_capacities(result_capacities_dfs, output_path, folder, units, save_fig=True)

#    result_flow_import_dfs = results_JS.prepare_data_for_flow_import_stacked(folder, list_folders, output_path, specific_scenario_name)
#    plot_results.plot_stacked_import(result_flow_import_dfs, output_path, folder, units, save_fig=True)
#
#    #Flow import BAU:
#    result_flow_import_dfs_BAU = results_JS.prepare_data_for_flow_import_stacked(folder, list_folders_BAU, output_path, specific_scenario_name)
#    plot_results.plot_stacked_import_BAU(folder, output_path, result_flow_import_dfs_BAU, units, save_fig=True)
#
#    ### Plot figure 3 (Map Capacities and Flow import)
#    results_JS.prepare_data_for_map_plot(folder, list_folders, output_path, scenarios=None)
#    results_JS.prepare_data_for_map_plot(folder, list_folders_BAU, output_path, BAU=True, scenarios=None)
    # #    #################################################################################

    #    #scenarios = ['scenario_0','scenario_25','scenario_75','scenario_100']

    # #    results_JS.plot_pareto_front(folder, output_path, units, specific_scenario_name, custom_order, area, pareto_group, list_folders=list_folders, save_fig=True
    #    df_caps_all_filtered = results_JS.aggregate_to_states(folder, output_path, 'capacity_addition', scenarios, list_folders)
    #    plot_results.plot_boxplot_capacities_states(scenarios, df_caps_all_filtered, output_path, folder)
    #    df_group, df_group_filtered = results_JS.import_flow_data(folder, output_path, scenarios, column_name='flow_import', filter_carriers= ['electricity', 'diesel'], list_folders= list_folders)
    #    plot_results.plot_boxplot_energy_states(scenarios, df_group_filtered, folder, output_path, target_column='flow_import')
    #    df_group_demand, df_group_filtered_demand = results_JS.import_flow_data(folder, output_path, scenarios, 'demand', filter_carriers=['irrigation_water'], list_folders= list_folders)
    #    plot_results.plot_boxplot_energy_states(scenarios, df_group_filtered_demand, folder, output_path, 'demand')

if __name__ == "__main__":
    main()
