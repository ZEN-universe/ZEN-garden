import pandas as pd
import os
from zen_garden.postprocess.results.results_JS2 import Results
from zen_garden.postprocess.results import results_JS5 as results_JS
from zen_garden.postprocess.results import plot_results

filename = 'county_CA_0507_288_3'
output_path = "../../../outputs/"

directory = os.path.join(output_path, filename)
print(directory)
res_basic = Results(directory)

df_co2_dict = res_basic.get_df('carbon_emissions_cumulative')
print(df_co2_dict)


# #################################################################
carriers = ['water','electricity']
node = 'LA_EV039'
max_hours = 8740
duration = 24

scenarios = [scenario for scenario in os.listdir(directory) if scenario.startswith('scenario_0')]
scenarios = ['scenario_0','scenario_25','scenario_75','scenario_100']
for scenario in scenarios:
    results_JS.plot_energy_balances_carriers(res_basic, node, carriers, directory, scenario=scenario, save_fig=True)

####################################################################

# Assuming parent_folder and df_tech_cap are defined
folder = "county_CA_0507_288_3"


unit_co2 = '0.000001*tons'
unit_cost = '1000000*USD'
unit_water_energy = '1000*meter ** 3'
unit_energy = 'MWh'
unit_power = 'MW'
area = 'Nodes 3'
custom_order = ['','100','75','25','0']
pareto_group = 'time_steps'
specific_scenario_name = 'analysis_1806'
save_fig = True
#
#
## Define parameters for the first set of plots
#cost_components = [
#    ('cost_co2', f'Net Present Cost vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'net_present_cost', 'Net Present Cost'),
#    ('costs', f'CAPEX vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'cost_capex_total', 'Capex'),
#    ('costs', f'OPEX vs. CO2 Emissions', 'cost_opex_total', 'Opex'),
#    ('costs', f'Cost Carrier vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'cost_carrier_total', 'Cost Carrier'),
#    ('costs', f'Cost Carbon Emissions vs. $\\mathrm{{CO_2}}$ Emissions in {area}', 'cost_carbon_emissions_total', 'Cost Carbon Emissions')
#]
#
## Generate the first set of plots
#for filter_component, title, y_axis, y_axis_label in cost_components:
#    df = results_JS.filter_boxplot_no_parent_folder(folder, specific_scenario_name, filter_component=filter_component)
#    print(df.head())
#    pareto_df = results_JS.filter_pareto_group(df, custom_order, pareto_group)
#    plot_results.plot_pareto_front(pareto_df, folder, output_path, title=title, unit_co2=unit_co2, unit_y_axis=unit_cost, y_axis=y_axis, y_axis_label=y_axis_label, save_fig=save_fig)
#
## Define parameters for the capacities plots
#capacity_components = [
#    ('water_storage, energy', f'Water Storage Capacity vs. $\\mathrm{{CO_2}}$ Emissions in {area}', unit_water_energy, 'Installed Capacity'),
#    ('battery, energy', f'Battery Capacity vs. $\\mathrm{{CO_2}}$ Emissions in {area}', unit_energy, 'Installed Capacity'),
#    ('PV, power', f'PV Capacity vs. $\\mathrm{{CO_2}}$ Emissions in {area}', unit_power, 'Installed Capacity')
#]
#
## Read the capacities boxplot data
##print current directory
##print(os.getcwd())
#df_tech_cap = pd.read_csv("capacities_boxplot_JS.csv")
#
## Generate the capacities plots
#capacities_df = results_JS.filter_boxplot_no_parent_folder(folder, filter_component='capacities', df_tech_cap=df_tech_cap, specific_scenario_name='analysis_1806')
#capacities_pareto_df = results_JS.filter_pareto_group(capacities_df, custom_order, pareto_group)
#for y_axis, title, unit_y_axis, y_axis_label in capacity_components:
#    plot_results.plot_pareto_front(capacities_pareto_df, folder, output_path, title=title, unit_co2=unit_co2, unit_y_axis=unit_y_axis, y_axis=y_axis, y_axis_label=y_axis_label, save_fig=save_fig)
#
#flow_import_df = results_JS.filter_boxplot_no_parent_folder(folder, filter_component='flow_import', specific_scenario_name='analysis_1806')
#flow_import_pareto_df = results_JS.filter_pareto_group(flow_import_df, custom_order, pareto_group)
#flow_import_components = [
#    ('electricity', f'Imported Electricity vs. $\\mathrm{{CO_2}}$ Emissions in {area}', unit_energy, 'Import'),
#    ('diesel', f'Imported Diesel vs. $\\mathrm{{CO_2}}$ Emissions in {area}', unit_energy, 'Import'),
#]
#
#for y_axis, title, unit_y_axis, y_axis_label in flow_import_components:
#    plot_results.plot_pareto_front(flow_import_pareto_df, folder, output_path, title=title, unit_co2=unit_co2, unit_y_axis=unit_y_axis, y_axis=y_axis, y_axis_label=y_axis_label, save_fig=save_fig)
#for y_axis, title, unit_y_axis, y_axis_label in flow_import_components:
#    plot_results.plot_pareto_front_cost(flow_import_pareto_df, folder, output_path, title=title, unit_cost=unit_cost, unit_y_axis=unit_y_axis, y_axis=y_axis, y_axis_label=y_axis_label, save_fig=True)
#for y_axis, title, unit_y_axis, y_axis_label in capacity_components:
#    plot_results.plot_pareto_front_cost(capacities_pareto_df, folder, output_path, title=title, unit_cost=unit_cost, unit_y_axis=unit_y_axis, y_axis=y_axis, y_axis_label=y_axis_label, save_fig=True)