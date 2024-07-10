import pandas as pd
import os
from zen_garden.postprocess.results.results_JS2 import Results
from zen_garden.postprocess.results import results_JS5 as results_JS

filename = 'county_0907/county_CA_0507_288_5'

<<<<<<< HEAD
directory = os.path.join("../../../data/outputs", filename)
=======
directory = os.path.join("../outputs", filename)
>>>>>>> 14e32f15 (Add postprocess results and unit handling scripts)
res_basic = Results(directory)

df_co2_dict = res_basic.get_df('carbon_emissions_cumulative')
print(df_co2_dict)

carriers = ['water','electricity']
node = 'OR_UM059'
max_hours = 8740
duration = 24

scenarios = [scenario for scenario in os.listdir(directory) if scenario.startswith('scenario_0')]
scenarios = ['scenario_low_CO2_grid','scenario_']
for scenario in scenarios:
    results_JS.plot_energy_balances_carriers(res_basic, node, carriers, directory, scenario=scenario, save_fig=False)
