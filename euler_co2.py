import pandas as pd
import os
from zen_garden.postprocess.results import Results

filename = 'county_CA_0507_288_3'

directory = os.path.join("outputs", filename)
res_basic = Results(directory)

df_co2_dict = res_basic.get_df('carbon_emissions_cumulative')
print(df_co2_dict)