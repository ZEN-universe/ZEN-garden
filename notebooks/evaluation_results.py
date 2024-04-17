<<<<<<< Updated upstream
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')
from pprint import pprint
from zen_garden.postprocess.results import Results
import seaborn as sns
import os
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker
import plotly.io as pio

pio.orca.config.executable = 'C:\\Users\\PaulaBaumann\\.conda\\envs\\zen-garden-linopy\\orca.cmd'


res_scenario = Results("../outputs/hard_to_abate_ammonia_scenarios_010224/")

df_emissions = res_scenario.get_df("carbon_emissions_cumulative")
print(df_emissions)
def get_emissions(folder_path, scenario):
    scenario_name_mapping = {
        'scenario_': 'Baseline Scenario',
        'scenario_high_demand': 'High Demand Scenario',
        'scenario_low_demand': 'Low Demand Scenario'
    }
    scenario_name = scenario_name_mapping.get(scenario, scenario)

    df_emissions = res_scenario.get_df("carbon_emissions_cumulative")
    print(df_emissions)

    os.makedirs(folder_path, exist_ok=True)

    carbon_emissions_carrier = res_scenario.get_df('carbon_emissions_carrier', scenario=scenario)
    file_path = os.path.join(folder_path, f"carbon_emissions_carrier_{scenario}.csv")
    carbon_emissions_carrier.to_csv(file_path)

    file_path = os.path.join(folder_path, f"carbon_emissions_carrier_{scenario}.csv")
    carbon_emissions = pd.read_csv(file_path)
    carbon_emissions = carbon_emissions.drop(['node', 'time_operation'], axis=1)
    carbon_emissions_grouped = carbon_emissions.groupby(['carrier']).sum()
    file_path = os.path.join(folder_path, f"carbon_emissions_grouped_{scenario_name}.csv")
    carbon_emissions_grouped.to_csv(file_path)

def save_total(folder_path, scenario):

    os.makedirs(folder_path, exist_ok=True)

    flow_conversion_input = res_scenario.get_total("flow_conversion_input", scenario=scenario)
    file_path = os.path.join(folder_path, f"flow_conversion_input_{scenario}.csv")
    flow_conversion_input.to_csv(file_path)

    flow_conversion_output = res_scenario.get_total("flow_conversion_output", scenario=scenario)
    file_path = os.path.join(folder_path, f"flow_conversion_output_{scenario}.csv")
    flow_conversion_output.to_csv(file_path)

    file_path = os.path.join(folder_path, f"flow_conversion_input_{scenario}.csv")
    total_input = pd.read_csv(file_path)
    total_input = total_input.drop('node', axis=1)
    total_input = total_input.groupby(['technology', 'carrier']).sum()
    file_path = os.path.join(folder_path, f"flow_conversion_input_grouped_{scenario}.csv")
    total_input.to_csv(file_path)

    file_path = os.path.join(folder_path, f"flow_conversion_output_{scenario}.csv")
    total_output = pd.read_csv(file_path)
    total_output = total_output.drop('node', axis=1)
    total_output = total_output.groupby(['technology', 'carrier']).sum()
    file_path = os.path.join(folder_path, f"flow_conversion_output_grouped_{scenario}.csv")
    total_output.to_csv(file_path)

def save_capacity(folder_path, scenario):

    os.makedirs(folder_path, exist_ok=True)
    capacity = res_scenario.get_total('capacity', scenario=scenario)
    file_path = os.path.join(folder_path, f"capacity_{scenario}.csv")
    capacity.to_csv(file_path)

    file_path = os.path.join(folder_path, f"capacity_{scenario}.csv")
    capacity = pd.read_csv(file_path)
    capacity = capacity.drop(["location", "capacity_type"], axis=1)
    capacity_grouped = capacity.groupby(['technology']).sum()
    file_path = os.path.join(folder_path, f"capacity_grouped_{scenario}.csv")
    capacity_grouped.to_csv(file_path)

def save_imports_exports(folder_path, scenario):

    os.makedirs(folder_path, exist_ok=True)
    imports = res_scenario.get_total("flow_import", scenario=scenario)
    file_path = os.path.join(folder_path, f"imports_{scenario}.csv")
    imports.to_csv(file_path)

    file_path = os.path.join(folder_path, f"imports_{scenario}.csv")
    imports = pd.read_csv(file_path)

    imports = imports.drop(["node"], axis=1)
    imports_grouped = imports.groupby(['carrier']).sum()
    file_path = os.path.join(folder_path, f"imports_grouped_{scenario}.csv")
    imports_grouped.to_csv(file_path)

    exports = res_scenario.get_total("flow_export", scenario=scenario)
    file_path = os.path.join(folder_path, f"exports_{scenario}.csv")
    exports.to_csv(file_path)

    file_path = os.path.join(folder_path, f"exports_{scenario}.csv")
    exports = pd.read_csv(file_path)

    exports = exports.drop(["node"], axis=1)
    exports_grouped = exports.groupby(['carrier']).sum()
    file_path = os.path.join(folder_path, f"exports_grouped_{scenario}.csv")
    exports_grouped.to_csv(file_path)

def generate_sankey_diagram(folder_path, scenario, target_technologies, intermediate_technologies, year, title, save_file):

    scenario_name_mapping = {
        'scenario_': 'Baseline Scenario',
        'scenario_high_demand': 'High Demand Scenario',
        'scenario_low_demand': 'Low Demand Scenario'
    }
    scenario_name = scenario_name_mapping.get(scenario, scenario)

    file_path_input = os.path.join(folder_path, f"flow_conversion_input_grouped_{scenario}.csv")
    inputs_df = pd.read_csv(file_path_input)
    file_path_output = os.path.join(folder_path, f"flow_conversion_output_grouped_{scenario}.csv")
    outputs_df = pd.read_csv(file_path_output)

    input_techs_target = inputs_df[inputs_df['technology'].isin(target_technologies)]

    input_techs_intermediate = inputs_df[inputs_df['technology'].isin(intermediate_technologies)]

    output_techs_intermediate = outputs_df[outputs_df['technology'].isin(intermediate_technologies)]

    output_techs_target = outputs_df[outputs_df['technology'].isin(target_technologies)]

    input_sankey_target = pd.DataFrame({
        'source': input_techs_target['carrier'],
        'target': input_techs_target['technology'],
        'value': input_techs_target[year]
    })

    input_sankey_intermediate = pd.DataFrame({
        'source': input_techs_intermediate['carrier'],
        'target': input_techs_intermediate['technology'],
        'value': input_techs_intermediate[year]
    })

    output_sankey_intermediate = pd.DataFrame({
        'source': output_techs_intermediate['technology'],
        'target': output_techs_intermediate['carrier'],
        'value': output_techs_intermediate[year]
    })

    output_sankey_target = pd.DataFrame({
        'source': output_techs_target['technology'],
        'target': output_techs_target['carrier'],
        'value': output_techs_target[year]
    })

    links = pd.concat([input_sankey_target, output_sankey_intermediate, input_sankey_intermediate, output_sankey_target], axis=0)

    unique_source_target = list(pd.unique(links[['source', 'target']].values.ravel('K')))

    mapping_dict = {k:v for v, k in enumerate(unique_source_target)}

    links['source'] = links['source'].map(mapping_dict)
    links['target'] = links['target'].map(mapping_dict)

    links_dict = links.to_dict(orient="list")

    color_palette = px.colors.qualitative.Dark24

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "sankey"}]])

    fig.add_trace(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=unique_source_target,
            color=color_palette
        ),
        link=dict(
            source=links_dict['source'],
            target=links_dict['target'],
            value=links_dict['value'],
            label=[f"{source} to {target}" for source, target in zip(links_dict['source'], links_dict['target'])],
            hovertemplate='%{value}'
        )
    ))

    if isinstance(year, str):
        year = int(year)
    displayed_year = year * 2 + 2024
    fig.update_layout(title_text=f"{title} {displayed_year} ({scenario_name})", font_size=18)

    fig.update_layout(font=dict(family="Times New Roman"))

    '''def generate_plot_title(target_technologies, displayed_year, scenario_name):
        ammonia_technologies = ['haber_bosch', 'e_haber_bosch']
        steel_technologies = ['BF_BOF', 'BF_BOF_CCS', 'EAF']
        cement_technologies = ['cement_plant', 'cement_plant_oxy_combustion', 'cement_plant_pcc_coal', 'cement_plant_pcc_ng']
        methanol_technologies = ['gasification_methanol', 'methanol_synthesis', 'methanol_from_hydrogen']
        refining_technologies = ['refinery']
        all_technologies = ['BF_BOF','BF_BOF_CCS', 'EAF', 'carbon_liquefication', 'carbon_removal', 'carbon_storage',
                           'cement_plant', 'cement_plant_oxy_combustion', 'cement_plant_pcc_coal', 'cement_plant_pcc_ng',
                           'e_haber_bosch', 'haber_bosch', 'gasification_methanol', 'methanol_from_hydrogen', 'methanol_synthesis',
                           'refinery']

        if all(tech in target_technologies for tech in ammonia_technologies):
            title_suffix = 'ammonia_production'
        elif all(tech in target_technologies for tech in steel_technologies):
            title_suffix = 'steel_production'
        elif all(tech in target_technologies for tech in cement_technologies):
            title_suffix = 'cement_production'
        elif all(tech in target_technologies for tech in methanol_technologies):
            title_suffix = 'methanol_production'
        elif all(tech in target_technologies for tech in refining_technologies):
            title_suffix = 'refining_production'
        elif set(target_technologies) == set(all_technologies):
            title_suffix = 'All Industries'
        else:
            title_suffix = 'Custom Title'

        return f"Title Prefix {displayed_year} ({scenario_name}) - {title_suffix}"'''

    if save_file:
        subfolder_name = "sankey"
        subfolder_path = os.path.join(folder_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        png_file_path = os.path.join(subfolder_path, f"{target_technologies}_{displayed_year}_{scenario}.png")

        fig.write_image(png_file_path, format="png", width=1600, height=1200)

    fig.show()



def plot_outputs(folder_path, scenario, carrier, save_file):

    scenario_name_mapping = {
        'scenario_': 'Baseline Scenario',
        'scenario_high_demand': 'High Demand Scenario',
        'scenario_low_demand': 'Low Demand Scenario'
    }
    scenario_name = scenario_name_mapping.get(scenario, scenario)

    plt.rcParams["font.family"] = "Times New Roman"

    file_name = os.path.join(folder_path, f"flow_conversion_output_grouped_{scenario}.csv")
    df_output = pd.read_csv(file_name)

    grouped_df = df_output[df_output['carrier'] == carrier]
    year_mapping = {str(i): str(2024 + 2*i) for i in range(14)}

    grouped_df.rename(columns=year_mapping, inplace=True)

    grouped_df.set_index('technology', inplace=True)
    grouped_df = grouped_df.dropna()

    for col in grouped_df.columns:
        if grouped_df[col].dtype == 'O':
            print(f"Unique values in {col}: {grouped_df[col].unique()}")

    grouped_df_values = grouped_df.drop(['carrier'], axis=1).transpose()

    palette = sns.color_palette("dark", n_colors=len(grouped_df_values.columns))

    grouped_df_values = grouped_df_values.astype(float)

    ax = grouped_df_values.plot(kind='bar', stacked=True, figsize=(10, 6), color=palette)

    ax.legend(title='Technology', bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    plt.subplots_adjust(right=0.8)

    ax.set_xlabel("Year", fontname="Times New Roman", fontsize=12)
    #ax.set_ylabel("Output in Mt", fontname="Times New Roman", fontsize=12)

    y_labels = [f"{int(label) / 1000:.2f}" for label in ax.get_yticks()]
    ax.set_yticklabels(y_labels)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(f"{carrier.capitalize()} Output by Year and Technology ({scenario_name})")
    plt.xlabel("Year")

    if carrier in ['hydrogen', 'electricity', 'natural_gas']:
        ax.set_ylabel("Yearly Production [1,000 GWh]", fontname="Times New Roman", fontsize=12)
    else:
        ax.set_ylabel("Yearly Production [Mt]", fontname="Times New Roman", fontsize=12)

    if save_file:
        subfolder_name = "output_bar_charts"
        subfolder_path = os.path.join(folder_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        png_file_path = os.path.join(subfolder_path, f"{carrier}_bar_chart_{scenario}.png")
        plt.savefig(png_file_path, bbox_inches='tight')

    plt.show()


def plot_capacities(folder_path, scenario, technology, save_file):

    scenario_name_mapping = {
        'scenario_': 'Baseline Scenario',
        'scenario_high_demand': 'High Demand Scenario',
        'scenario_low_demand': 'Low Demand Scenario'
    }
    scenario_name = scenario_name_mapping.get(scenario, scenario)

    plt.rcParams["font.family"] = "Times New Roman"

    file_name = os.path.join(folder_path, f"flow_conversion_output_grouped_{scenario}.csv")
    df_output = pd.read_csv(file_name)

    grouped_df = df_output[df_output['technology'] == technology]
    year_mapping = {str(i): str(2024 + 2 * i) for i in range(14)}

    grouped_df.rename(columns=year_mapping, inplace=True)
    grouped_df = grouped_df.dropna()

    palette = sns.color_palette("dark", n_colors=len(grouped_df.columns))

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_df.plot(kind='bar', stacked=False, ax=ax, color=palette)

    ax.legend(title='Technology', bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    plt.subplots_adjust(right=0.8)

    ax.set_xlabel("Year", fontname="Times New Roman", fontsize=12)
    ax.set_ylabel("Capacity in Mt", fontname="Times New Roman", fontsize=12)

    # Convert y-axis labels to represent values in Megatons
    y_labels = [f"{int(label) / 1000:.2f}" for label in ax.get_yticks()]
    ax.set_yticklabels(y_labels)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(f"{technology.capitalize()} Capacity by Year and Technology ({scenario_name})")
    plt.xlabel("Year")
    plt.ylabel(f"Yearly {technology.capitalize()} Capacity [Mt]")

    if save_file:
        subfolder_path = os.path.join(folder_path, "capacities")
        os.makedirs(subfolder_path, exist_ok=True)
        png_file_path = os.path.join(subfolder_path, f"{technology}_bar_chart_{scenario}.png")
        plt.savefig(png_file_path, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':

    # save (grouped) inputs and outputs as csv
    folder_path = 'ammonia_scenarios_100224'
    scenarios = ['scenario_', 'scenario_high_demand', 'scenario_low_demand'
                ]
    carriers = ['ammonia',
                #'cement',
                #'steel',
                #'methanol',
                #'oil_products',
                #'direct_reduced_iron',
                #'hydrogen',
                #'electricity',
                #'natural_gas',
                #'carbon'
                ]

    # get emissions
    for scenario in scenarios:
        for carrier in carriers:
            get_emissions(folder_path, scenario)

    for scenario in scenarios:
        save_total(folder_path, scenario)

    # save (grouped) capacity as csv
    for scenario in scenarios:
        save_capacity(folder_path, scenario)

    # save imports and exports

    for scenario in scenarios:
        save_imports_exports(folder_path, scenario)

    # generate sankey diagram
    target_technologies = [#'BF_BOF','BF_BOF_CCS', 'EAF', 'BF_BOF_retrofit', 'BF_BOF_CCS_retrofit'
                           #'carbon_liquefication', 'carbon_removal', 'carbon_storage',
                           #'cement_plant', 'cement_plant_oxy_combustion', 'cement_plant_pcc_coal', 'cement_plant_pcc_ng', 'cement_plant_post_comb',
                           'e_haber_bosch', 'haber_bosch',
                           #'gasification_methanol', 'methanol_from_hydrogen', 'methanol_synthesis',
                           #'refinery'
                            ]
    intermediate_technologies = ['anaerobic_digestion', 'biomethane_conversion',
                                 'ASU',
                                 'biomass_to_coal_conversion', 'hydrogen_for_cement_conversion',
                                 'DAC',
                                 'DRI', 'h2_to_ng', 'scrap_conversion_EAF', 'scrap_conversion_BF_BOF',
                                 'SMR', 'SMR_CCS', 'gasification', 'gasification_CCS',
                                 'electrolysis',
                                 'photovoltaics', 'pv_ground', 'pv_rooftop', 'wind_offshore', 'wind_onshore',
                                 'carbon_liquefication', 'carbon_removal', 'carbon_storage'
                                ]
    years = ['0', '3', '13']
    #for year in years:
        #for scenario in scenarios:
           # generate_sankey_diagram(folder_path, scenario, target_technologies, intermediate_technologies, year, title="Process depiction in", save_file=True)

    # generate bar charts for industry outputs

    for scenario in scenarios:
        for carrier in carriers:
            plot_outputs(folder_path, scenario, carrier, save_file=True)

    # plot capacities for chosen technologies
    technologies = ['haber_bosch', 'e_haber_bosch']
    #for technology in technologies:
        #for scenario in scenarios:
            #plot_capacities(folder_path, scenario, technology, save_file=False)

=======
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from zen_garden.postprocess.results import Results
import seaborn as sns
import os
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.ticker as ticker
import plotly.io as pio
import folium
from folium import plugins
import geopandas as gpd
import json
from IPython.display import display
import webbrowser
import tempfile
import matplotlib.patches as mpatches
from matplotlib.patches import Wedge
from folium.plugins import MarkerCluster
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import networkx as nx
from shapely.geometry import Point
from functools import partial
from functools import partial
import pyproj
from shapely.ops import transform
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({'font.size': 22})

pio.orca.config.executable = 'C:\\Users\\PaulaBaumann\\.conda\\envs\\zen-garden-linopy\\orca.cmd'
#

res_scenario = Results("../outputs/hard_to_abate_scenarios_140324/")

'''df = res_scenario.get_total('flow_conversion_input', scenario='scenario_biomass_high_price_10').xs('carbon_storage')
df_new = df.xs('carbon_liquid').sum()
print(df_new)'''

df = pd.read_csv("coal_for_cement/price_import.csv")
print('mean')
print(df['price_import'].mean())

'''df2 = pd.read_csv('biomass_cement/price_import_yearly_variation.csv')
mean_price_import_2024 = df2[df2['year'] == 2050]['price_import'].mean()
print('neu')
print(mean_price_import_2024)'''
#print(res_scenario.get_total('availability_import', scenario='scenario_').xs('dry_biomass', level='carrier').sum())

#df_emissions = res_scenario.get_df("carbon_emissions_cumulative")

#print(df_emissions)

def get_emissions(folder_path, scenario):
    scenario_name_mapping = {
        'scenario_': 'Baseline Scenario',
        #'scenario_high_demand': 'High Demand Scenario',
        #'scenario_low_demand': 'Low Demand Scenario',
        'scenario_electrification': 'Direct Electrification Scenario',
        'scenario_CCS': 'CCS Scenario',
        'scenario_biomass': 'Biomass Scenario',
        #'scenario_hydrogen': 'Hydrogen Scenario'
    }
    scenario_name = scenario_name_mapping.get(scenario, scenario)

    df_emissions = res_scenario.get_df("carbon_emissions_cumulative")

    os.makedirs(folder_path, exist_ok=True)

    carbon_emissions_carrier = res_scenario.get_df('carbon_emissions_carrier', scenario=scenario).round(4)
    file_path_carrier = os.path.join(folder_path, f"carbon_emissions_carrier_{scenario}.csv")
    carbon_emissions_carrier.to_csv(file_path_carrier)

    file_path_carrier_scenario = os.path.join(folder_path, f"carbon_emissions_carrier_{scenario}.csv")
    carbon_emissions_carrier = pd.read_csv(file_path_carrier_scenario)
    carbon_emissions_carrier = carbon_emissions_carrier.drop(['node', 'carrier'], axis=1)
    carbon_emissions_carrier = carbon_emissions_carrier.round(4)
    carbon_emissions_carrier_grouped = carbon_emissions_carrier.groupby(['time_operation']).sum()
    file_path = os.path.join(folder_path, f"carbon_emissions_carrier_grouped_{scenario}.csv")
    carbon_emissions_carrier_grouped.to_csv(file_path)

    '''carbon_emissions_carrier_yearly = pd.read_csv(file_path_carrier_scenario)
    carbon_emissions_carrier_yearly = carbon_emissions_carrier_yearly.drop(['carrier', 'node'], axis=1)
    carbon_emissions_carrier_yearly_grouped = carbon_emissions_carrier_yearly.groupby(['time_operation']).sum()'''

    carbon_emissions_tech = res_scenario.get_df('carbon_emissions_technology', scenario=scenario)
    file_path_tech = os.path.join(folder_path, f"carbon_emissions_technology_{scenario}.csv")
    carbon_emissions_tech.to_csv(file_path_tech)

    file_path_tech_scenario = os.path.join(folder_path, f"carbon_emissions_technology_{scenario}.csv")
    carbon_emissions_tech = pd.read_csv(file_path_tech_scenario)
    carbon_emissions_tech = carbon_emissions_tech.drop(['location', 'technology'], axis=1)
    carbon_emissions_tech = carbon_emissions_tech.round(4)
    carbon_emissions_tech_grouped = carbon_emissions_tech.groupby(['time_operation']).sum()
    file_path = os.path.join(folder_path, f"carbon_emissions_tech_grouped_{scenario}.csv")
    carbon_emissions_tech_grouped.to_csv(file_path)

    carbon_emissions_cumulative = res_scenario.get_df('carbon_emissions_cumulative', scenario=scenario)
    file_path_cumulative = os.path.join(folder_path, f"carbon_emissions_cumulative_{scenario}.csv")
    carbon_emissions_cumulative.to_csv(file_path_cumulative)

    '''carbon_emissions_yearly_grouped = pd.merge(carbon_emissions_tech_yearly_grouped, carbon_emissions_carrier_yearly_grouped, on='time_operation', how='outer')
    carbon_emissions_yearly_grouped['carbon_emissions'] = carbon_emissions_yearly_grouped['carbon_emissions_technology'] + carbon_emissions_yearly_grouped['carbon_emissions_carrier']
    carbon_emissions_yearly_grouped = carbon_emissions_yearly_grouped[['carbon_emissions']]

    file_path_emissions_yearly = os.path.join(folder_path, f"carbon_emissions_yearly_grouped_{scenario}.csv")
    carbon_emissions_yearly_grouped.to_csv(file_path_emissions_yearly)'''

def save_total(folder_path, scenario):

    os.makedirs(folder_path, exist_ok=True)

    flow_conversion_input = res_scenario.get_total("flow_conversion_input", scenario=scenario)
    file_path = os.path.join(folder_path, f"flow_conversion_input_{scenario}.csv")
    flow_conversion_input.to_csv(file_path)

    flow_conversion_output = res_scenario.get_total("flow_conversion_output", scenario=scenario)
    file_path = os.path.join(folder_path, f"flow_conversion_output_{scenario}.csv")
    flow_conversion_output.to_csv(file_path)

    file_path = os.path.join(folder_path, f"flow_conversion_input_{scenario}.csv")
    total_input = pd.read_csv(file_path)
    total_input = total_input.drop('node', axis=1)
    total_input = total_input.round(4)
    total_input = total_input.groupby(['technology', 'carrier']).sum()
    file_path = os.path.join(folder_path, f"flow_conversion_input_grouped_{scenario}.csv")
    total_input.to_csv(file_path)

    file_path = os.path.join(folder_path, f"flow_conversion_output_{scenario}.csv")
    total_output = pd.read_csv(file_path)
    total_output_grouped = total_output.drop('node', axis=1)
    total_output_grouped = total_output_grouped.round(4)
    total_output_grouped = total_output_grouped.groupby(['technology', 'carrier']).sum()
    file_path = os.path.join(folder_path, f"flow_conversion_output_grouped_{scenario}.csv")
    total_output_grouped.to_csv(file_path)

    hydrogen_output = total_output.copy()
    hydrogen_output = hydrogen_output.query("carrier == 'hydrogen'")
    file_path = os.path.join(folder_path, f"flow_conversion_output_hydrogen_{scenario}.csv")
    hydrogen_output.to_csv(file_path, index=False)

def save_capacity(folder_path, scenario):

    os.makedirs(folder_path, exist_ok=True)
    capacity = res_scenario.get_total('capacity', scenario=scenario)
    file_path = os.path.join(folder_path, f"capacity_{scenario}.csv")
    capacity.to_csv(file_path)

    file_path = os.path.join(folder_path, f"capacity_{scenario}.csv")
    capacity = pd.read_csv(file_path)
    capacity = capacity.drop(["location", "capacity_type"], axis=1)
    capacity_grouped = capacity.groupby(['technology']).sum()
    file_path = os.path.join(folder_path, f"capacity_grouped_{scenario}.csv")
    capacity_grouped.to_csv(file_path)

def save_imports_exports(folder_path, scenario):

    os.makedirs(folder_path, exist_ok=True)
    imports = res_scenario.get_total("flow_import", scenario=scenario)
    file_path = os.path.join(folder_path, f"imports_{scenario}.csv")
    imports.to_csv(file_path)

    file_path = os.path.join(folder_path, f"imports_{scenario}.csv")
    imports = pd.read_csv(file_path)

    imports = imports.drop(["node"], axis=1)
    imports = imports.round(4)
    imports_grouped = imports.groupby(['carrier']).sum()
    file_path = os.path.join(folder_path, f"imports_grouped_{scenario}.csv")
    imports_grouped.to_csv(file_path)

    exports = res_scenario.get_total("flow_export", scenario=scenario)
    file_path = os.path.join(folder_path, f"exports_{scenario}.csv")
    exports.to_csv(file_path)

    file_path = os.path.join(folder_path, f"exports_{scenario}.csv")
    exports = pd.read_csv(file_path)

    exports = exports.drop(["node"], axis=1)
    exports = exports.round(4)
    exports_grouped = exports.groupby(['carrier']).sum()
    file_path = os.path.join(folder_path, f"exports_grouped_{scenario}.csv")
    exports_grouped.to_csv(file_path)

def energy_carrier(folder_path, scenario):
    scenario_name_mapping = {
        'scenario_': 'Baseline Scenario',
        #'scenario_high_demand': 'High Demand Scenario',
        #'scenario_low_demand': 'Low Demand Scenario',
        'scenario_electrification': 'Direct Electrification Scenario',
        #'scenario_hydrogen': 'Hydrogen Scenario',
        'scenario_CCS': 'CCS Scenario',
        'scenario_biomass': 'Biomass Scenario'
    }
    scenario_name = scenario_name_mapping.get(scenario, scenario)

    inputs = pd.read_csv(f"{folder_path}/flow_conversion_input_{scenario}.csv")
    outputs = pd.read_csv(f"{folder_path}/flow_conversion_output_{scenario}.csv")

    inputs['node'] = inputs['node'].str.slice(0, 2)
    inputs_grouped = inputs.groupby(['technology', 'carrier', 'node']).sum().reset_index()
    energy_carriers_inputs = ['coal', 'coal_for_cement', 'natural_gas', 'electricity', 'dry_biomass',
                        'biomethane', 'biomass_cement']
    energy_carriers_outputs = ['coal', 'coal_for_cement', 'natural_gas', 'electricity', 'dry_biomass',
                              'biomethane', 'biomass_cement']

    inputs_grouped = inputs_grouped[inputs_grouped['carrier'].isin(energy_carriers_inputs)]

    file_path = os.path.join(folder_path, f"energy_carrier_inputs_{scenario}.csv")
    inputs_grouped.to_csv(file_path, index=False)

    outputs['node'] = outputs['node'].str.slice(0, 2)
    outputs_grouped = outputs.groupby(['technology', 'carrier', 'node']).sum().reset_index()
    outputs_grouped = outputs_grouped[outputs_grouped['carrier'].isin(energy_carriers_outputs)]

    file_path = os.path.join(folder_path, f"energy_carrier_outputs_{scenario}.csv")
    outputs_grouped.to_csv(file_path, index=False)

    natural_gas = outputs_grouped[(outputs_grouped['technology'].isin(['biomethane_conversion', 'h2_to_ng']))]
    natural_gas = natural_gas.drop('technology', axis=1)
    summed_natural_gas = natural_gas.groupby(['node', 'carrier']).sum().reset_index()

    coal = outputs_grouped[(outputs_grouped['technology'].isin(['scrap_conversion_BF_BOF', 'BF_BOF_CCS']))]
    coal = coal.drop('technology', axis=1)
    summed_coal = coal.groupby(['node', 'carrier']).sum().reset_index()

    electricity = outputs_grouped[(outputs_grouped['technology'].isin(['SMR', 'SMR_CCS', 'gasification_methanol', 'pv_ground', 'wind_offshore', 'wind_onshore']))]
    electricity = electricity.drop('technology', axis=1)
    summed_electricity = electricity.groupby(['node', 'carrier']).sum().reset_index()

    coal_for_cement = outputs_grouped[(outputs_grouped['technology'].isin(['hydrogen_for_cement_conversion', 'biomass_to_coal_conversion']))]
    coal_for_cement = coal_for_cement.drop('technology', axis=1)
    summed_coal_for_cement = coal_for_cement.groupby(['node', 'carrier']).sum().reset_index()

    renewables = outputs_grouped[(outputs_grouped['technology'].isin(['pv_ground', 'wind_offshore', 'wind_onshore']))]
    renewables = renewables.drop('technology', axis=1)
    summed_renewables = renewables.groupby(['node', 'carrier']).sum().reset_index()
    summed_renewables.loc[summed_renewables['carrier'] == 'electricity', 'carrier'] = 'renewable_electricity'

    for i in range(1, 14):
        summed_renewables[f'{i}'] = summed_renewables[f'{i}'].round(4)

    inputs_grouped_new = inputs_grouped.drop('technology', axis=1)
    inputs_grouped_new = inputs_grouped_new.groupby(['node', 'carrier']).sum().reset_index()
    inputs_grouped_new.to_csv(f"{folder_path}/inputs_grouped_new_{scenario}.csv", index=False)

    dfs = {
        'inputs_grouped_new': inputs_grouped_new,
        'coal': summed_coal,
        'natural_gas': summed_natural_gas,
        'electricity': summed_electricity,
        'coal_for_cement': summed_coal_for_cement
    }

    for df_name, df in dfs.items():
        if df_name != 'inputs_grouped_new':
            filtered_df = df[(df['carrier'].isin(inputs_grouped_new['carrier'])) & (df['node'].isin(inputs_grouped_new['node']))]

            for index, row in filtered_df.iterrows():
                carrier = row['carrier']
                node = row['node']

                inputs_grouped_new.loc[(inputs_grouped_new['carrier'] == carrier) & (inputs_grouped_new['node'] == node), '0':'13'] -= row['0':'13']

    inputs_grouped_new[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']] = inputs_grouped_new[
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']].apply(pd.to_numeric,
                                                                                          errors='coerce')
    for i in range(1, 14):
        inputs_grouped_new[f'{i}'] = inputs_grouped_new[f'{i}'].round(4)

    energy_carrier_country = pd.concat([inputs_grouped_new, summed_renewables])

    energy_carrier = energy_carrier_country.groupby(['carrier']).sum().reset_index()
    energy_carrier = energy_carrier.drop(['node'], axis=1)
    energy_carrier.loc[energy_carrier['carrier'] == 'coal', '0':'13'] /= (27.35/3.6)

    energy_carrier_country.to_csv(f"{folder_path}/energy_carrier_country_{scenario}.csv", index=False)
    energy_carrier.to_csv(f"{folder_path}/energy_carrier_{scenario}.csv", index=False)

    energy_carrier_new = pd.read_csv(f"{folder_path}/energy_carrier_{scenario}.csv")
    df = energy_carrier_new

    df.set_index('carrier', inplace=True)

    file_name_carrier = os.path.join(folder_path, f"carbon_emissions_carrier_grouped_{scenario}.csv")
    df_emissions_carrier = pd.read_csv(file_name_carrier)

    file_name_tech = os.path.join(folder_path, f"carbon_emissions_tech_grouped_{scenario}.csv")
    df_emissions_tech = pd.read_csv(file_name_tech)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    df.T.plot(kind='bar', stacked=True, ax=ax1)

    ax2 = ax1.twinx()
    (df_emissions_carrier['carbon_emissions_carrier'] + df_emissions_tech['carbon_emissions_technology']).plot(ax=ax2,
                                                                                                               color='r',
                                                                                                           label='Yearly Emissions')

    years = range(2024, 2051, 2)
    plt.xticks(range(len(years)), years, rotation=0)
    plt.yticks(fontname="Times New Roman")

    ax1.set_xlabel("Year", fontname="Times New Roman", fontsize=12)
    ax1.set_ylabel("Energy and Feedstock Input per year [100 TWh]", fontname="Times New Roman", fontsize=12)
    ax2.set_ylabel("Yearly Emissions [Gt]", fontname="Times New Roman", fontsize=12)

    ax1.tick_params(axis='x', labelrotation=0, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)

    ax1.set_title(f"Feedstocks and Energy Carrier ({scenario_name})", fontname="Times New Roman", fontsize=14)

    #ax1_legend = ax1.legend(title='Energy Carrier', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False,
                            #title_fontsize='12', prop={'family': 'Times New Roman'})
    #ax2_legend = ax2.legend(loc='upper left', bbox_to_anchor=(1.05, 0.65), frameon=False,
                            #={'family': 'Times New Roman'})

    #ax1_legend.set_title(ax1_legend.get_title().get_text(), prop={'family': 'Times New Roman'})
    #ax2_legend.set_title(ax2_legend.get_title().get_text(), prop={'family': 'Times New Roman'})

    emissions_line = Line2D([0], [0], color='r', linestyle='-', linewidth=2, label='Cumulative Emissions')
    budget_line = Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Emissions Budget')

    handles, labels = ax1.get_legend_handles_labels()

    handles.extend([emissions_line, budget_line])
    labels.extend(['Cumulative Emissions', 'Emissions Budget'])

    ax1.legend(handles, labels, title='Legend', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

    plt.show()

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(True)

    def format_y_ticks(y, pos):
        return '{:,.0f}'.format(y / 100000)

    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_y_ticks))

    def format_y_ticks_right(y, pos):
        return '{:,.0f}'.format(y / 10)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(format_y_ticks_right))

    plt.tight_layout()
    plt.show()


def plot_emissions(scenario):
    industries = ['ammonia',
                  #'steel', 'cement','methanol', 'refining'
                  ]
    colors = ['mediumorchid',
              #'steelblue', 'darkslateblue', 'fuchsia', 'firebrick'
              ]
    industry_colors = dict(zip(industries, colors))

    fig, ax = plt.subplots(figsize=(10, 6))

    for industry in industries:
        df_emissions_carrier = pd.read_csv(f"{industry}_130324/carbon_emissions_carrier_grouped_{scenario}.csv")
        df_emissions_tech = pd.read_csv(f"{industry}_130324/carbon_emissions_tech_grouped_{scenario}.csv")

        total_emissions = df_emissions_carrier['carbon_emissions_carrier'] + df_emissions_tech[
            'carbon_emissions_technology']

        # Plotting
        total_emissions.plot(ax=ax, color=industry_colors[industry], label=industry.capitalize())

    ax.axhline(0, color='black', linewidth=1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    years = list(range(2024, 2051, 2))
    labels = [str(year) for year in years]
    positions = list(range(len(years)))

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)

    ax.set_xlabel('Year')
    ax.set_ylabel('Yearly emissions [10Mt]')
    #ax.set_title('Yearly Emissions by Industry')
    ax.legend()

    plt.show()

def draw_wedges_on_map(csv_path, shapefile_path, year, scenario, radius_factor=0.005, figsize=(15, 15)):

    df = pd.read_csv(csv_path)

    df['production'] = df[str(year)].astype(float)

    gdf = gpd.read_file(shapefile_path).to_crs('EPSG:3035')
    level = [0]
    gdf = gdf[gdf['LEVL_CODE'].isin(level)]

    country_color = 'ghostwhite'
    norway_color = 'darkgray'
    border_color = 'dimgrey'

    countries_to_exclude = ['IS', 'TR']
    gdf = gdf[~gdf['CNTR_CODE'].isin(countries_to_exclude)]

    #print(gdf[gdf['CNTR_CODE'] == 'DE']['geometry'])

    carrier_color_map = {
        'biomass_cement': 'forestgreen',
        'biomethane': 'darkkhaki',
        'coal': 'lightgray',
        'coal_for_cement': 'lightgrey',
        'dry_biomass': 'olivedrab',
        'natural_gas': 'dimgrey',
        'electricity': 'teal',
        'renewable_electricity': 'cyan'
    }

    france_centroid = (3713381.55, 2686876.92)
    #france_centroid = (4926344.393627158, 3512935.035715804)

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    gdf.plot(ax=ax, color=country_color, edgecolor=border_color)

    for country_code in ['NO', 'UK', 'CH']:
        specific_gdf = gdf[gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\\\\\\\\\", edgecolor='dimgrey', linewidth=0.8)

    '''if 'NO' in df['node'].values:
        norway = gdf[gdf['CNTR_CODE'] == 'NO']
        norway.plot(ax=ax, color=norway_color, edgecolor=border_color)

    #if 'UK' in df['node'].values:
    uk = gdf[gdf['CNTR_CODE'] == 'UK']
    uk.plot(ax=ax, color=norway_color, edgecolor=border_color)

    #if 'CH' in df['node'].values:
    ch = gdf[gdf['CNTR_CODE'] == 'CH']
    ch.plot(ax=ax, color=norway_color, edgecolor=border_color)'''

    legend_patches = []
    for node, group in df.groupby('node'):
        total_production = group['production'].sum()
        if total_production <= 0:
            continue


        radius = np.sqrt(total_production) * radius_factor * 100000

        if node == 'FR':
            centroid = france_centroid
        else:
            if node not in gdf['NUTS_ID'].values:
                print(f"Keine geografischen Daten für {node} gefunden.")
                continue
            country_geom = gdf.loc[gdf['NUTS_ID'] == node, 'geometry'].iloc[0]
            print(country_geom.centroid)
            centroid = (country_geom.centroid.x, country_geom.centroid.y)

        start_angle = 0
        for _, row in group.iterrows():
            carrier = row['carrier']
            production = row['production']
            if production <= 0:
                continue
            print('hallo')

            angle = (production / total_production) * 360
            color = carrier_color_map.get(carrier, 'white')

            wedge = Wedge(centroid, radius, start_angle, start_angle + angle,
                          edgecolor='black', facecolor=color, linewidth=0.8)
            ax.add_patch(wedge)

            start_angle += angle

    ax.set_xlim([-500000, 6000000])
    ax.set_ylim([-500000, 5400000])

    #plt.axis('equal')
    plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    #print(scenario)
   # print(year)
    plt.savefig(f"map_energy_carrier_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_carrier_diff(csv_path_base, csv_path_scenario, sc, y_limits=None):
    df_base = pd.read_csv(csv_path_base)
    df_base = df_base.groupby('carrier').sum()
    df_scenario = pd.read_csv(csv_path_scenario)
    df_scenario = df_scenario.groupby('carrier').sum()

    merged_df = df_base.merge(df_scenario, on='carrier', suffixes=('_base', '_scenario'))

    for i in range(14):
        merged_df[str(i)] = merged_df[str(i) + '_scenario'] - merged_df[str(i) + '_base']

    diff_df = merged_df.drop(columns=[str(i) + '_scenario' for i in range(14)] + [str(i) + '_base' for i in range(14)])

    diff_df.iloc[:, 0:] = diff_df.iloc[:, 0:].round(4)

    diff_df = diff_df.fillna(0)

    diff_df['sum'] = diff_df.iloc[:, 3:16].sum(axis=1) / 1000

    carrier_colors = {
        'biomass_cement': 'forestgreen',
        'biomethane': 'darkkhaki',
        'coal': 'gray',
        'coal_for_cement': 'grey',
        'dry_biomass': 'olivedrab',
        'natural_gas': 'dimgrey',
        'electricity': 'teal',
        'renewable_electricity': 'cyan'
    }

    relevant_carriers = ['biomethane', 'natural_gas', 'renewable_electricity', 'dry_biomass']

    plt.figure(figsize=(8, 5))
    for idx, row in diff_df.iterrows():
        if idx.lower() in relevant_carriers:
            plt.bar(idx, row['sum'], color=carrier_colors.get(idx.lower(), 'gray'))

    plt.axhline(0, color='black', linestyle='-')

    plt.xlabel('Energy Carrier')
    plt.ylabel('Input Energy Carrier [TWh]')
    # plt.title(f"Comparison of energy inputs 2024-2050 (Scenario{sc})")

    plt.xticks(rotation=45, ha='right')
    # plt.yticks(fontname='Times New Roman')

    if y_limits is not None:
        plt.ylim(y_limits)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def generate_sankey_diagram(folder_path, scenario, target_technologies, intermediate_technologies, year, title, save_file):

    scenario_name_mapping = {
        'scenario_': 'Baseline Scenario',
        #'scenario_high_demand': 'High Demand Scenario',
        #'scenario_low_demand': 'Low Demand Scenario'
    }
    scenario_name = scenario_name_mapping.get(scenario, scenario)

    file_path_input = os.path.join(folder_path, f"flow_conversion_input_grouped_{scenario}.csv")
    inputs_df = pd.read_csv(file_path_input)
    file_path_output = os.path.join(folder_path, f"flow_conversion_output_grouped_{scenario}.csv")
    outputs_df = pd.read_csv(file_path_output)

    technologies_to_remove = ['biomethane_SMR', 'biomethane_SMR_CCS', 'biomethane_haber_bosch', 'biomethane_DRI', 'biomethane_SMR_methanol']

    for year_col in map(str, range(14)):
        if year_col in inputs_df.columns and year_col in outputs_df.columns:
            biomethane_output_indices = outputs_df[outputs_df['carrier'] == 'biomethane'].index
            natural_gas_input_indices = inputs_df[inputs_df['carrier'] == 'natural_gas'].index

            if not biomethane_output_indices.empty and not natural_gas_input_indices.empty:
                for index in natural_gas_input_indices:
                    inputs_df.at[index, year_col] -= outputs_df.at[biomethane_output_indices[0], year_col]


    outputs_df = outputs_df[~outputs_df['technology'].isin(technologies_to_remove)]
    inputs_df['technology'] = inputs_df['technology'].str.replace('biomethane_', '')

    for year_col in map(str, range(14)):

        if year_col in inputs_df.columns and year_col in outputs_df.columns:

            coal_BF_BOF_input_indices = inputs_df[(inputs_df['carrier'] == 'coal')].index
            iron_ore_BF_BOF_input_indices = inputs_df[(inputs_df['carrier'] == 'iron_ore')].index
            limestone_BF_BOF_input_indices = inputs_df[(inputs_df['carrier'] == 'limestone')].index
            direct_reduced_iron_EAF_indices = inputs_df[(inputs_df['carrier'] == 'direct_reduced_iron') & (inputs_df['technology'] == 'EAF')].index
            scrap_EAF_input_indices = inputs_df[(inputs_df['carrier'] == 'scrap') & (inputs_df['technology'] == 'scrap_conversion_EAF')].index

            scrap_BF_BOF_output_coal_indices = outputs_df[(outputs_df['carrier'] == 'coal')].index
            scrap_BF_BOF_output_iron_ore_indices = outputs_df[(outputs_df['carrier'] == 'iron_ore')].index
            scrap_BF_BOF_output_limestone_indices = outputs_df[(outputs_df['carrier'] == 'limestone')].index


            if not scrap_BF_BOF_output_coal_indices.empty and not coal_BF_BOF_input_indices.empty:
                for index in coal_BF_BOF_input_indices:
                    inputs_df.at[index, year_col] -= outputs_df.at[scrap_BF_BOF_output_coal_indices[0], year_col]

            if not scrap_BF_BOF_output_iron_ore_indices.empty and not iron_ore_BF_BOF_input_indices.empty:
                for index in iron_ore_BF_BOF_input_indices:
                    inputs_df.at[index, year_col] -= outputs_df.at[
                        scrap_BF_BOF_output_iron_ore_indices[0], year_col]

            if not scrap_BF_BOF_output_limestone_indices.empty and not limestone_BF_BOF_input_indices.empty:
                for index in limestone_BF_BOF_input_indices:
                    inputs_df.at[index, year_col] -= outputs_df.at[
                        scrap_BF_BOF_output_limestone_indices[0], year_col]

            if not direct_reduced_iron_EAF_indices.empty and not scrap_EAF_input_indices.empty:
                for index in direct_reduced_iron_EAF_indices:
                    inputs_df.at[index, year_col] -= inputs_df.at[
                        scrap_EAF_input_indices[0], year_col]

    inputs_df['technology'] = inputs_df['technology'].str.replace('h2_to_ng', 'DRI')
    inputs_df['technology'] = inputs_df['technology'].str.replace('scrap_conversion_EAF', 'EAF')
    inputs_df['technology'] = inputs_df['technology'].str.replace('scrap_conversion_BF_BOF', 'BF_BOF')

    technologies_to_remove_steel = ['h2_to_ng', 'scrap_conversion_EAF', 'scrap_conversion_BF_BOF']
    outputs_df = outputs_df[~outputs_df['technology'].isin(technologies_to_remove_steel)]

    for year_col in map(str, range(14)):
        if year_col in inputs_df.columns and year_col in outputs_df.columns:

            carbon_methanol_input_indices = inputs_df[(inputs_df['carrier'] == 'carbon_methanol') & (inputs_df['technology'] == 'methanol_synthesis')].index
            #carbon_output_indices = outputs_df[(outputs_df['carrier'] == 'carbon') & (inputs_df['technology'] == 'carbon_evaporation')].index
            #natural_gas_input_indices = inputs_df[(inputs_df['carrier'] == 'natural_gas') & (inputs_df['technology'] == 'SMR_methanol')].index
            #biomethane_output_indices = outputs_df[(outputs_df['carrier'] == 'biomethane') & (outputs_df['technology'] == 'anaerobic_digestion')].index

            #if not carbon_methanol_input_indices.empty and not carbon_output_indices.empty:
               # inputs_df.at[index, year_col] -= outputs_df.at[carbon_output_indices[0], year_col]

            #if not natural_gas_input_indices.empty and not carbon_output_indices.empty:
             #   inputs_df.at[index, year_col] -= outputs_df.at[biomethane_output_indices[0], year_col]

    inputs_df['technology'] = inputs_df['technology'].str.replace('carbon_conversion', 'methanol_synthesis')
    inputs_df['carrier'] = inputs_df['carrier'].str.replace('carbon_methanol', 'carbon')
    outputs_df['carrier'] = outputs_df['carrier'].str.replace('carbon_methanol', 'carbon')

    technologies_to_remove_methanol = ['carbon_conversion']
    outputs_df = outputs_df[~outputs_df['technology'].isin(technologies_to_remove_methanol)]


    for year_col in map(str, range(14)):

        if year_col in inputs_df.columns and year_col in outputs_df.columns:

            coal_for_cement_input_indices = inputs_df[(inputs_df['carrier'] == 'coal_for_cement')].index
            biomass_output_indices = outputs_df[(outputs_df['technology'] == 'biomass_to_coal_conversion')].index
            hydrogen_output_indices = outputs_df[(outputs_df['technology'] == 'hydrogen_for_cement_conversion')].index

            if not coal_for_cement_input_indices.empty and not biomass_output_indices.empty:
                for index in coal_for_cement_input_indices:
                    inputs_df.at[index, year_col] -= outputs_df.at[
                        biomass_output_indices[0], year_col]

            if not coal_for_cement_input_indices.empty and not hydrogen_output_indices.empty:
                for index in coal_for_cement_input_indices:
                    inputs_df.at[index, year_col] -= outputs_df.at[
                        hydrogen_output_indices[0], year_col]

    inputs_df['technology'] = inputs_df['technology'].str.replace('biomass_to_coal_conversion', 'cement_plant')
    inputs_df['technology'] = inputs_df['technology'].str.replace('hydrogen_for_cement_conversion', 'cement_plant')

    technologies_to_remove_cement = ['biomass_to_coal_conversion', 'hydrogen_for_cement_conversion']
    outputs_df = outputs_df[~outputs_df['technology'].isin(technologies_to_remove_cement)]


    input_techs_target = inputs_df[inputs_df['technology'].isin(target_technologies)]

    input_techs_intermediate = inputs_df[inputs_df['technology'].isin(intermediate_technologies)]

    output_techs_intermediate = outputs_df[outputs_df['technology'].isin(intermediate_technologies)]

    output_techs_target = outputs_df[outputs_df['technology'].isin(target_technologies)]

    input_sankey_target = pd.DataFrame({
        'source': input_techs_target['carrier'],
        'target': input_techs_target['technology'],
        'value': input_techs_target[year]
    })

    input_sankey_intermediate = pd.DataFrame({
        'source': input_techs_intermediate['carrier'],
        'target': input_techs_intermediate['technology'],
        'value': input_techs_intermediate[year]
    })

    output_sankey_intermediate = pd.DataFrame({
        'source': output_techs_intermediate['technology'],
        'target': output_techs_intermediate['carrier'],
        'value': output_techs_intermediate[year]
    })

    output_sankey_target = pd.DataFrame({
        'source': output_techs_target['technology'],
        'target': output_techs_target['carrier'],
        'value': output_techs_target[year]
    })

    links = pd.concat([input_sankey_target, output_sankey_intermediate, input_sankey_intermediate, output_sankey_target], axis=0)

    unique_source_target = list(pd.unique(links[['source', 'target']].values.ravel('K')))
    mapping_dict = {k:v for v, k in enumerate(unique_source_target)}
    inv_mapping_dict = {v: k for k, v in mapping_dict.items()}
    links['source'] = links['source'].map(mapping_dict)
    links['target'] = links['target'].map(mapping_dict)

    links_dict = links.to_dict(orient="list")

    color_mapping = {
        'steel': 'royalblue',
        'steel_BF_BOF': 'steelblue',
        'steel_DRI_EAF': 'skyblue',
        'scrap': 'cornflowerblue',
        'steel_inputs': 'mediumblue',
        'cement': 'darkslateblue',
        'ammonia': 'darkorchid',
        'hydrogen': 'sandybrown',
        'SMR': 'goldenrod',
        'SMR_CCS': 'darkgoldenrod',
        'methanol': 'fuchsia',
        'gasification_methanol': 'violet',
        'methanol_from_hydrogen': 'palevioletred',
        'refining': 'indianred',
        'electricity': 'aqua',
        'CCS': 'gainsboro',
        #'other_techs': 'red',
        'natural_gas': 'dimgrey',
        'wet_biomass': 'yellowgreen',
        'dry_biomass': 'olivedrab',
        'biomass_cement': 'forestgreen',
        'biomethane': 'darkkhaki',
        'coal': 'gray',
        'electrolysis': 'darksalmon',
        'gasification': 'wheat',
        'gasification_CCS': 'peru',
        'nitrogen': 'mediumorchid',
        'oxygen': 'darkslateblue',
        'BF_BOF_CCS': 'lightsteelblue',
        'BF_BOF_CCS': 'lightsteelblue',
        'e_haber_bosch': 'darkviolet',
        'haber_bosch': 'mediumorchid',
        'ASU': 'mediumpurple',
        'fossil_fuel': 'black',
        'biomass': 'darkseagreen',
        'default_color': 'green'

    }

    category_mapping = {
        'BF_BOF': 'steel_BF_BOF',
        'EAF': 'steel_DRI_EAF',
        'DRI': 'steel_DRI_EAF',
        'scrap': 'scrap',
        'iron_ore': 'steel_inputs',
        'limestone': 'steel_inputs',
        'h2_to_ng': 'steel_inputs',
        'scrap_conversion_EAF': 'scrap',
        'scrap_conversion_BF_BOF': 'scrap',
        'steel': 'steel',
        'coal': 'coal',
        'natural_gas': 'natural_gas',
        'biomethane': 'biomethane',
        'biomethane_conversion': 'biomethane',
        'electricity': 'electricity',
        'direct_reduced_iron': 'steel_DRI_EAF',
        'carbon_liquid': 'CCS',
        'ASU': 'ASU',
        'DAC': 'CCS',
        'SMR': 'SMR',
        'SMR_CCS': 'SMR_CCS',
        'anaerobic_digestion': 'biomethane',
        'electrolysis': 'electrolysis',
        'gasification': 'gasification',
        'gasification_CCS': 'gasification_CCS',
        'wet_biomass': 'wet_biomass',
        'dry_biomass': 'dry_biomass',
        'hydrogen': 'hydrogen',
        'nitrogen': 'nitrogen',
        'oxygen': 'oxygen',
        'carbon': 'CCS',
        'oil_products': 'refining',
        'methanol': 'methanol',
        'ammonia': 'ammonia',
        'BF_BOF_CCS': 'BF_BOF_CCS',
        'carbon_liquefication': 'CCS',
        'carbon_removal': 'CCS',
        'carbon_storage': 'CCS',
        'e_haber_bosch': 'e_haber_bosch',
        'haber_bosch': 'haber_bosch',
        'gasification_methanol': 'gasification_methanol',
        'methanol_from_hydrogen': 'methanol_from_hydrogen',
        'methanol_synthesis': 'methanol',
        'refinery': 'refining',
        'cement_plant': 'cement',
        'cement_plant_oxy_combustion': 'CCS',
        'cement_plant_post_comb': 'CCS',
        'photovoltaics': 'electricity',
        'pv_ground': 'electricity',
        'pv_rooftop': 'electricity',
        'wind_offshore': 'electricity',
        'wind_onshore': 'electricity',
        'coal_for_cement': 'fossil_fuel',
        'biomass_cement': 'biomass',
        'biomass_to_coal_conversion': 'biomass',
        'cement': 'cement',
        'hydrogen_for_cement_conversion': 'hydrogen',
        'carbon_evaporation': 'CCS',
        'hydrogen_compressor_low': 'hydrogen',
        'hydrogen_decompressor': 'hydrogen',
        'carbon_methanol': 'methanol',
        'SMR_methanol': 'methanol',
        'gasification_methanol_h2': 'methanol',
        'carbon_conversion': 'methanol',
        'carbon_methanol_conversion': 'methanol',
        'biomethane_SMR': 'hydrogen',
        'biomethane_SMR_CCS': 'hydrogen',
        'biomethane_SMR_methanol': 'methanol',
        'biomethane_haber_bosch':'ammonia'

    }

    for cat in category_mapping.values():
        if cat not in color_mapping:
            print(f"Missing color mapping for category: {cat}")

    colors = [color_mapping.get(category_mapping.get(tech, 'other_techs')) for tech in unique_source_target]

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "sankey"}]])
    #print([src for src in links['source']])
    fig.add_trace(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            #line=dict(color='black', width=0.5),
            label=unique_source_target,
            color=colors
        ),
        link=dict(
            #color=["rgba"+str(matplotlib.colors.to_rgba(color_mapping.get(category_mapping.get(inv_mapping_dict[link], 'other_techs')), alpha=0.6)) for link in links['source']],
            color=[
                "rgba" + str(matplotlib.colors.to_rgba(
                    color_mapping.get(category_mapping.get(inv_mapping_dict[link], 'other_techs'), 'default_color'),
                    # Use a default color if not found
                    alpha=0.6
                )) for link in links['source']
            ],
            #[str(matplotlib.colors.to_rgba(color_mapping[links_dict['source'][str(src)]])).replace("0.6", "1.0") for src in links_dict['source']],
            source=links_dict['source'],
            target=links_dict['target'],
            value=links_dict['value'],
            #text=[f"Value: {link}" for link in links_dict['value']],
            #hoverinfo='all'
            label=[f"{source} to {target}" for source, target in zip(links_dict['source'], links_dict['target'])],
            hovertemplate='%{value}'
        )
    ))

    if isinstance(year, str):
        year = int(year)
    displayed_year = year * 2 + 2024
    fig.update_layout(title_text=f"{title} {displayed_year} ({scenario_name})", font_size=18)

    fig.update_layout(font=dict(size=26))

    if save_file:
        subfolder_name = "sankey"
        subfolder_path = os.path.join(folder_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        png_file_path = os.path.join(subfolder_path, f"{target_technologies}_{displayed_year}_{scenario}.png")

        fig.write_image(png_file_path, format="png", width=1600, height=1200)

    fig.show()


def plot_outputs(folder_path,  scenario, carrier, save_file):
    scenario_name_mapping = {
        'scenario_': 'Baseline Scenario',
       # 'scenario_high_demand': 'High Demand Scenario',
       # 'scenario_low_demand': 'Low Demand Scenario'
    }
    scenario_name = scenario_name_mapping.get(scenario, scenario)

    file_name = os.path.join(folder_path, f"flow_conversion_output_grouped_{scenario}.csv")
    df_output = pd.read_csv(file_name)

    file_name = os.path.join(folder_path, f"flow_conversion_input_grouped_{scenario}.csv")
    df_input = pd.read_csv(file_name)

    filtered_rows = df_input[(df_input['technology'] == 'biomethane_SMR') & (df_input['carrier'] == 'biomethane')]
    filtered_rows_1 = df_input[(df_input['technology'] == 'biomethane_SMR_CCS') & (df_input['carrier'] == 'biomethane')]


    columns_to_multiply = [str(year) for year in range(0, 14)]
    filtered_rows[columns_to_multiply] = filtered_rows[columns_to_multiply].apply(lambda x: x / 1.2987)
    filtered_rows_1[columns_to_multiply] = filtered_rows_1[columns_to_multiply].apply(lambda x: x / 1.2987)

    new_row = filtered_rows[columns_to_multiply].mean().to_dict()
    new_row['technology'] = 'SMR_biomethane'
    new_row['carrier'] = 'hydrogen'

    df_output = pd.concat([df_output, pd.DataFrame([new_row])], ignore_index=True)

    new_row_1 = filtered_rows_1[columns_to_multiply].mean().to_dict()
    new_row_1['technology'] = 'SMR_CCS_biomethane'
    new_row_1['carrier'] = 'hydrogen'

    df_output = pd.concat([df_output, pd.DataFrame([new_row_1])], ignore_index=True)

    smr_hydrogen_row = df_output[(df_output['technology'] == 'SMR_biomethane') & (df_output['carrier'] == 'hydrogen')]
    smr_values = smr_hydrogen_row.iloc[0][[str(i) for i in range(14)]].values

    smr_biomethane_hydrogen_row = df_output[
        (df_output['technology'] == 'SMR') & (df_output['carrier'] == 'hydrogen')]
    smr_biomethane_hydrogen_index = smr_biomethane_hydrogen_row.index[0]

    for i in range(14):
        df_output.at[smr_biomethane_hydrogen_index, str(i)] -= smr_values[i]

    smr_hydrogen_row = df_output[(df_output['technology'] == 'SMR_CCS_biomethane') & (df_output['carrier'] == 'hydrogen')]
    smr_values = smr_hydrogen_row.iloc[0][[str(i) for i in range(14)]].values

    smr_biomethane_hydrogen_row = df_output[
        (df_output['technology'] == 'SMR_CCS') & (df_output['carrier'] == 'hydrogen')]
    smr_biomethane_hydrogen_index = smr_biomethane_hydrogen_row.index[0]

    for i in range(14):
        df_output.at[smr_biomethane_hydrogen_index, str(i)] -= smr_values[i]

    file_name_emissions = os.path.join(folder_path, f"carbon_emissions_cumulative_{scenario}.csv")
    df_emissions_cumulative = pd.read_csv(file_name_emissions)

    file_name_emissions_tech = os.path.join(folder_path, f"carbon_emissions_tech_grouped_{scenario}.csv")
    df_emissions_tech = pd.read_csv(file_name_emissions_tech)

    file_name_emissions_carrier = os.path.join(folder_path, f"carbon_emissions_carrier_grouped_{scenario}.csv")
    df_emissions_carrier = pd.read_csv(file_name_emissions_carrier)

    grouped_df = df_output[df_output['carrier'] == carrier]

    year_mapping = {str(i): str(2024 + 2 * i) for i in range(14)}
    grouped_df.rename(columns=year_mapping, inplace=True)
    grouped_df.set_index('technology', inplace=True)
    grouped_df = grouped_df.dropna()
    grouped_df_values = grouped_df.drop(['carrier'], axis=1).transpose()

    desired_order = [
        'SMR', 'SMR_biomethane', 'SMR_CCS', 'SMR_CCS_biomethane',
        'electrolysis', 'gasification', 'gasification_CCS'
    ]

    available_technologies = [tech for tech in desired_order if tech in grouped_df_values.columns]

    grouped_df_values = grouped_df_values[available_technologies]

    if carrier == 'hydrogen':
        technology_colors = {
            'SMR': 'darkgray',
            'SMR_CCS': 'deeppink',
            'electrolysis': 'lime',
            'gasification': 'darkturquoise',
            'gasification_CCS': 'cyan',
            'SMR_biomethane': 'lightgrey',
            'SMR_CCS_biomethane': 'pink'
        }
        #palette = [technology_colors[tech] for tech in grouped_df_values.columns if tech in technology_colors]
    #else:
    #palette = sns.color_palette(n_colors=len(grouped_df_values.columns))

        filtered_technologies = [tech for tech in grouped_df_values.columns if tech in technology_colors]
        grouped_df_values = grouped_df_values[filtered_technologies]

        palette = [technology_colors[tech] for tech in grouped_df_values.columns]

    else:
        #palette = ['gray'] * len(grouped_df_values.columns)
        palette = plt.cm.tab20(np.linspace(0, 1, len(grouped_df_values.columns)))

    fig, ax1 = plt.subplots(figsize=(16, 10))
    ax2 = ax1.twinx()

    cumulative_emissions_line = df_emissions_cumulative['carbon_emissions_cumulative'].plot(ax=ax2, color='black', label='Cumulative Emissions')
    emissions_budget_line, = ax2.plot(df_emissions_cumulative.index, [4698963]*len(df_emissions_cumulative.index), label='Emissions Budget', color="black", linestyle='--')
    #yearly_emissions = (df_emissions_tech['carbon_emissions_technology'] + df_emissions_carrier['carbon_emissions_carrier']).plot(ax=ax2, color='r', label='Yearly Emissions')
    industries = ['cement']
    for industry in industries:
        df_emissions_carrier = pd.read_csv(f"{industry}_130324/carbon_emissions_carrier_grouped_{scenario}.csv")
        df_emissions_tech = pd.read_csv(f"{industry}_130324/carbon_emissions_tech_grouped_{scenario}.csv")

        #total_emissions = df_emissions_carrier['carbon_emissions_carrier'] + df_emissions_tech[
            #'carbon_emissions_technology']

        #total_emissions = ax2.plot(total_emissions, color='red', label='Yearly Emissions')


    bottom = np.zeros(len(grouped_df_values))

    # Plotting stacked bars with dynamic colors
    for idx, technology in enumerate(grouped_df_values.columns):
        ax1.bar(grouped_df_values.index, grouped_df_values[technology], bottom=bottom, color=palette[idx], label=technology, width=0.6)
        bottom += grouped_df_values[technology].values

    # Legends and adjustments
    ax1.legend(title='Technology', bbox_to_anchor=(1.08, 1), loc='upper left', frameon=False)
    ax2.legend(loc='upper left', bbox_to_anchor=(1.08, 0.73), frameon=False)

    plt.subplots_adjust(right=0.75)
    ax1.set_xlabel("Year", fontsize=22)

    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, fontsize=14)

    #y_labels = [f"{int(label) / 1000000:.2f}" for label in ax2.get_yticks()]
    #ax2.set_yticklabels(y_labels)

    y_labels_right = [f"{int(label / 1000000)}" for label in ax2.get_yticks()]
    ax2.set_yticklabels(y_labels_right)

    ax2.set_ylabel("Cumulative Emissions [Gt CO2 eq.]", fontsize=22)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    #plt.title(f"{carrier.capitalize()} Output by Year and Technology ({scenario_name})", fontname="Times New Roman", fontsize=14)

    ax1.set_ylabel("Yearly Hydrogen Production [TWh]" if carrier in ['hydrogen', 'electricity', 'natural_gas', 'biomethane'] else "Yearly Production [Mt]", fontsize=22)

    ax1.set_ylim([0, 850000])
    ax1.set_yticks(range(0, 850001, 100000))

    y_labels_left = [f"{label / 1000}" for label in ax1.get_yticks()]
    ax1.set_yticklabels(y_labels_left)

    if save_file:
        subfolder_name = "output_bar_charts"
        subfolder_path = os.path.join(folder_path, subfolder_name)
        os.makedirs(subfolder_path, exist_ok=True)
        png_file_path = os.path.join(subfolder_path, f"{carrier}_bar_chart_{scenario}.png")
        plt.savefig(png_file_path, bbox_inches='tight')

    plt.show()


def plot_outputs_carbon(scenario1, scenario2):

    df1 = res_scenario.get_total("flow_conversion_input", scenario=scenario1).xs('carbon_storage')
    grouped_df1 = df1.xs('carbon_liquid').sum()

    df2 = res_scenario.get_total("flow_conversion_input", scenario=scenario2).xs('carbon_storage')
    grouped_df2 = df2.xs('carbon_liquid').sum()

    years = list(range(2024, 2051, 2))

    bar_width = 0.45
    years1 = [x - bar_width / 2 for x in range(len(years))]
    years2 = [x + bar_width / 2 for x in range(len(years))]

    plt.figure(figsize=(12, 7))

    plt.bar(years1, grouped_df1.values, width=bar_width, color='cadetblue', label='Baseline')
    plt.bar(years2, grouped_df2.values, width=bar_width, color='lightskyblue', label='Biomass_high_price_10')

    plt.xlabel('Year')
    plt.ylabel('Stored carbon [Mt]')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    formatter = FuncFormatter(lambda y, _: '{:,.0f}'.format(y / 1000))
    ax.yaxis.set_major_formatter(formatter)

    plt.xticks(range(len(years)), years)
    plt.legend()

    plt.axhline(y=82817, color='r', linestyle='--', linewidth=2)

    plt.show()


def plot_capacities(folder_path, scenario, technology, save_file):

    scenario_name_mapping = {
        'scenario_': 'Baseline Scenario',
        #'scenario_high_demand': 'High Demand Scenario',
       # 'scenario_low_demand': 'Low Demand Scenario'
    }
    scenario_name = scenario_name_mapping.get(scenario, scenario)

    plt.rcParams["font.family"] = "Times New Roman"

    file_name = os.path.join(folder_path, f"flow_conversion_output_grouped_{scenario}.csv")
    df_output = pd.read_csv(file_name)

    grouped_df = df_output[df_output['technology'] == technology]
    year_mapping = {str(i): str(2024 + 2 * i) for i in range(14)}

    grouped_df.rename(columns=year_mapping, inplace=True)
    grouped_df = grouped_df.dropna()

    palette = sns.color_palette("dark", n_colors=len(grouped_df.columns))

    fig, ax = plt.subplots(figsize=(10, 6))
    grouped_df.plot(kind='bar', stacked=False, ax=ax, color=palette)

    ax.legend(title='Technology', bbox_to_anchor=(1, 0.5), loc='center left', frameon=False)
    plt.subplots_adjust(right=0.8)

    ax.set_xlabel("Year", fontname="Times New Roman", fontsize=12)
    ax.set_ylabel("Capacity in Mt", fontname="Times New Roman", fontsize=12)

    # Convert y-axis labels to represent values in Megatons
    y_labels = [f"{int(label) / 1000:.2f}" for label in ax.get_yticks()]
    ax.set_yticklabels(y_labels)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.title(f"{technology.capitalize()} Capacity by Year and Technology ({scenario_name})")
    plt.xlabel("Year")
    plt.ylabel(f"Yearly {technology.capitalize()} Capacity [Mt]")

    if save_file:
        subfolder_path = os.path.join(folder_path, "capacities")
        os.makedirs(subfolder_path, exist_ok=True)
        png_file_path = os.path.join(subfolder_path, f"{technology}_bar_chart_{scenario}.png")
        plt.savefig(png_file_path, bbox_inches='tight')

    plt.show()

def plot_dataframe_on_map(df, df_input, technology_column, output_columns, shapefile_path=None, save_png=True):

    columns = ['node', 'technology', 'carrier'] + [str(year) for year in range(0, 14)]
    df_output = df[df['carrier'] == 'hydrogen']

    columns_to_operate = [str(year) for year in range(0, 14)]

    for node in df_input['node'].unique():
        filtered_rows = df_input[(df_input['technology'] == 'biomethane_SMR') &
                                 (df_input['carrier'] == 'biomethane') &
                                 (df_input['node'] == node)]

        filtered_rows_1 = df_input[(df_input['technology'] == 'biomethane_SMR_CCS') &
                                   (df_input['carrier'] == 'biomethane') &
                                   (df_input['node'] == node)]

        if not filtered_rows.empty:
            adjusted_values = (filtered_rows[columns_to_operate] / 1.2987).iloc[0].to_dict()
            new_row = {**{'node': node, 'technology': 'SMR_biomethane', 'carrier': 'hydrogen'}, **adjusted_values}
            df_output = pd.concat([df_output, pd.DataFrame([new_row])], ignore_index=True)

        if not filtered_rows_1.empty:
            adjusted_values_1 = (filtered_rows_1[columns_to_operate] / 1.2987).iloc[0].to_dict()
            new_row_1 = {**{'node': node, 'technology': 'SMR_CCS_biomethane', 'carrier': 'hydrogen'},
                         **adjusted_values_1}
            df_output = pd.concat([df_output, pd.DataFrame([new_row_1])], ignore_index=True)

    columns_to_check = [str(i) for i in range(14)]

    df_output = df_output.loc[df_output[columns_to_check].sum(axis=1) > 0]

    columns_to_operate = [str(i) for i in range(14)]

    df_smr = df_output[df_output['technology'] == 'SMR'].set_index('node')
    df_smr_biomethane = df_output[df_output['technology'] == 'SMR_biomethane'].set_index('node')

    for node, row in df_smr.iterrows():
        if node in df_smr_biomethane.index:
            df_output.loc[(df_output['node'] == node) & (df_output['technology'] == 'SMR'), columns_to_operate] -= df_smr_biomethane.loc[
                node, columns_to_operate]

    columns_to_operate = [str(i) for i in range(14)]

    df_smr_ccs = df_output[df_output['technology'] == 'SMR_CCS'].set_index('node')
    df_smr_ccs_biomethane = df_output[df_output['technology'] == 'SMR_CCS_biomethane'].set_index('node')

    for node, row in df_smr_ccs.iterrows():
        if node in df_smr_ccs_biomethane.index:
            df_output.loc[
                (df_output['node'] == node) & (df_output['technology'] == 'SMR_CCS'), columns_to_operate] -= \
            df_smr_ccs_biomethane.loc[
                node, columns_to_operate]

    columns_to_operate = [str(i) for i in range(14)]
    df_output[columns_to_operate] = df_output[columns_to_operate].clip(lower=0)

    df_output.to_csv('output_3.csv', index=False)

    gdf = gpd.read_file(shapefile_path).to_crs(epsg=3035)

    level = [0]
    country_gdf = gdf[gdf['LEVL_CODE'].isin(level)]

    countries_to_exclude = ['IS', 'TR']
    country_gdf = country_gdf[~country_gdf['CNTR_CODE'].isin(countries_to_exclude)]

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    for year_col in output_columns:
        year = int(year_col)
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=300)

        country_gdf.plot(ax=ax, color='ghostwhite', edgecolor='dimgrey', linewidth=0.8)

        for country_code in ['NO', 'UK', 'CH']:
            specific_gdf = country_gdf[country_gdf['CNTR_CODE'] == country_code]
            specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\\\\\\\\\", edgecolor='dimgrey', linewidth=0.8)

        for index, row in df_output.iterrows():
            technology = row[technology_column]
            color = 'lime' if technology == 'electrolysis' else 'pink' if technology == 'SMR_CCS_biomethane' \
                else 'lightgrey' if technology == 'SMR_biomethane' else 'darkgray' if technology == 'SMR' \
                else 'deeppink' if technology == 'SMR_CCS' else 'darkturquoise' if technology == 'gasification' \
                else 'cyan' if technology == 'gasification_CCS' else 'None' if technology == 'hydrogen_decompressor' else 'None'
            output = row[year_col]
            if output == 0:
                continue
            radius = output * 0.02
            node = row['node']
            point = gdf.loc[gdf['NUTS_ID'] == node].to_crs(epsg=3035).geometry.centroid.iloc[0]
            ax.scatter(point.x, point.y, s=radius, color=color, label=technology, alpha=0.7)

        #ax.set_title(f"Hydrogen Production in Europe {year}")

        plt.axis('off')
        ax.set_xlim([20000, 6000000])
        ax.set_ylim([20000, 5500000])
        plt.tight_layout()
        plt.savefig(f"hyrogen_production_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)

        plt.show()

scenario = 'scenario_'
opex= res_scenario.get_total('cost_opex_total', scenario=scenario).sum()
#opex= res_scenario.get_total('cost_opex_total').sum()
capex = res_scenario.get_total('cost_capex_total', scenario=scenario).sum()
#capex = res_scenario.get_total('cost_capex_total').sum()
carrier_cost = res_scenario.get_total('cost_carrier', scenario=scenario).sum().sum()
#carrier_cost = res_scenario.get_total('cost_carrier').sum().sum()
print(opex)
print(capex)
print(carrier_cost)
print(opex + capex + carrier_cost)

def get_industry_capex_data(scenario):
    industries = ['ammonia', 'steel', 'cement', 'methanol', 'refining']
    df_industries = pd.DataFrame()

    for industry in industries:
        res_scenario = Results(f"../outputs/hard_to_abate_{industry}_130324/")
        df_industry = res_scenario.get_total("cost_capex_total", scenario)
        df_industry = df_industry.loc[scenario]
        df_industries[industry] = df_industry.squeeze()

    return df_industries

def get_total_industry_capex_data(scenario):
    res_scenario = Results("../outputs/hard_to_abate_scenarios_140324/")
    df_all_industries_total = res_scenario.get_total("cost_capex_total", scenario)
    df_all_industries_total = df_all_industries_total.loc[scenario]

    return df_all_industries_total

def get_industry_opex_data(scenario):
    industries = ['ammonia', 'steel', 'cement', 'methanol', 'refining']
    df_industries = pd.DataFrame()

    for industry in industries:
        res_scenario = Results(f"../outputs/hard_to_abate_{industry}_130324/")
        df_industry = res_scenario.get_total("cost_opex_total", scenario)
        df_industry = df_industry.loc["scenario_"]
        df_industries[industry] = df_industry.squeeze()

    return df_industries

def get_total_industry_opex_data(scenario):
    res_scenario = Results("../outputs/hard_to_abate_scenarios_140324/")
    df_all_industries_total = res_scenario.get_total("cost_opex_total", scenario)
    df_all_industries_total = df_all_industries_total.loc[scenario]

    return df_all_industries_total

def get_industry_carrier_costs_data(scenario):
    industries = ['ammonia', 'steel', 'cement', 'methanol', 'refining']
    df_all_industries = pd.DataFrame()

    for industry in industries:
        res_scenario = Results(f"../outputs/hard_to_abate_{industry}_130324/")
        df_industry = res_scenario.get_total("cost_carrier")
        df_industry = df_industry.loc[scenario]
        df_industry_sum = df_industry.sum()
        df_all_industries[industry] = df_industry_sum

    return df_all_industries

def get_total_industry_cost_carrier_data(scenario):
    res_scenario = Results("../outputs/hard_to_abate_scenarios_140324/")
    df_all_industries_total = res_scenario.get_total("cost_carrier")
    df_all_industries_total = df_all_industries_total.loc[scenario]
    total_carrier_costs_by_year = df_all_industries_total.sum()
    print(total_carrier_costs_by_year)

    return total_carrier_costs_by_year

def plot_costs_with_unique_colors(scenario):
    df_capex = get_industry_capex_data(scenario)
    df_opex = get_industry_opex_data(scenario)
    df_carrier_costs = get_industry_carrier_costs_data(scenario)
    total_all = get_total_industry_capex_data(scenario) + get_total_industry_opex_data(
        scenario) + get_total_industry_cost_carrier_data(scenario)

    years = np.arange(2024, 2051, 2)
    index = np.arange(len(years))
    bar_width = 0.35

    colors = {
        'ammonia': {'capex': 'mediumorchid', 'opex': '#9B30FF', 'carrier_costs': '#D15FEE'},
        'steel': {'capex': 'steelblue', 'opex': '#3B9F9F', 'carrier_costs': '#5CACEE'},
        'cement': {'capex': 'darkslateblue', 'opex': '#6A5ACD', 'carrier_costs': '#836FFF'},
        'methanol': {'capex': 'fuchsia', 'opex': '#E800E8', 'carrier_costs': '#FF77FF'},
        'refining': {'capex': 'firebrick', 'opex': '#CD2626', 'carrier_costs': '#FF3030'}
    }

    fig, ax = plt.subplots(figsize=(18, 10))

    legend_elements = []

    bottom = np.zeros(len(years))

    for industry, color_map in colors.items():
        for cost_type, color in color_map.items():
            if cost_type == 'capex':
                value = df_capex[industry]
            elif cost_type == 'opex':
                value = df_opex[industry]
            else:
                value = df_carrier_costs[industry]

            bars = ax.bar(index - bar_width / 2, value, bottom=bottom, color=color, width=bar_width,
                          label=f"{industry} {cost_type}")
            bottom += value.values

            if np.all(bottom == value.values):
                legend_elements.append(bars[0])

    ax.bar(index + bar_width / 2, total_all, color='grey', width=bar_width, label='totex integrated optimization')

    handles, labels = ax.get_legend_handles_labels()
    unique_labels, unique_handles = [], []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)
    #title_font = FontProperties(family='Times New Roman', size=12)
    ax.legend(unique_handles, unique_labels, loc='upper left', bbox_to_anchor=(1, 1),
              #title="Cost Types per Industry",
              frameon=False,
              prop={'size': 22})

    ax.set_xlabel('Year', fontsize=22)
    ax.set_ylabel('Yearly Costs in B€', fontsize=22)
    #ax.set_title('Cost Comparison: Individual Industries and Overall Optimization', fontname='Times New Roman')
    ax.set_xticks(index)
    ax.set_xticklabels(years)

    y_labels = [f"{int(label) / 1000:.2f}" for label in ax.get_yticks()]
    ax.set_yticklabels(y_labels)

    ax.tick_params(axis='x', which='major', labelsize=14, labelcolor='black', labelrotation=0)

    ax.tick_params(axis='y', which='major', labelsize=22, labelcolor='black', labelrotation=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"cost_comparison_{scenario}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def plot_npc(scenario, scenario_1):

    individual_capex_total = get_industry_capex_data(scenario).sum().sum()
    individual_opex_total = get_industry_opex_data(scenario).sum().sum()
    individual_carrier_costs_total = get_industry_carrier_costs_data(scenario).sum().sum()

    total_capex_total = get_total_industry_capex_data(scenario_1).sum()
    total_opex_total = get_total_industry_opex_data(scenario_1).sum()
    total_carrier_costs_total = get_total_industry_cost_carrier_data(scenario_1).sum().sum()

    categories = ['carrier costs', 'CAPEX', 'OPEX']
    colors = ['#A9A9A9', 'dimgray', '#505050']
    industrial_totals = np.array([individual_carrier_costs_total, individual_capex_total, individual_opex_total])
    total_totals = np.array([total_carrier_costs_total, total_capex_total, total_opex_total])

    fig, ax = plt.subplots(figsize=(9, 10))

    bar_width = 0.20
    bar_spacing = 0.07
    bar_positions = np.array([0, bar_width + bar_spacing])

    bottom_industrial = 0
    for i, color in enumerate(colors):
        ax.bar(bar_positions[0], industrial_totals[i], bottom=bottom_industrial, color=color, width=bar_width,
               label=categories[i])
        bottom_industrial += industrial_totals[i]

    bottom_total = 0
    for i, color in enumerate(colors):
        ax.bar(bar_positions[1], total_totals[i], bottom=bottom_total, color=color, width=bar_width)
        bottom_total += total_totals[i]

    y_labels = [f"{int(label) / 1000:.2f}" for label in ax.get_yticks()]
    ax.set_yticklabels(y_labels, fontsize=22)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(['individual', 'integrated'], fontsize=22)
    ax.set_ylabel('Totex 2024 - 2050 [B€]', fontsize=22)
    #ax.set_title('Cost Comparison',  fontname='Times New Roman')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False, prop={'size': 22})

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_cost_comp_npc(scenario, scenario_1, scenario_2, scenario_3):

    total_capex_total = get_total_industry_capex_data(scenario).sum()
    total_opex_total = get_total_industry_opex_data(scenario).sum()
    total_carrier_costs_total = get_total_industry_cost_carrier_data(scenario).sum().sum()

    total_capex_total_1 = get_total_industry_capex_data(scenario_1).sum()
    total_opex_total_1 = get_total_industry_opex_data(scenario_1).sum()
    total_carrier_costs_total_1 = get_total_industry_cost_carrier_data(scenario_1).sum().sum()

    total_capex_total_2 = get_total_industry_capex_data(scenario_2).sum()
    total_opex_total_2 = get_total_industry_opex_data(scenario_2).sum()
    total_carrier_costs_total_2 = get_total_industry_cost_carrier_data(scenario_2).sum().sum()

    total_capex_total_3 = get_total_industry_capex_data(scenario_3).sum()
    total_opex_total_3 = get_total_industry_opex_data(scenario_3).sum()
    total_carrier_costs_total_3 = get_total_industry_cost_carrier_data(scenario_3).sum().sum()

    categories = ['CAPEX', 'OPEX', 'Carrier Costs']
    colors = ['dimgray', 'slategray', 'lightsteelblue']
    scenario_data = [
        [get_total_industry_capex_data(scenario).sum(), get_total_industry_opex_data(scenario).sum(),
         get_total_industry_cost_carrier_data(scenario).sum().sum()],
        [get_total_industry_capex_data(scenario_1).sum(), get_total_industry_opex_data(scenario_1).sum(),
         get_total_industry_cost_carrier_data(scenario_1).sum().sum()],
        [get_total_industry_capex_data(scenario_2).sum(), get_total_industry_opex_data(scenario_2).sum(),
         get_total_industry_cost_carrier_data(scenario_2).sum().sum()],
        [get_total_industry_capex_data(scenario_3).sum(), get_total_industry_opex_data(scenario_3).sum(),
         get_total_industry_cost_carrier_data(scenario_3).sum().sum()]
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.2
    n_scenarios = len(scenario_data)
    indices = np.arange(n_scenarios)

    for scenario_index, scenario_costs in enumerate(scenario_data):
        for category_index, cost in enumerate(scenario_costs):
            bar = ax.bar(scenario_index + bar_width * category_index, cost, bar_width, color=colors[category_index],
                         label=categories[category_index] if scenario_index == 0 else "")
            ax.text(scenario_index + bar_width * category_index, cost, f'{cost:.2f}', ha='center', va='bottom')

    ax.set_xlabel('Szenarien')
    ax.set_ylabel('Kosten')
    ax.set_title('Kostenvergleich der Szenarien')
    ax.set_xticks(indices + bar_width / 2 * (len(categories) - 1))
    ax.set_xticklabels(['Szenario 1', 'Szenario 2', 'Szenario 3', 'Szenario 4'])

    if n_scenarios == 1:
        ax.legend(categories, loc='upper right')

    plt.tight_layout()
    plt.grid(True)
    plt.show()
def draw_transport_arrows_carbon(csv_path, shapefile_path, year, scenario, figsize=(20, 20)):
    df = pd.read_csv(csv_path)
    df = df[df['Unnamed: 0'] == scenario]
    df = df[df['technology'] == 'carbon_pipeline']
    nuts_gdf = gpd.read_file(shapefile_path)
    nuts_gdf = nuts_gdf.to_crs('EPSG:3035')
    countries_to_exclude = ['UK', 'CH', 'IS', 'TR']
    nuts2_gdf = nuts_gdf[(nuts_gdf['LEVL_CODE'] == 2) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]
    nuts2_gdf['centroid'] = nuts2_gdf.geometry.centroid
    centroid_dict = nuts2_gdf.set_index('NUTS_ID')['centroid'].to_dict()

    background_gdf = nuts_gdf[(nuts_gdf['LEVL_CODE'] == 0) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]


    df[['source', 'target']] = df['edge'].str.split('-', expand=True)

    fig, ax = plt.subplots(figsize=figsize)
    background_gdf.plot(ax=ax, color='lightgrey', edgecolor='darkgray')
    background_gdf[background_gdf['CNTR_CODE'] == 'NO'].plot(ax=ax, color='darkgray', edgecolor='darkgray')
    nuts_gdf.plot(ax=ax, color='none', edgecolor='none')


    for _, row in df.iterrows():
        source = row['source']
        target = row['target']
        amount = row[year]
        tech = row['technology']

        color = 'forestgreen' if tech == 'biomethane_transport' else 'black' if tech =='carbon_pipeline' else 'red' if tech == 'hydrogen_pipeline' else 'olive' if tech == 'dry_biomass_truck' else 'purple'
        linewidth = np.sqrt(amount) * 0.03

        if source in centroid_dict and target in centroid_dict:
            source_point = centroid_dict[source]
            target_point = centroid_dict[target]

            ax.annotate('', xy=(target_point.x, target_point.y), xytext=(source_point.x, source_point.y),
                        arrowprops=dict(arrowstyle='->,head_width=0.2,head_length=0.4', color=color, lw=linewidth))

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def draw_transport_and_capture(csv_path, csv_path_capture, shapefile_path, year, scenario, figsize=(20, 20)):
    df_transport = pd.read_csv(csv_path)
    df_transport = df_transport[df_transport['technology'] == 'carbon_pipeline']

    nuts_gdf = gpd.read_file(shapefile_path)
    nuts_gdf = nuts_gdf.to_crs('EPSG:3035')
    countries_to_exclude = ['IS', 'TR']
    nuts2_gdf = nuts_gdf[(nuts_gdf['LEVL_CODE'] == 2) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]
    nuts2_gdf['centroid'] = nuts2_gdf.geometry.centroid
    centroid_dict = nuts2_gdf.set_index('NUTS_ID')['centroid'].to_dict()

    level = [0]
    countries_to_exclude = ['IS', 'TR']
    nuts_gdf_filtered = nuts_gdf[
        (nuts_gdf['LEVL_CODE'].isin(level)) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]

    df_capture = pd.read_csv(csv_path_capture)
    capture_technologies = [
        "BF_BOF_CCS", "gasification_CCS", "SMR_CCS",
        "cement_plant_oxy_combustion", "cement_plant_post_comb", "DAC"
    ]
    relevant_carriers = ["carbon", "carbon_liquid"]
    df_capture = df_capture[
        df_capture['technology'].isin(capture_technologies) &
        df_capture['carrier'].isin(relevant_carriers)
    ]
    df_capture_sum = df_capture.groupby('node')[str(year)].sum().reset_index(name='Total_Carbon_Captured')
    nuts2_with_capture = nuts2_gdf.merge(df_capture_sum, left_on='NUTS_ID', right_on='node', how='left')
    nuts2_with_capture['Total_Carbon_Captured'].fillna(0, inplace=True)

    df_transport[['source', 'target']] = df_transport['edge'].str.split('-', expand=True)

    fig, ax = plt.subplots(figsize=figsize)

    vmin = nuts2_with_capture['Total_Carbon_Captured'].min()
    #vmax = nuts2_with_capture['Total_Carbon_Captured'].max()
    vmax = 3000
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    nuts2_with_capture.plot(column='Total_Carbon_Captured', ax=ax, cmap='Blues', norm=sm.norm,
                            edgecolor='lightgray', linewidth=0.8)

    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.4, aspect=30)

    level = [0]
    nuts_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'].isin(level)]
    border_color = 'dimgray'

    nuts_gdf_filtered.plot(ax=ax, color="None", edgecolor=border_color)

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    for country_code in ['UK', 'CH']:
        specific_gdf = nuts_gdf[nuts_gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\\\\\\\\\", edgecolor='dimgrey', linewidth=0.8)


    norway = nuts_gdf[nuts_gdf['CNTR_CODE'] == 'NO']
    norway.plot(ax=ax, color='lightgrey', edgecolor=border_color)

    def format_by_thousand(x, pos):
        return '{}'.format(x / 1000)

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_by_thousand))
    cbar.ax.tick_params(labelsize=14)
    #cbar.set_label('Total Carbon Captured [Mt]', fontsize=14)
    cbar.set_label('Annual carbon capture rates [Mt]', fontsize=22)

    df_transport = df_transport[(df_transport['time_operation'] == int(year)) & (df_transport['flow_transport'] > 0)]
    # Draw transport arrows
    for _, row in df_transport.iterrows():
        source = row['source']
        target = row['target']
        amount = df_transport[(df_transport['time_operation'] == int(year)) & (df_transport['source'] == source) & (
                    df_transport['target'] == target)]['flow_transport'].sum()
        color = 'red'
        if amount > 0:
            linewidth = np.sqrt(amount) * 5
            if source in centroid_dict and target in centroid_dict:
                source_point = centroid_dict[source]
                target_point = centroid_dict[target]

                ax.annotate('', xy=(target_point.x, target_point.y), xytext=(source_point.x, source_point.y),
                            arrowprops=dict(arrowstyle='->, head_width=0.1, head_length=0.3', color=color,
                                            lw=linewidth))

    ax.set_xlim([-2000, 6200000])
    ax.set_ylim([-2000, 5500000])

    plt.axis('off')
    plt.tight_layout
    plt.savefig(f"carbon_transport_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def draw_hydrogen_pipelines(csv_path, csv_path_hydrogen, shapefile_path, year, scenario, figsize=(20, 20)):
    df_transport = pd.read_csv(csv_path)
    df_transport = df_transport[df_transport['technology'] == 'hydrogen_pipeline']

    nuts_gdf = gpd.read_file(shapefile_path)
    nuts_gdf = nuts_gdf.to_crs('EPSG:3035')
    countries_to_exclude = ['IS', 'TR']
    nuts2_gdf = nuts_gdf[(nuts_gdf['LEVL_CODE'] == 2) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]
    nuts2_gdf['centroid'] = nuts2_gdf.geometry.centroid
    centroid_dict = nuts2_gdf.set_index('NUTS_ID')['centroid'].to_dict()

    level = [0]
    countries_to_exclude = ['IS', 'TR']
    nuts_gdf_filtered = nuts_gdf[
        (nuts_gdf['LEVL_CODE'].isin(level)) & (~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude))]

    df_hydrogen = pd.read_csv(csv_path_capture)
    capture_technologies = [
        "SMR", "SMR_CCS", "gasification",
        "gasification_CCS", "electrolysis"
    ]
    relevant_carriers = ["hydrogen"]
    df_hydrogen = df_hydrogen[
        df_hydrogen['technology'].isin(capture_technologies) &
        df_hydrogen['carrier'].isin(relevant_carriers)
    ]
    df_capture_sum = df_hydrogen.groupby('node')[str(year)].sum().reset_index(name='Total_Carbon_Captured')
    nuts2_with_capture = nuts2_gdf.merge(df_capture_sum, left_on='NUTS_ID', right_on='node', how='left')
    nuts2_with_capture['Total_Carbon_Captured'].fillna(0, inplace=True)

    df_transport[['source', 'target']] = df_transport['edge'].str.split('-', expand=True)

    fig, ax = plt.subplots(figsize=figsize)

    vmin = nuts2_with_capture['Total_Carbon_Captured'].min()
    #vmax = nuts2_with_capture['Total_Carbon_Captured'].max()
    vmax = 10000
    sm = plt.cm.ScalarMappable(cmap='YlOrBr', norm=Normalize(vmin=vmin, vmax=vmax))
    sm._A = []

    nuts2_with_capture.plot(column='Total_Carbon_Captured', ax=ax, cmap='YlOrBr', norm=sm.norm,
                            edgecolor='lightgray', linewidth=0.8)

    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.4, aspect=30)

    level = [0]
    nuts_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'].isin(level)]
    border_color = 'dimgray'
    country_color = 'darkgrey'

    nuts_gdf_filtered.plot(ax=ax, color="None", edgecolor=border_color)

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    for country_code in ['NO', 'UK', 'CH']:
        specific_gdf = nuts_gdf[nuts_gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\\\\\\\\\", edgecolor='dimgrey', linewidth=0.8)

    def format_by_thousand(x, pos):
        return '{}'.format(x / 1000)

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(format_by_thousand))
    cbar.ax.tick_params(labelsize=14)

    cbar.set_label('Yearly hydrogen production [TWh]', fontsize=22)

    df_transport = df_transport[(df_transport['time_operation'] == int(year)) & (df_transport['flow_transport'] > 0)]
    # Draw transport arrows
    for _, row in df_transport.iterrows():
        source = row['source']
        target = row['target']
        amount = df_transport[(df_transport['time_operation'] == int(year)) & (df_transport['source'] == source) & (
                    df_transport['target'] == target)]['flow_transport'].sum()
        color = 'blue'
        if amount > 0:
            linewidth = np.sqrt(amount) * 5
            if source in centroid_dict and target in centroid_dict:
                source_point = centroid_dict[source]
                target_point = centroid_dict[target]

                ax.annotate('', xy=(target_point.x, target_point.y), xytext=(source_point.x, source_point.y),
                            arrowprops=dict(arrowstyle='->, head_width=0.1, head_length=0.3', color=color,
                                            lw=linewidth))

    ax.set_xlim([-2000, 6200000])
    ax.set_ylim([-2000, 5500000])

    plt.axis('off')
    plt.tight_layout
    plt.savefig(f"hydrogen_transport_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

def draw_transport_arrows_and_biomass_usage(csv_path, shapefile_path, year, scenario, figsize=(20, 20)):

    df = pd.read_csv(csv_path)
    transport_technologies = ['biomethane_transport', 'dry_biomass_truck']
    df = df[df['technology'].isin(transport_technologies)]


    availability_dry_biomass = res_scenario.get_total("availability_import", scenario= "scenario_").xs("dry_biomass", level = 'carrier')


    availability_wet_biomass = res_scenario.get_total("availability_import", scenario="scenario_").xs("wet_biomass",
                                                                                                      level='carrier')



    import_dry_biomass = res_scenario.get_total("flow_import", scenario = scenario).xs("dry_biomass", level='carrier')

    import_wet_biomass = res_scenario.get_total("flow_import", scenario=scenario).xs("wet_biomass", level='carrier')

    nuts_gdf = gpd.read_file(shapefile_path)
    nuts_gdf = nuts_gdf.to_crs('EPSG:3035')
    countries_to_exclude = ['IS', 'TR']

    nuts_gdf = nuts_gdf[~nuts_gdf['CNTR_CODE'].isin(countries_to_exclude)]
    nuts2_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'] == 2]
    nuts2_gdf['centroid'] = nuts2_gdf.geometry.centroid
    centroid_dict = nuts2_gdf.set_index('NUTS_ID')['centroid'].to_dict()

    df[['source', 'target']] = df['edge'].str.split('-', expand=True)

    combined_dry_biomass = pd.merge(availability_dry_biomass[year], import_dry_biomass[year], left_on='node', right_on='node', how='outer')
    combined_dry_biomass['utilization'] = np.where(combined_dry_biomass[f'{year}_x'] > 0,
                                                      combined_dry_biomass[f'{year}_y'] / combined_dry_biomass[
                                                          f'{year}_x'], 0)

    combined_wet_biomass = pd.merge(availability_wet_biomass[year], import_wet_biomass[year], left_on='node', right_on='node', how='outer')
    combined_wet_biomass['utilization'] = np.where(combined_wet_biomass[f'{year}_x'] > 0,
                                                   combined_wet_biomass[f'{year}_y'] / combined_wet_biomass[
                                                       f'{year}_x'], 0)




    combined_biomass = pd.merge(combined_dry_biomass, combined_wet_biomass, left_on='node', right_on='node', how='outer')
    combined_biomass = combined_biomass.fillna(0)

    combined_biomass['utilization'] = (combined_biomass['utilization_x'] + combined_biomass['utilization_y']) / 2

    nuts2_gdf = nuts2_gdf.merge(combined_biomass, left_on='NUTS_ID', right_on='node', how='left').fillna(0)


    fig, ax = plt.subplots(figsize=figsize)
    #nuts2_gdf.loc["NO09", 'utilization'] = 2
    vmin = nuts2_gdf['utilization'].min()
    vmax = 1.4
    norm = Normalize(vmin=vmin, vmax=vmax)
    nuts2_gdf.plot(column='utilization', ax=ax, cmap='Purples', edgecolor='lightgray', vmax=vmax, norm=norm)


    sm = plt.cm.ScalarMappable(cmap='Purples', norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.4, aspect=30)
    cbar.ax.set_ylim(0,1)
    level = [0]
    nuts_gdf = nuts_gdf[nuts_gdf['LEVL_CODE'].isin(level)]
    border_color = 'dimgray'
    country_color = 'darkgrey'

    nuts_gdf.plot(ax=ax, color="None", edgecolor=border_color)

    plt.rcParams['hatch.color'] = 'grey'
    plt.rcParams['hatch.linewidth'] = 0.4

    for country_code in ['NO', 'UK', 'CH']:
        specific_gdf = nuts_gdf[nuts_gdf['CNTR_CODE'] == country_code]
        specific_gdf.plot(ax=ax, facecolor='lightgrey', hatch="\\\\\\\\\\\\", edgecolor='dimgrey', linewidth=0.8)

    cbar.set_label('Usage of biomass potential', fontsize=24)

    cbar.ax.tick_params(labelsize=22)

    fig.subplots_adjust(left=0.05, right=0.75)

    df = df[(df['time_operation'] == int(year)) & (df['flow_transport'] > 0)]

    for _, row in df.iterrows():
        source = row['source']
        target = row['target']
        amount = df[(df['time_operation'] == int(year)) & (df['source'] == source) & (df['target'] == target)][
            'flow_transport'].sum()
        tech = row['technology']

        color = 'coral' if tech == 'biomethane_transport' else 'crimson' if tech == 'dry_biomass_truck' else 'purple'

        if pd.notnull(amount) and amount > 0:
            linewidth = np.sqrt(amount) * 5
            if source in centroid_dict and target in centroid_dict:
                source_point = centroid_dict[source]
                target_point = centroid_dict[target]

                ax.annotate('', xy=(target_point.x, target_point.y), xytext=(source_point.x, source_point.y),
                            arrowprops=dict(arrowstyle='->, head_width=0.2, head_length=0.4', color=color, lw=linewidth))

    ax.set_xlim([50000, 6300000])
    ax.set_ylim([50000, 5500000])
    print(nuts2_gdf.total_bounds)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"biomass_transport_map_{scenario}_{year}.png", dpi=300, bbox_inches='tight', pad_inches=0)

    plt.show()

df_transport_scenarios_140324 = res_scenario.get_df("flow_transport", scenario='scenario_')
df_transport_scenarios_140324.to_csv("scenarios_no_pipeline_200324/transport_flow_.csv")

#carbon_storage = pd.read_csv("capacity_limit_carbon_storage.csv")
#sum = carbon_storage['capacity_limit'].sum()*8760
#print(sum)



def calc_lco(scenario, discount_rate, carrier):
    capex_df = res_scenario.get_total("cost_capex_total", scenario)
    opex_df = res_scenario.get_total("cost_opex_total", scenario)
    carrier_cost_df = res_scenario.get_total("cost_carrier")
    production_df = pd.read_csv(f"{folder_path}/flow_conversion_output_grouped_{scenario}.csv")
    #capex_df = pd.read_csv(f'{folder_path}/capex.csv', index_col=0)
    #opex_df = pd.read_csv(f'{folder_path}/opex.csv', index_col=0)
    #carrier_cost_df = pd.read_csv(f'{folder_path}/cost-carrier.csv')
    #production_df = pd.read_csv(f'{folder_path}/production.csv')
    production_df = production_df[production_df['carrier'] == carrier]
    print(production_df)

    capex_df = capex_df.loc[[scenario]]
    opex_df = opex_df.loc[[scenario]]
    carrier_cost_df = carrier_cost_df.loc[[scenario]]
    carrier_cost_df = carrier_cost_df.sum()
    print(carrier_cost_df)


    total_discounted_costs = 0
    total_discounted_production = 0

    for year in capex_df.columns:
        discount_factor = (1 + discount_rate) ** int(year)
        discounted_costs = (capex_df[year].iloc[0] +
                            opex_df[year].iloc[0] +
                            carrier_cost_df[year]) / discount_factor
        discounted_costs = discounted_costs * 1000

        discounted_production = production_df[str(year)].sum() / discount_factor

        total_discounted_costs += discounted_costs
        total_discounted_production += discounted_production
        print(total_discounted_costs)
        print(total_discounted_production)

    lcoa = total_discounted_costs / total_discounted_production if total_discounted_production else float('inf')

    print(lcoa)

    return lcoa

def plot_demand(scenario):
    industries = ['ammonia', 'steel', 'cement', 'methanol', 'oil_products']
    demand_data = {}
    custom_colors = ['mediumpurple', 'steelblue', 'darkslateblue', 'fuchsia', 'firebrick']
    years = np.arange(2024, 2051, 2)  # Jahre von 2024 bis 2050
    demand_df = pd.DataFrame(index=years)

    for industry in industries:
        demand = res_scenario.get_total("demand", scenario=scenario).xs(industry, level='carrier').sum()
        demand_df[industry] = demand.values  # Füge die Werte zum DataFrame hinzu

    plt.figure(figsize=(12, 7))
    stacks = plt.stackplot(demand_df.index, demand_df.T, labels=demand_df.columns, colors=custom_colors, alpha=0.9)

    # Linien entlang der oberen Kanten jeder Fläche zeichnen
    cum_values = np.zeros(len(years))
    for i, col in enumerate(demand_df.columns):
        cum_values += demand_df[col].values
        plt.plot(years, cum_values, color='dimgray', lw=2)
    #plt.legend(loc='upper right', frameon=False)
    #plt.title(f'Gestapeltes Flächendiagramm der Nachfrage nach Industrie ({scenario})')
    plt.xlabel('Year')
    plt.ylabel('Demand [Mt]')

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    def format_by_thousand(x, pos):
        return '{}'.format(int(x / 1000))

    ax.yaxis.set_major_formatter(FuncFormatter(format_by_thousand))

    plt.show()




if __name__ == '__main__':

    # save (grouped) inputs and outputs as csv
    folder_path = ('scenarios_140324')
    scenarios = ['scenario_',
                 #'scenario_electrification',
                 #'scenario_hydrogen',
                 #'scenario_biomass',
                 #'scenario_CCS',
                 #'scenario_high_demand',
                 #'scenario_low_demand',
                 #'scenario_biomass_high_price_2',
                 #'scenario_biomass_high_price_3',
                 #'scenario_biomass_high_price_5',
                 #'scenario_biomass_high_price_7',
                 #'scenario_biomass_high_price_10',
                 ]
    carriers = [#'ammonia',
                #'steel',
                #'cement',
                #'methanol',
                #'oil_products',
                #'direct_reduced_iron',
                #'electricity',
                #'natural_gas',
                #'carbon',
                #'coal_for_cement',
                #'biomethane',
                'hydrogen',
                #'carbon_methanol',
                #'carbon_liquid'
                ]

    # get emissions
    #for scenario in scenarios:
        #for carrier in carriers:
             # get_emissions(folder_path, scenario)

    #for scenario in scenarios:
       # save_total(folder_path, scenario)

    # save (grouped) capacity as csv
    #for scenario in scenarios:
        #(folder_path, scenario)

    # save imports and exports

    #for scenario in scenarios:
        #save_imports_exports(folder_path, scenario)

    #for scenario in scenarios:
       # energy_carrier(folder_path, scenario)

    #scenario = "scenario_"
    #plot_emissions(scenario)

    years = ['0', '8', '13']
    for scenario in scenarios:
        for year in years:

            shapefile_path = "nuts_data/NUTS_RG_20M_2021_4326.shp"
            #year = '13'
            csv_path = f"{folder_path}/energy_carrier_country_{scenario}.csv"
            #draw_wedges_on_map(csv_path, shapefile_path, year, radius_factor=0.005, scenario=scenario)

    csv_path_base = f"{folder_path}/energy_carrier_country_scenario_.csv"
    sc_carriers = [#'_biomass',
                   #'_electrification', '_CCS',
                   #'_hydrogen',
                    '_biomass_high_price_2', '_biomass_high_price_3', '_biomass_high_price_5', '_biomass_high_price_7', '_biomass_high_price_10',
                    ]
    for sc in sc_carriers:
        csv_path_scenario = f"{folder_path}/energy_carrier_country_scenario{sc}.csv"
        #plot_carrier_diff(csv_path_base, csv_path_scenario, sc, y_limits=(-3000, 3000))

    # generate sankey diagram
    target_technologies = ['BF_BOF',
                           'BF_BOF_CCS',
                           'EAF',
                           'carbon_liquefication',
                           'carbon_storage',
                           'cement_plant',
                            'cement_plant_oxy_combustion', 'cement_plant_post_comb',
                           'e_haber_bosch', 'haber_bosch',
                           'gasification_methanol', 'methanol_from_hydrogen', 'methanol_synthesis',
                           'refinery',
                           'gasification_methanol_h2',
                           'biomethane_SMR_methanol', 'biomethane_SMR', 'biomethane_SMR_CCS', 'biomethane_haber_bosch'
                            ]
    intermediate_technologies = [#'anaerobic_digestion',
                                 'biomethane_conversion', 'biomethane_haber_bosch',
                                 'ASU',
                                 'biomass_to_coal_conversion', 'hydrogen_for_cement_conversion',
                                 'DAC',
                                 'DRI', 'h2_to_ng', 'scrap_conversion_EAF', 'scrap_conversion_BF_BOF',
                                 'SMR', 'SMR_CCS', 'gasification', 'gasification_CCS',
                                 'electrolysis',
                                 'biomethane_SMR', 'biomethane_SMR_CCS',
                                 'biomethane_DRI',
                                 'biomethane_methanol_synthesis',
                                 'carbon_conversion', 'carbon_methanol_conversion',
                                 'SMR_methanol', 'gasification_methanol_h2'
                                 #'photovoltaics', 'pv_ground', 'pv_rooftop', 'wind_offshore', 'wind_onshore',
                                 'carbon_liquefication', 'carbon_removal',
                                 'carbon_storage',
                                 'carbon_evaporation'
                                ]
    years = ['0',
            # '3',
             #'8',
             '13'
             ]
    #for year in years:
     #   for scenario in scenarios:
      #        generate_sankey_diagram(folder_path, scenario, target_technologies, intermediate_technologies, year, title="Process depiction in", save_file=False)

    # generate bar charts for industry outputs

    #for scenario in scenarios:
     #   for carrier in carriers:
      #      plot_outputs(folder_path, scenario, carrier, save_file=True)

    scenario = 'scenario_'
    scenario2 = 'scenario_biomass_high_price_10'
    #plot_outputs_carbon(scenario, scenario2)

    # plot capacities for chosen technologies
    #technologies = ['haber_bosch', 'e_haber_bosch']
    #for technology in technologies:
        #for scenario in scenarios:
            #plot_capacities(folder_path, scenario, technology, save_file=False)

    #shapefile_path = "nuts_data/NUTS_RG_20M_2021_4326.shp"
    #legend_name = "legend_name"
    #create_countries_map_from_shapefile(shapefile_path, save_path=None)

    for scenario in scenarios:
        #df = pd.read_csv('output_3.csv')
        df = pd.read_csv(f'{folder_path}/flow_conversion_output_{scenario}.csv')
        df_input = pd.read_csv(f'{folder_path}/flow_conversion_input_{scenario}.csv')

        #plot_dataframe_on_map(df, df_input, 'technology', ['0', #'1', '2',
                                                 #'3', #'4', '5', '6', '7',
                  #                               '8', #'9', '10', '11', '12',
                   #                              '13'],
                   #           'nuts_data/NUTS_RG_20M_2021_4326.shp', save_png=True)

    scenario_1 = "scenario_"
    scenario = "scenario_"
    #for scenario in scenarios:
    plot_npc(scenario, scenario_1)
    #plot_costs_with_unique_colors(scenario)


    scenario_1 = "scenario_CCS"
    scenario_2 = "scenario_hydrogen"
    scenario_3 = "scenario_electrification"
    #plot_cost_comp_npc(scenario, scenario_1, scenario_2, scenario_3)

    scenario = "scenario_"
    csv_path = "scenarios_140324/transport_flow.csv"
    shapefile_path = "nuts_data/NUTS_RG_20M_2021_4326.shp"
    csv_path_capture = "scenarios_140324/flow_conversion_output_scenario_.csv"
    csv_path_hydrogen = "scenarios_140324/flow_conversion_output_scenario_.csv"
    years = [13]

    for year in years:
        #draw_transport_and_capture(csv_path, csv_path_capture, shapefile_path, year, scenario, figsize=(20, 20))
        #draw_hydrogen_pipelines(csv_path, csv_path_hydrogen, shapefile_path, year, scenario, figsize=(20, 20))
        draw_transport_arrows_and_biomass_usage(csv_path, shapefile_path, year, scenario, figsize=(20, 20))


    #calc_lco(scenario = "scenario_", discount_rate = 0.06, carrier="methanol")

    #for scenario in scenarios:
        #plot_demand(scenario)
>>>>>>> Stashed changes
