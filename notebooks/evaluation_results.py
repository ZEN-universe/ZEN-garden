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

