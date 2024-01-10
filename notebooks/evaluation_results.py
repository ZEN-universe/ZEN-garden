import pandas as pd
from pprint import pprint
from plotly.graph_objects import Sankey
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from zen_garden.postprocess.results import Results
import seaborn as sns

res_scenario = Results("../outputs/hard_to_abate/")

df = res_scenario.get_df('flow_conversion_input', scenario="scenario_")
pprint(df)
print(df.index)

df_import = res_scenario.get_df("flow_import", scenario="scenario_")
pprint(df_import)

#res_scenario.standard_plots()
res_scenario.plot("flow_conversion_input", reference_carrier="steel", yearly=True)

def plot_technology_bargraph(df, selected_technology):
    filtered_df = df.xs((selected_technology, 0), level=('technology', 'time_operation')).reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='node', y='flow_conversion_input', hue='carrier', data=filtered_df)
    plt.title(f'Bar chart input conversion flow in 2024')
    plt.xlabel('Node')
    plt.ylabel('Value')
    plt.legend(title='Carrier', bbox_to_anchor=(1, 1))
    plt.show()

def plot_selected_technologies_over_nodes(df, selected_technologies):
    filtered_df = df[df.index.get_level_values('technology').isin(selected_technologies)].reset_index()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='node', y='flow_conversion_input', hue='carrier', data=filtered_df)
    plt.title(f'Balkendiagramm für ausgewählte Technologien bei time_operation = 0')
    plt.xlabel('Node')
    plt.ylabel('Wert')
    plt.legend(title='Carrier', bbox_to_anchor=(1, 1))
    plt.show()

def plot_aggregated_values(data, selected_technology_groups):
    grouped_data = data.groupby(level=['technology']).sum()

    selected_data = grouped_data[grouped_data.index.isin(selected_technology_groups)]

    fig, ax = plt.subplots(figsize=(10, 6))
    selected_data.plot(kind='bar', ax=ax, legend=False)

    plt.title('Aggregierte Werte für ausgewählte Technologiegruppen')
    plt.xlabel('Technologie')
    plt.ylabel('Summierte Werte')

    plt.show()

def create_sankey_diagram(df, value_column='value'):
    if value_column not in df.columns:
        raise KeyError(f"Column not found: {value_column}")

    df_long = df.reset_index()

    nodes = pd.Index(df_long['technology'].append(df_long['carrier']).unique())
    link_sources = nodes.get_indexer(df_long['technology'])
    link_targets = nodes.get_indexer(df_long['carrier'])
    values = df_long[value_column]

    sankey_fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color='black', width=0.5),
            label=nodes
        ),
        link=dict(
            source=link_sources,
            target=link_targets,
            value=values
        )
    ))

    # Update layout
    sankey_fig.update_layout(title_text="Sankey Diagram",
                             font_size=10)

    # Show the diagram
    sankey_fig.show()



if __name__ == '__main__':

    #plot_technology_bargraph(df, 'BF_BOF')

    selected_technologies = ['ASU', 'scrap_conversion_EAF']

    steel_conversion_technologies = ["BF_BOF", "DRI", "EAF", "BF_BOF_CCS"]
    cement_conversion_technologies = ["cement_plant", "cement_plant_oxy_combustion", "cement_plant_pcc_coal", "cement_plant_pcc_ng",]
    #plot_selected_technologies_over_nodes(df, steel_conversion_technologies)

    #technology_groups = steel_conversion_technologies + cement_conversion_technologies

    #plot_stacked_technology_groups(df, technology_groups)

    #plot_aggregated_values(df, technology_groups)
    #create_sankey_diagram(df, value_column='flow_conversion_input')

    #plot_stacked_steel_technologies(df)
