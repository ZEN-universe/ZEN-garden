"""
:Title:        ZEN-GARDEN plot_results
:Created:      March-2024
:Authors:      Jara Spate (jspaete@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Used for plotting the results of the optimization model.
"""
import os
import pandas as pd
import numpy as np
import pint
import math
from bokeh.palettes import Category10
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from matplotlib.gridspec import GridSpec

import seaborn as sns
from matplotlib.colors import Normalize
from zen_garden.postprocess.results.folder_structur_JS import create_folder, get_folder_path
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def plots_cost(res, directory, save_fig=True, file_type='png'):
    """
    Plot the costs of the energy system.

    :param res: The results object containing energy system data.
    :param directory: The directory where the plots will be saved.
    :param save_fig: Boolean indicating if the figure should be saved.
    :param file_type: The file type of the figure to be saved (default is 'png').
    :return: None
    """
    # Plotting individual cost component
    res.plot("cost_capex_total", yearly=False,
                   plot_strings={"title": "Total Capex", "ylabel": "Capex"},
                   save_fig=save_fig, file_type=file_type)
    res.plot("cost_opex_total", yearly=False,
                   plot_strings={"title": "Total Opex", "ylabel": "Opex"},
                   save_fig=save_fig, file_type=file_type)
    res.plot("cost_carrier", yearly=False,
                   plot_strings={"title": "Carrier Cost", "ylabel": "Cost"},
                   save_fig=save_fig, file_type=file_type)
    res.plot("cost_carrier_total", yearly=False,
                   plot_strings={"title": "Carrier Cost", "ylabel": "Cost"},
                   save_fig=save_fig, file_type=file_type)

    # Get the scenario from the results
    scenario = res.scenarios[0]

    # List of cost components to be plotted
    costs = ["cost_capex_total", "cost_opex_total", "cost_carrier_total"]
    total_costs = pd.DataFrame()

    # Collecting cost data into a DataFrame
    for cost in costs:
        # Assuming _get_component_data returns a DataFrame
        component_name, total = res._get_component_data(cost, scenario=scenario)
        df_total = total.to_frame()
        total_costs = pd.concat([total_costs, df_total], axis=1)

    # Plotting the stacked bar chart of costs
    total_costs.plot(kind='bar', stacked=True)
    plt.xlabel('Year')
    plt.ylabel('Cost')
    plt.title('Stacked Bar Chart of Costs Over Years')
    plt.show()

    # Directory and file handling for saving the plot
    folder = 'Figures'
    path_folder = os.path.join(directory, folder)
    if not os.path.exists(path_folder):
        os.makedirs(path_folder)

    filename_save = 'Stacked_Bar_Chart_of_Costs_Over_Years.' + file_type
    path_filename_save = os.path.join(path_folder, filename_save)
    plt.savefig(path_filename_save, bbox_inches='tight', pad_inches=0.1)



def plot_energy_balance_JS2(data_plot, node, carrier, start_hour, directory, scenario, short=False, save_fig=True):
    # Filter DataFrame based on node and carrier
    data_plot = data_plot.reset_index()
    if carrier == 'electricity':
        data_plot = data_plot[(data_plot['node'] == node) & ((data_plot['carrier'] == 'electricity') | (data_plot['carrier'] == 'diesel'))]
        data_plot['label'] = data_plot['variable'] + ', ' + data_plot['technology'] + ', ' + data_plot['carrier']
        data_plot['label'] = data_plot['label'].str.replace(', no_technology','')
    else:
        data_plot = data_plot[(data_plot['node'] == node) & (data_plot['carrier'] == carrier)]
        # Create a combined label and set as index
        data_plot['label'] = data_plot['variable'] + ', ' + data_plot['technology']
        data_plot['label'] = data_plot['label'].str.replace(', no_technology','')

    # change 'el_WP' to 'electric_WP'
    data_plot['label'] = data_plot['label'].str.replace('el_WP','electric_WP')
    data_plot['label'] = data_plot['label'].str.replace('dieselectric_WP','diesel_WP')
    # delete duplicate labels:
    data_plot.drop_duplicates(subset='label', keep='first', inplace=True)

    data_plot.sort_values(by='label', inplace=True)
    data_plot.set_index('label', inplace=True)

    # Adjust values to be negative if label contains 'flow_conversion_out' and 'irrigation_sys'
    data_plot.loc[data_plot.index.str.contains('flow_conversion_input, irrigation_sys'), :] *= -1
    data_plot.loc[data_plot.index.str.contains('flow_storage_charge, water_storage'), :] *= -1
    data_plot.loc[data_plot.index.str.contains('flow_storage_charge, battery'), :] *= -1
    data_plot.loc[data_plot.index.str.contains('flow_conversion_input, electric_WP'), :] *= -1
    data_plot.loc[data_plot.index.str.contains('flow_conversion_input, diesel_WP'), :] *= -1

    # Drop unnecessary columns and filter rows with all zeros
    data_plot = data_plot.drop(columns=['variable', 'node', 'technology', 'carrier'])
    data_plot = data_plot.loc[(data_plot != 0).any(axis=1)]

    # Determine colors for the plot based on the labels in the index
    color_dict = {
        'electric_WP': '#64557B',                   # muted pink
        'irrigation_sys': '#F4D35E',                # muted dark blue
        'diesel_WP': '#F67B45',                     # muted dark orange
        'PV': '#CB7876',                            # muted dark pink/violet
        'flow_storage_charge, water_storage': '#81B2D9',  # muted sky blue
        'flow_storage_discharge, water_storage': '#32769B',  # muted steel blue
        'flow_storage_charge, battery': '#62866C',  # muted dark green
        'flow_storage_discharge, battery': '#B4CFA4',  # muted light green
        'flow_import, electricity': '#E39B99',      # muted light pink
        'flow_import, diesel': '#FF9C5B'            # muted light orange
    }

    if short:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 12), sharex=True, sharey=True)
    else:
        # Prepare the plot layout
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 12), sharex=True, sharey=True)

    # Loop through each month to plot
    for i, month in enumerate(range(1, 13)):
        if short:
            # Continue for all month beside January and July
            if month not in [1, 7]:
                print(f"Skipping month {month}")
                continue
        print(f"Processing month {month}")
        # Calculate start_hour and duration for the current month
        start_hour_month = start_hour + (i * 24*31)  # 720 hours per month
        duration_month = 24  # 720 hours in a month
        # Filter columns for the current month
        data_plot_month = data_plot.iloc[:, start_hour_month:start_hour_month+duration_month]

        # Select the first 24 columns
        filtered_df = data_plot_month.iloc[:, :duration_month]


        # Transpose the DataFrame for plotting
        filtered_df = data_plot_month.T

        # Generate a time index for the plot
        time_index = pd.date_range(start=f'2023-{month:02}-01', periods=duration_month, freq='h')

        # Determine colors for the plot based on the labels in the index
        colors = []
        for label in filtered_df.columns:
            assigned_color = 'grey'  # Default color if no match found
            for key in color_dict:
                if key in label:
                    assigned_color = color_dict[key]
                    break
            colors.append(assigned_color)

        # Plot as stacked area plot with specified colors on respective subplot
        if short:
            if month == 1:
                ax = axes[0]
            if month == 7:
                ax = axes[1]
            else:
                #Raise a warning
                print('Month not in [1, 7]')
        else:
            row = i // 4
            col = i % 4
            ax = axes[row, col]
        filtered_df.plot(kind='area', stacked=True, color=colors, ax=ax)
        # don0t show the legend
        ax.get_legend().remove()

        # Set custom x-axis labels
        ax.set_xticks(range(0, duration_month, 3))
        ax.set_xticklabels([time.strftime('%H') for i, time in enumerate(time_index) if i % 3 == 0], rotation=0, ha='right')

        # Set y-axis label
        ax.set_ylabel('Power [MW]')

        # Set title for each subplot
        ax.set_title(time_index[0].strftime('%B'))

     # Set common x-axis label and adjust layout
    fig.text(0.5, -0.04, 'Hour', ha='center', va='center')
    fig.tight_layout()
    #add overall title
    if carrier == 'electricity':
        fig.suptitle(f'Energy balance for {node}', fontsize=16, y=1.05)
    else:
        fig.suptitle(f'Water balance for {node}', fontsize=16, y=1.05)

    # Add legend to the last subplot
    if short:
        axes[0].legend(loc='upper left')
    else:
        axes[0, 0].legend(loc='upper left')
    if save_fig:
        # Directory and file handling for saving the plot
        folder = 'Figures'
        print(directory)
        path_folder = os.path.join(directory, folder)
        if not os.path.exists(path_folder):
            os.makedirs(path_folder)
        if carrier == 'electricity':
            filename_save = f'Energy_Balance_{node}_{scenario}.png'
        else:
            filename_save = f'Water_Balance_{node}_{scenario}.png'
        path_filename_save = os.path.join(path_folder, filename_save)

        plt.savefig(path_filename_save, bbox_inches='tight', pad_inches=0.1)
    # Show plot
    #plt.show()



def get_short_unit_name(unit_str):
    print(unit_str)
    if 'watt' in unit_str or 'ton' in unit_str:
        unit_str = unit_str.replace('peta', 'p')
        unit_str = unit_str.replace('giga', 'G')
        unit_str = unit_str.replace('tera', 'T')
        unit_str = unit_str.replace('kilo', 'k')
        unit_str = unit_str.replace('mega', 'M')
        unit_str = unit_str.replace('watt', 'W')
        unit_str = unit_str.replace('_hour', 'h')
    elif 'meter ** 3' or 'km**3' in unit_str:
        unit_str = unit_str.replace('meter ** 3', '$\\mathrm{{m^3}}$')
        unit_str = unit_str.replace('km**3', '$\\mathrm{{km^3}}$')
        unit_str = unit_str.replace('kilo', '$\\mathrm{{10^3}}$')
        unit_str = unit_str.replace('mega', '$\\mathrm{{10^6}}$')
    elif 'USD' in unit_str:
        print(unit_str)
        unit_str = unit_str.replace('USD', '$')
        unit_str = unit_str.replace('tera', 'trillion')
        unit_str = unit_str.replace('giga', 'billion')
        unit_str = unit_str.replace('mega', 'million')
        unit_str = unit_str.replace('kilo', 'thousand')
    unit_str = unit_str.replace('ton', 't')
    return unit_str


def get_best_unit(df, column_name, unit_str, show_unit = False, vmax=None):
    # Create a Unit Registry
    ureg = pint.UnitRegistry()

    # Define USD as a unit
    ureg.define('USD = [currency]')

    if vmax is None:
        # if column name is a list
        if isinstance(column_name, list):
            vmax = 0
            for col in column_name:
                if df[col].max() > vmax:
                    vmax = df[col].max()
        else:
            vmax = df[column_name].max()

    # Adjust vmax to the highest power of 10
    if vmax > 0:
        vmax_10 = 10 ** int(math.log10(vmax))
    else:
        vmax_10 = 1  # Set to 1 if vmax is 0 or negative
    star = False
    if '*' in unit_str:
        star = True
        # Find the position of the first occurrence of a letter (unit part)
        unit_pos = next((i for i, c in enumerate(unit_str) if c.isalpha()), None)

        if unit_pos is not None:
            print(unit_pos)
            # Split the string based on the position of the unit part
            input_value_unit = unit_str[:(unit_pos-2)]
            input_value = float(input_value_unit)
            unit = unit_str[unit_pos:]

            # Evaluate the value part as a float
            value = input_value * vmax_10

        else:
            raise ValueError("Invalid unit string format")
    else:
        input_value = 1
        value = vmax_10  # Assume the value is 1 if "*" is not present
        unit = unit_str.strip()

    unit = ureg.parse_expression(unit_str.strip())  # Parse the unit string to handle expressions


    # Convert to the best-fitting unit
    value_best_unit = (value * unit).to_compact()

    # Extract the magnitude and unit
    output_unit = value_best_unit.units
    # Calculate the factor to convert original values to the new unit
    factor = (input_value * unit).to(output_unit)
    output_factor = factor.magnitude
    if output_factor == 1 and star:
        output_unit = unit_str
        #Delet 1* from the unit string
        output_unit = output_unit[2:]
    output_unit_name = get_short_unit_name(str(output_unit))




    df_out = df.copy()
    df_out[column_name] = df_out[column_name] * output_factor
    vmax_output = vmax * output_factor
    if show_unit:
        print(f"Best unit: {output_unit}")
        print(f"Vmax before conversion: {vmax}")
        print(f"Vmax after conversion: {vmax_output}")
        print(f"Factor to convert from {unit_str}: {output_factor}")

    return output_unit_name, df_out, output_factor



def plot_map_data(directory, df, us_gdf, column_name, plot_dict, default_vmax=None):
    """
    Plot the data on the map.

    :param directory: The directory where the map will be saved.
    :param df: The DataFrame containing the data.
    :param us_gdf: The GeoDataFrame containing the US map.
    :param column_name: The column name containing the data.
    :param plot_dict: The dictionary containing the information for the plot.
    :param default_vmax: The default vmax for the colorbar.
    :return: None
    """
    print("\n")
    print(f"Plotting map data for {plot_dict['title']}")
    input_unit = plot_dict['unit']
    output_unit, df_converted, default_vmax = get_best_unit(df, column_name, input_unit, default_vmax)
    # Merge the map and the data
    filename = plot_dict['filename']
    title = plot_dict['title'] + f" [{output_unit}]"
    ylabel = plot_dict['ylabel'] + f" [{output_unit}]"

    df_converted[column_name] = df_converted[column_name].round(1)
    # Set CRS for DataFrame df to EPSG:4326
    df_converted.crs = 'EPSG:4326'
    #us_gdf = us_gdf[us_gdf['state'] == 'CA']

    # Reproject GeoDataFrame us_gdf to match the CRS of DataFrame df
    us_gdf = us_gdf.to_crs(df_converted.crs)


    if 'county_code' in us_gdf.columns:
        df_map = us_gdf.merge(df_converted, on="county_code", how='left')
    else:
        # Merge GeoDataFrame with DataFrame
        df_map = us_gdf.merge(df_converted, on="State_Code", how='left')



    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))

    # Formatting changes to the map
    plt.xticks(rotation=90)  # Rotate the x-labels 90 degrees
    ax.axis('off')  # Remove the frame around the map
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 14

    # Formatting changes to the colorbar
    vmax = df_map[column_name].max() if default_vmax is None else default_vmax

    # Focus around the contiguous US (excluding Alaska and Hawaii)
    ax.set_xlim(-125, -65)
    ax.set_ylim(24, 50)

    # #set the lim around californina
    # ax.set_xlim(-125, -114)
    # ax.set_ylim(32, 42)

    # Define the normalization and ticks for the colorbar
    norm = Normalize(vmin=0, vmax=vmax)
    ticks = np.linspace(0, vmax, 5)

    # Plot the geospatial data with normalization
    df_map.plot(column=column_name, cmap="Reds", linewidth=0.4, ax=ax, edgecolor=".4",
                missing_kwds={"color": "white"}, legend=True, vmin=0, vmax=vmax,
                legend_kwds={'label': ylabel, 'orientation': 'vertical', 'shrink': 0.4,
                             'pad': 0.12, 'ticks': ticks})

    plt.title(title, fontsize=20)

    # Save the plot
    create_folder(directory)  # Assuming this is a defined function
    path_folder = get_folder_path(df, column_name, plot_dict, directory)  # Assuming this is a defined function
    filename_save = f"{filename}.png"
    path_filename_save = os.path.join(path_folder, filename_save)
    plt.savefig(path_filename_save, bbox_inches='tight', pad_inches=0.1)

def plot_capacity_boxplot(df):
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for (tech, cap_type), subset in df.groupby(['technology', 'capacity_type']):
        ax.bar(subset['scenario_name'], subset['capacity'], label=f'{tech} ({cap_type})')

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Capacity')
    ax.set_title('Technology and Capacity Type by Scenario')
    ax.legend()

    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()


def plot_capacity_barplot(df, unit_power):
    unit_output, df_converted, vmax_output = get_best_unit(df, 'capacity', unit_power)
    print(df_converted['capacity'].max())
    max_value = df_converted['capacity'].max()

    # Create a new column combining technology and capacity type
    df_converted['tech_cap'] = df_converted['technology'] + ' (' + df_converted['capacity_type'] + ')'

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Create barplot with seaborn
    sns.barplot(x='scenario_name', y='capacity', hue='tech_cap', data=df_converted, ax=ax1)

    # Set the axis labels and title
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel(f'Capacity (Energy) [{unit_output}h]')
    ax1.set_title('Technology and Capacity Type by Scenario')

    # Set the second y-axis for Power capacity
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'Capacity (Power) [{unit_output}]')

    # Rotate x-axis labels for better readability
    for tick in ax1.get_xticklabels():
        tick.set_rotation(90)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()



def plot_flow_import_boxplot(df, unit):
    unit_output, df_converted, vmax_out = get_best_unit(df, 'flow_import', unit)
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    for carrier, subset in df_converted.groupby(['carrier']):
        ax.bar(subset['scenario_name'], subset['flow_import'], label=carrier[0])  # Extracting carrier name from tuple

    ax.set_xlabel('Scenario')
    ax.set_ylabel('flow_import' + f' [{unit_output}]')
    ax.set_title('Flow import by Scenario and Carrier' + f' [{unit_output}]')

    # Add legend outside the loop
    ax.legend()

    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

def plot_flow_import_boxplots_separate(df, unit):
    print(df['flow_import'].max())
    unit_output, df_converted, vmax_out = get_best_unit(df, 'flow_import', unit)
    print(df['flow_import'].max())
    # Get unique carriers
    unique_carriers = df_converted['carrier'].unique()

    # Plotting
    for carrier in unique_carriers:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Filter DataFrame for the current carrier
        df_carrier = df_converted[df_converted['carrier'] == carrier]

        # Plotting for the current carrier
        ax.bar(df_carrier['scenario_name'], df_carrier['flow_import'])

        ax.set_xlabel('Scenario')
        ax.set_ylabel('flow_import' + f' [{unit_output}]')
        ax.set_title(f'Flow import for {carrier}' + f' [{unit_output}]')

        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.show()

def plot_flow_import_boxplot_v2(df, unit):
    print(df['flow_import'].max())
    unit_output, df_converted, vmax_output = get_best_unit(df, 'flow_import', unit)
    print(df_converted['flow_import'].max())
    max_value = df_converted['flow_import'].max()
    # Reshape the DataFrame to have separate columns for flow_import_diesel and flow_import_el
    df_pivot = df_converted.pivot(index='scenario_name', columns='carrier', values='flow_import').reset_index()

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Define custom colors
    el_color = '#1f77b4'  # Example: blue color
    diesel_color = '#ff7f0e'  # Example: orange color

    # Define width of the bars
    bar_width = 0.35

    # Define positions for the bars
    scenario_positions = np.arange(len(df_pivot['scenario_name']))

    # Plotting for 'flow_import_el' on the left axis
    ax1.bar(scenario_positions - bar_width/2, df_pivot['electricity'], bar_width, color=el_color, label='Electricity')
    ax1.set_xlabel('Scenario')
    ylabel1 = 'Electricity'  + f' [{unit_output}]'
    ax1.set_ylabel(ylabel1, color=el_color)
    ax1.tick_params(axis='y', labelcolor=el_color)
    ax1.set_ylim(0, max_value)

    # Adding legend for left axis
    ax1.legend(loc='upper left')

    # Creating a second y-axis for 'flow_import_diesel' on the right side
    ax2 = ax1.twinx()

    # Plotting for 'flow_import_diesel' on the right axis
    ax2.bar(scenario_positions + bar_width/2, df_pivot['diesel'], bar_width, color=diesel_color, label='Diesel')
    ylabel2 = 'Diesel'  + f' [{unit_output}]'
    ax2.set_ylabel(ylabel2, color=diesel_color)
    ax2.tick_params(axis='y', labelcolor=diesel_color)
    ax2.set_ylim(0, max_value)

    # Adding legend for right axis
    ax2.legend(loc='upper right')

    ax1.set_title('Flow Import by Scenario' + f' [{unit_output}]')
    ax1.set_xticks(scenario_positions)
    ax1.set_xticklabels(df_pivot['scenario_name'], rotation=70)
    plt.tight_layout()
    plt.show()

def plot_co2_cost_boxplots(df):
    # Get unique columns
    unique_columns = ['net_present_cost', 'carbon_emissions_cumulative']

    # Plotting
    for column in unique_columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plotting for the current column
        ax.bar(df['scenario_name'], df[column])

        ax.set_xlabel('Scenario')
        ax.set_ylabel(column)
        ax.set_title(f'{column.capitalize()} by Scenario')

        plt.xticks(rotation=70)
        plt.tight_layout()
        plt.show()


def plot_co2_cost_boxplots_v2(df, unit_co2, unit_cost):
    output_unit_co2, df_converted_co2, vmax_output = get_best_unit(df, 'carbon_emissions_cumulative', unit_co2)
    output_unit_cost, df_converted_cost, vmax_output = get_best_unit(df, 'net_present_cost', unit_cost)
    # Define custom colors
    net_present_cost_color = '#1f77b4'  # Example: blue color
    carbon_emissions_color = '#ff7f0e'  # Example: orange color

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Define width of the bars
    bar_width = 0.35

    # Define positions for the bars
    scenario_positions = np.arange(len(df['scenario_name']))

    # Plotting for 'net_present_cost' on the left axis
    ax1.bar(scenario_positions - bar_width/2, df_converted_cost['net_present_cost'], bar_width, color=net_present_cost_color, label='Net Present Cost')
    ylabel1 = 'Net Present Cost'  + f' [{output_unit_cost}]'
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel(ylabel1, color=net_present_cost_color)
    ax1.tick_params(axis='y', labelcolor=net_present_cost_color)

    # Adding legend for left axis
    ax1.legend(loc='upper left')

    # Creating a second y-axis for 'carbon_emissions_cumulative' on the right side
    ax2 = ax1.twinx()

    # Plotting for 'carbon_emissions_cumulative' on the right axis
    ylabel2 = 'Carbon Emissions'  + f' [{output_unit_co2}]'
    ax2.bar(scenario_positions + bar_width/2, df_converted_co2['carbon_emissions_cumulative'], bar_width, color=carbon_emissions_color, label='Carbon Emissions')
    ax2.set_ylabel(ylabel2, color=carbon_emissions_color)
    ax2.tick_params(axis='y', labelcolor=carbon_emissions_color)

    # Adding legend for right axis
    ax2.legend(loc='upper right')

    ax1.set_title('Net Present Cost and Carbon Emissions by Scenario')
    ax1.set_xticks(scenario_positions)
    ax1.set_xticklabels(df['scenario_name'], rotation=70)
    plt.tight_layout()
    plt.show()




def plot_pareto_front(df, parent_folder, output_path, title, unit_co2, unit_y_axis, y_axis, y_axis_label, save_fig=True):
    """
    Plot the Pareto front for the given data.

    Parameters:
    df (DataFrame): The data containing optimization results.
    parent_folder (str): The parent folder for saving the plot.
    output_path (str): The base output path where the plot will be saved.
    title (str): The title of the plot.
    unit_co2 (str): The unit for CO2 emissions.
    unit_y_axis (str): The unit for the y-axis.
    y_axis (str): The column name for the y-axis data.
    y_axis_label (str): The label for the y-axis.
    save_fig (bool): Whether to save the figure. Defaults to True.
    """

    # Print information about the step in the optimization process
    print(f"Plotting Pareto front for {title}:\n")

    # Define the colors to use
def plot_pareto_front(df, parent_folder, output_path, title, unit_co2, unit_y_axis, y_axis, y_axis_label, save_fig=True):
    """
    Plot the Pareto front for the given data.

    Parameters:
    df (DataFrame): The data containing optimization results.
    parent_folder (str): The parent folder for saving the plot.
    output_path (str): The base output path where the plot will be saved.
    title (str): The title of the plot.
    unit_co2 (str): The unit for CO2 emissions.
    unit_y_axis (str): The unit for the y-axis.
    y_axis (str): The column name for the y-axis data.
    y_axis_label (str): The label for the y-axis.
    save_fig (bool): Whether to save the figure. Defaults to True.
    """

    # Print information about the step in the optimization process
    print(f"Plotting Pareto front for {title}:\n")

    # Define the colors to use
    colors = ['#818F42', '#3395ab', '#C55D57']

    # Convert units for carbon emissions and the y-axis data
    # Convert units for carbon emissions and the y-axis data
    output_unit_co2, df_converted, _ = get_best_unit(df, 'carbon_emissions_cumulative', unit_co2)
    output_unit_y_axis, df_converted, _ = get_best_unit(df_converted, y_axis, unit_y_axis)

    # Create the plot
    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each Pareto group
    # Plot each Pareto group
    pareto_groups = df_converted['pareto_group'].unique()
    for idx, pareto_group in enumerate(pareto_groups):
        group_df = df_converted[df_converted['pareto_group'] == pareto_group]
        group_df.sort_values(by='carbon_emissions_cumulative', inplace=True)
        plt.plot(group_df['carbon_emissions_cumulative'], group_df[y_axis], marker='o', color=colors[idx % len(colors)])

    # Label the axes and add a title
    # Label the axes and add a title
    plt.xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
    plt.ylabel(f'{y_axis_label} [{output_unit_y_axis}]')
    plt.title(title)
    #plt.legend()
    plt.grid(True)

    # Create the directory if it does not exist
    save_folder = os.path.join(output_path, parent_folder, 'Figures')
    os.makedirs(save_folder, exist_ok=True)

    # Save the figure if required
    # Save the figure if required
    if save_fig:
        save_file = f'{y_axis}.png'
        save_file_path = os.path.join(save_folder, save_file)
        plt.savefig(save_file_path)
        print(f"Saving Pareto front for {title} as {save_file_path}\n")


    #plt.show()



    # Define the colors to use
def plot_pareto_front_cost(df, parent_folder, output_path, title, unit_cost, unit_y_axis, y_axis, y_axis_label, save_fig=True):
    """
    Plot the Pareto front for the given data based on net present cost.

    Parameters:
    df (DataFrame): The data containing optimization results.
    parent_folder (str): The parent folder for saving the plot.
    output_path (str): The base output path where the plot will be saved.
    title (str): The title of the plot.
    unit_cost (str): The unit for net present cost.
    unit_y_axis (str): The unit for the y-axis.
    y_axis (str): The column name for the y-axis data.
    y_axis_label (str): The label for the y-axis.
    save_fig (bool): Whether to save the figure. Defaults to True.
    """

    # Print information about the step in the optimization process
    print(f"Plotting Pareto front for {title}:\n")

    # Define the colors to use
    colors = ['#818F42', '#3395ab', '#C55D57']

    # Convert units for net present cost and the y-axis data
    output_unit_cost, df_converted, _ = get_best_unit(df, 'net_present_cost', unit_cost)
    output_unit_y_axis, df_converted, _ = get_best_unit(df_converted, y_axis, unit_y_axis)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot each Pareto group
    pareto_groups = df_converted['pareto_group'].unique()
    for idx, pareto_group in enumerate(pareto_groups):
        group_df = df_converted[df_converted['pareto_group'] == pareto_group]
        group_df.sort_values(by='net_present_cost', inplace=True)
        plt.plot(group_df['net_present_cost'], group_df[y_axis],
                 label=f'Time steps {pareto_group}', marker='o', color=colors[idx % len(colors)])

    # Label the axes and add a title
    plt.xlabel(f'Net Present Cost [{output_unit_cost}]')
    plt.ylabel(f'{y_axis_label} [{output_unit_y_axis}]')
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Create the directory if it does not exist
    save_folder = os.path.join(output_path, parent_folder, 'Figures')
    os.makedirs(save_folder, exist_ok=True)

    # Save the figure if required
    if save_fig:
        save_file = f'{y_axis}_net_present_cost.png'
        save_file_path = os.path.join(save_folder, save_file)
        plt.savefig(save_file_path)
        print(f"Saving Pareto front for {title} as {save_file_path}\n")

    #plt.show()

def plot_pareto_front_3d(df, parent_folder, title, unit_co2, unit_z_axis, unit_cost, z_axis, z_axis_label, save_fig=False):
    output_unit_co2, df_converted, _ = get_best_unit(df, 'carbon_emissions_cumulative', unit_co2)

    output_unit_y_axis, df_converted, _ = get_best_unit(df_converted, 'net_present_cost', unit_cost)
    output_unit_z_axis, df_converted, _ = get_best_unit(df_converted, z_axis, unit_z_axis)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for pareto_group in df_converted['pareto_group'].unique():
        group_df = df_converted[df_converted['pareto_group'] == pareto_group]
        group_df.sort_values(by='carbon_emissions_cumulative', inplace=True)

        X = group_df['carbon_emissions_cumulative'].values
        Y = group_df['net_present_cost'].values
        Z = group_df[z_axis].values

        # Creating a surface plot
        ax.plot_trisurf(X, Y, Z, label=f'Time steps {pareto_group}', linewidth=0.2, antialiased=True, alpha=0.6)

    ax.set_xlabel(f'Cumulative Carbon Emissions [{output_unit_co2}]')
    ax.set_ylabel(f'Net Present Cost [{output_unit_y_axis}]')
    ax.set_zlabel(f'{z_axis_label} [{output_unit_z_axis}]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    save_folder = os.path.join("../data/outputs", parent_folder, 'Figures')
    # Make directory if it does not exist
    os.makedirs(save_folder, exist_ok=True)
    if save_fig:
        save_file = f'{z_axis}_3d_plot.png'
        save_file_path = os.path.join(save_folder, save_file)

        plt.savefig(save_file_path)
    plt.show()


def plot_pareto_front_3d_scatter(df, parent_folder, title, unit_co2, unit_z_axis, unit_cost, z_axis, z_axis_label, save_fig=False):
    output_unit_co2, df_converted, _ = get_best_unit(df, 'carbon_emissions_cumulative', unit_co2)
    output_unit_y_axis, df_converted, _ = get_best_unit(df_converted, 'net_present_cost', unit_cost)
    output_unit_z_axis, df_converted, _ = get_best_unit(df_converted, z_axis, unit_z_axis)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    for pareto_group in df_converted['pareto_group'].unique():
        group_df = df_converted[df_converted['pareto_group'] == pareto_group]
        group_df.sort_values(by='carbon_emissions_cumulative', inplace=True)

        X = group_df['carbon_emissions_cumulative'].values
        Y = group_df['net_present_cost'].values
        Z = group_df[z_axis].values

        # Creating a scatter plot
        ax.scatter(X, Y, Z, label=f'Time steps {pareto_group}', alpha=0.6)

    ax.set_xlabel(f'Cumulative Carbon Emissions [{output_unit_co2}]')
    ax.set_ylabel(f'Net Present Cost [{output_unit_y_axis}]')
    ax.set_zlabel(f'{z_axis_label} [{output_unit_z_axis}]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

    save_folder = os.path.join("../data/outputs", parent_folder, 'Figures')
    # Make directory if it does not exist
    os.makedirs(save_folder, exist_ok=True)
    if save_fig:
        save_file = f'{z_axis}_3d_plot.png'
        save_file_path = os.path.join(save_folder, save_file)

        plt.savefig(save_file_path)
    plt.show()


def plot_costs_boxplot(df, unit):

    melted_df = pd.melt(df, id_vars=['scenario_name'], value_vars=['cost_capex_total', 'cost_opex_total', 'cost_carbon_emissions_total', 'cost_carrier_total'], var_name='cost_type', value_name='cost_value')
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    output_unit, df_converted, vmax_output = get_best_unit(melted_df, 'cost_value', unit)
    for carrier, subset in df_converted.groupby(['cost_type']):
        ax.bar(subset['scenario_name'], subset['cost_value'], label=carrier[0])  # Extracting carrier name from tuple

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Costs' + f' [{output_unit}]')
    ax.set_title('Costs by Scenario and Cost')

    # Add legend outside the loop
    ax.legend()

    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()

def plot_comparison_PV_in_out(df_PV_max_load, node, start_hour, duration):
    input_col = 'input_' + node
    output_col = 'output_' + node
    label_max_load = 'Max Load ' + node
    label_flow_output = 'Flow Conversion Output PV ' + node
    titel = 'Max Load and Flow Conversion Output for ' + node
    columns = df_PV_max_load.columns
    index = df_PV_max_load.index
    # Ensure 'time' is the index
    if 'time' in df_PV_max_load.columns:
        df_PV_max_load.set_index('time', inplace=True, drop=False)
    #df_PV_max_load.set_index('time', inplace=True, drop=False)
    #df_PV_max_load.index = pd.date_range(start='2023-01-01', periods=len(df_PV_max_load), freq='h')

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=(30, 6))

    # Plot the input data on the primary y-axis
    ax1.plot(df_PV_max_load['time'], df_PV_max_load[input_col], label=label_max_load, color='#1f77b4', marker='o')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Max Load', color='#1f77b4')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')

    # Create a secondary y-axis for the output data
    ax2 = ax1.twinx()
    ax2.plot(df_PV_max_load['time'], df_PV_max_load[output_col], label=label_flow_output, color='#ff7f0e', marker='x')
    ax2.set_ylabel('Flow Conversion Output PV', color='#ff7f0e')
    ax2.tick_params(axis='y', labelcolor='#ff7f0e')

    # Set x-axis limits
    ax1.set_xlim([df_PV_max_load['time'][start_hour], df_PV_max_load['time'][start_hour + duration - 1]])
    #ax1.set_xlim([df_PV_max_load.index[start_hour], df_PV_max_load.index[start_hour + duration - 1]])

    # Add titles and labels
    plt.title(titel)

    # Add grid, legends, and show plot
    ax1.grid(True)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()


def plot_boxplot_capacities_states(scenarios, df_filtered, output_path, folder):
    """
    Plot boxplots of capacity additions for different states and scenarios.

    Parameters:
    scenarios (list of str): List of scenario names to plot.
    df_filtered (DataFrame): Filtered DataFrame containing the data to plot.
    output_path (str): The base directory where the plots will be saved.
    folder (str): The subdirectory within the output_path to save the plots.

    Returns:
    None
    """
    # Ensure the output folder exists
    os.makedirs(os.path.join(output_path, folder), exist_ok=True)

    for scenario in scenarios:
        scenario_col = f'capacity_addition_{scenario}'
        df_scenario = df_filtered[['state_full', 'tech_cap', scenario_col]].copy()
        df_scenario.rename(columns={scenario_col: 'capacity_addition'}, inplace=True)

        # Plot using FacetGrid
        g = sns.FacetGrid(df_scenario, col='state_full', col_wrap=6, sharey=False)
        g.map_dataframe(sns.barplot, x='tech_cap', y='capacity_addition', hue='tech_cap', palette='Set2')

        # Add legend and adjust the layout
        g.add_legend()
        g.set_axis_labels("Technology", "Capacity Addition")
        g.set_titles("{col_name}")
        g.set_xticklabels(rotation=90)
        max_capacity = df_filtered[[f'capacity_addition_{scen}' for scen in scenarios]].max().max()
        g.set(ylim=(0, max_capacity))

        # Set x labels
        tech_labels = [label if label else '' for label in df_scenario['tech_cap'].unique()]
        for ax in g.axes.flat:
            ax.set_xticklabels(labels=tech_labels, rotation=90)

        plt.subplots_adjust(top=0.9)
        g.fig.suptitle(f"Capacity Addition for {scenario}", y=1.02)

        # Save the plot
        save_file = f"capacity_addition_{scenario}.png"
        save_folder = os.path.join(output_path, folder, 'Figures')
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, save_file)

        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to avoid overlapping

        print(f"Saved plot for {scenario} at {save_path}")


def plot_boxplot_energy_states(scenarios, filtered_data, parent_folder, output_path, target_column):
    """
    Plots boxplots of energy data for various states and scenarios.

    Args:
        scenarios (list): List of scenario names to plot.
        filtered_data (DataFrame): DataFrame containing the filtered data to plot.
        output_folder (str): Parent folder where the plots will be saved.
        target_column (str): The column name to be plotted.

    Returns:
        None
    """

    for scenario in scenarios:
        # Inform what is being plotted
        print(f"Plotting {target_column} for {scenario}")
        scenario_column = f'{target_column}_{scenario}'
        scenario_data = filtered_data[['state_full', 'carrier', scenario_column]].copy()
        scenario_data.rename(columns={scenario_column: 'flow_import'}, inplace=True)

        # Plot using FacetGrid
        g = sns.FacetGrid(scenario_data, col='state_full', col_wrap=6, sharey=False)
        g.map_dataframe(sns.barplot, x='carrier', y='flow_import', hue='carrier', legend=False, palette='Set2')
        g.add_legend()

        # Create the y label from the column name
        y_label = target_column.replace('_', ' ').capitalize()
        g.set_axis_labels("Carrier", y_label)
        g.set_titles("{col_name}")
        plt.subplots_adjust(top=0.9)
        g.set_xticklabels(rotation=90)

        # Set x labels
        x_labels = [label if label else '' for label in scenario_data['carrier'].unique()]
        for ax in g.axes.flat:
            ax.set_xticklabels(labels=x_labels, rotation=90)

        # Set plot title and y-axis limit
        g.fig.suptitle(f"{y_label} for {scenario}", y=1.02)
        max_flow_import = filtered_data[[f'{target_column}_{scn}' for scn in scenarios]].max().max()
        g.set(ylim=(0, max_flow_import))

        # Save the plot
        save_file = f"{target_column}_{scenario}.png"
        save_folder = os.path.join(output_path, parent_folder, 'Figures')
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, save_file)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()  # Close the figure to avoid overlapping
        print(f"Saved plot to {save_path}")
        # plt.show()

def plot_stacked_costs(df, output_path,parent_folder, units, point=None, save_fig=True):
    # Drop the row where the scenario is ''
    df = df[df['scenario'] != '']

    file_title = 'cost_stacked'
    # Convert units for carbon emissions and the y-axis data
    output_unit_co2, df_converted, factor_co2 = get_best_unit(df, 'carbon_emissions_cumulative', units['co2'])
    columns = ['cost_capex_total', 'cost_opex_total', 'cost_carbon_emissions_total', 'cost_carrier_total']
    output_unit_y_axis, df_converted, factor_cost = get_best_unit(df_converted, columns, units['cost'])

    # Sort df by carbon_emissions_cumulative
    df_converted = df_converted.sort_values(by='carbon_emissions_cumulative')
    # Extracting the data needed for the plot
    x = df_converted['carbon_emissions_cumulative']
    y1 = df_converted['cost_capex_total']
    y2 = df_converted['cost_opex_total']
    y3 = df_converted['cost_carrier_total']
    y4 = df_converted['cost_carbon_emissions_total']
    # Plotting the stacked area chart using Seaborn
    plt.figure(figsize=(10, 6))
    sns.set(style="white")
    colors = ['#64557B', '#F4D35E',  '#62866C', '#CB7876']
    plt.stackplot(x, y1, y2, y3, y4, labels=['Capex', 'Opex', 'Cost Carrier', 'Cost $\\mathrm{{CO_2}}$ Emissions'], colors=colors)
    print(f'max capex: {y1.max()}')


    ################################################################################################
    # Add a separate point
    if point.any():
        print(factor_cost)
        print(factor_co2)
        bau_co2_emissions = point[0]*factor_co2  # Example value for CO2 emissions
        bau_cost = point[1]*factor_cost  # Example value for cost
        print(bau_cost)
        plt.scatter(bau_co2_emissions, bau_cost, color='#32769B', zorder=5)
        plt.text(bau_co2_emissions, bau_cost, 'BAU', fontsize=12, color='black', ha='right')

    ################################################################################################
    plt.xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
    plt.ylabel(f'Cost [{output_unit_y_axis}]')
    plt.title('Cost vs. $\\mathrm{{CO_2}}$ Emissions')
    plt.legend(loc='upper right')

    #Save the figure if required
    save_folder = os.path.join(output_path, parent_folder, 'Figures')
    os.makedirs(save_folder, exist_ok=True)

    if save_fig:
        save_file = f'{file_title}.png'
        save_file_path = os.path.join(save_folder, save_file)
        plt.savefig(save_file_path, bbox_inches='tight')
        print(f"Saving Pareto front for {file_title} as {save_file_path}\n")

    #plt.show()


def plot_percentage_stacked_costs(df, output_path, parent_folder, units, save_fig=True):
    # Drop the row where the scenario is ''
    df = df[df['scenario'] != '']

    file_title = 'cost_stacked_percentage'
    # Convert units for carbon emissions and the y-axis data
    output_unit_co2, df_converted, _ = get_best_unit(df, 'carbon_emissions_cumulative', units['co2'])
    columns = ['cost_capex_total', 'cost_opex_total', 'cost_carbon_emissions_total', 'cost_carrier_total']
    output_unit_y_axis, df_converted, _ = get_best_unit(df_converted, columns, units['cost'])

    # Calculate the total cost for each row
    df_converted['total_cost'] = df_converted[columns].sum(axis=1)

    # Calculate the percentage distribution
    for column in columns:
        df_converted[column + '_pct'] = df_converted[column] / df_converted['total_cost'] * 100

    # Sort df by carbon_emissions_cumulative
    df_converted = df_converted.sort_values(by='carbon_emissions_cumulative')

    # Extracting the data needed for the plot
    x = df_converted['carbon_emissions_cumulative']
    y1 = df_converted['cost_capex_total_pct']
    y2 = df_converted['cost_opex_total_pct']
    y3 = df_converted['cost_carrier_total_pct']
    y4 = df_converted['cost_carbon_emissions_total_pct']

    # Plotting the stacked area chart using Seaborn
    plt.figure(figsize=(10, 6))
    sns.set(style="white")
    colors = ['#64557B', '#F4D35E',  '#62866C', '#CB7876']
    plt.stackplot(x, y1, y2, y3, y4, labels=['Capex', 'Opex', 'Cost Carrier', 'Cost $\\mathrm{{CO_2}}$ Emissions'], colors=colors)



    plt.xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
    plt.ylabel('Cost Percentage [%]')
    plt.title('Cost Percentage Distribution vs. $\\mathrm{{CO_2}}$ Emissions')
    plt.legend(loc='upper right')

    # Save the figure if required
    save_folder = os.path.join(output_path, parent_folder, 'Figures')
    os.makedirs(save_folder, exist_ok=True)

    if save_fig:
        save_file = f'{file_title}.png'
        save_file_path = os.path.join(save_folder, save_file)
        plt.savefig(save_file_path, bbox_inches='tight')
        print(f"Saving Pareto front for {file_title} as {save_file_path}\n")
    #plt.show()


def plot_pareto_capacities(result_capacities_dfs, output_path, parent_folder, units, save_fig=True):
    # Capacity components list
    capacity_components = [
        ('PV, power', f'PV', units['power'], 'Installed Capacity'),
        ('water_storage, energy', f'Water Storage', units['water_energy'], 'Installed Capacity'),
        ('battery, energy', f'Battery', units['energy'], 'Installed Capacity'),
#        ('diesel_WP, power', f'Diesel Water Pump', units['power'], 'Installed Capacity'),
        ('el_WP, power', f'Electric Water Pump', units['water_power'], 'Installed Capacity')
    ]
    # Convert units for carbon emissions
    output_unit_co2, df_converted, _ = get_best_unit(result_capacities_dfs, 'carbon_emissions_cumulative', units['co2'])

    # Create the subplots
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5), sharex=True)

    # Plot each component
    for ax, (y_axis, title, unit_y_axis, y_axis_label) in zip(axes, capacity_components):
        output_unit_y_axis, df_converted, _ = get_best_unit(df_converted, y_axis, unit_y_axis)
        ax.plot(df_converted['carbon_emissions_cumulative'], df_converted[y_axis], marker='o')
        ax.set_title(title)
        ax.set_xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
        ax.set_ylabel(f'{y_axis_label} [{output_unit_y_axis}]')
    # Add overall title
    #fig.suptitle(f'Pareto Front: Capacities vs. $\\mathrm{{CO_2}}$ Emissions', y=1.05, fontsize=16)
    # Adjust subplot spacing
    plt.subplots_adjust(wspace=0.5)  # Adjust horizontal space between subplots
    # Save the figure if required
    save_folder = os.path.join(output_path, parent_folder, 'Figures')
    os.makedirs(save_folder, exist_ok=True)

    if save_fig:
        save_file = 'capacities_pareto.png'
        save_file_path = os.path.join(save_folder, save_file)
        plt.savefig(save_file_path, bbox_inches='tight')
        print(f"Saving Pareto front for {save_file} as {save_file_path}\n")


def plot_stacked_import(df, output_path, parent_folder, units, save_fig=True):
    # Drop the row where the scenario is ''
    df = df[df['scenario'] != '']

    file_title = 'import_stacked'
    # Convert units for carbon emissions and the y-axis data
    output_unit_co2, df_converted, _ = get_best_unit(df, 'carbon_emissions_cumulative', units['co2'])
    columns = ['diesel', 'electricity']
    output_unit_y_axis, df_converted, _ = get_best_unit(df_converted, columns, units['energy'])

    # Sort df by carbon_emissions_cumulative
    df_converted = df_converted.sort_values(by='carbon_emissions_cumulative')
    # Extracting the data needed for the plot
    x = df_converted['carbon_emissions_cumulative']
    y1 = df_converted['diesel']
    y2 = df_converted['electricity']
    # Plotting the stacked area chart using Seaborn
    plt.figure(figsize=(10, 6))
    sns.set(style="white")
    colors = ['#64557B', '#F4D35E']
    plt.stackplot(x, y1, y2, labels=['Diesel', 'Electricity'], colors=colors)
    plt.xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
    plt.ylabel(f'Import [{output_unit_y_axis}]')
    plt.title('Import energy carrier vs. $\\mathrm{{CO_2}}$ Emissions')
    plt.legend(loc='upper left')

    #Save the figure if required
    save_folder = os.path.join(output_path, parent_folder, 'Figures')
    os.makedirs(save_folder, exist_ok=True)

    if save_fig:
        save_file = f'{file_title}.png'
        save_file_path = os.path.join(save_folder, save_file)
        plt.savefig(save_file_path, bbox_inches='tight')
        print(f"Saving Pareto front for {file_title} as {save_file_path}\n")



# Define a function to create a main figure and axes
def create_main_figure(num_scenarios):
    num_cols = num_scenarios  # Adjust as needed based on your scenarios
    fig, axes = plt.subplots(1, num_cols, figsize=(30, 20))
    return fig, axes.flatten()

def plot_scenario2(ax, df_merge, us_gdf, column_name, title, ylabel, default_vmax):
    df_merge.crs = 'EPSG:4326'

    # Reproject GeoDataFrame us_gdf to match the CRS of DataFrame df
    us_gdf = us_gdf.to_crs(df_merge.crs)

    df_map = us_gdf.merge(df_merge, on="county_code", how='left')

    ax.axis('off')  # Remove the frame around the map

    # Formatting changes to the colorbar
    vmax = df_merge[column_name].max() if default_vmax is None else default_vmax
    default_vmax = vmax  # Ensure default_vmax is set for ranges

    # Focus around the contiguous US (excluding Alaska and Hawaii)
    ax.set_xlim(-125, -65)
    ax.set_ylim(24, 50)

    # Define the ranges and colors
    boundaries = [0, 0.05 * default_vmax, 0.25 * default_vmax, 0.75 * default_vmax, default_vmax]
    colors = ['#B0B0B0', '#62866C', '#F4A261', '#CB7876']  # Define the colors for low, medium, high ranges

    # Create a custom colormap
    cmap = ListedColormap(colors)

    # Create the normalization and color mapping
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)

    # Plot the geospatial data with normalization
    df_map.plot(column=column_name, cmap=cmap, linewidth=0.4, ax=ax, edgecolor=".4",
                missing_kwds={"color": "white"}, legend=True, norm=norm,
                legend_kwds={'label': ylabel, 'orientation': 'vertical', 'shrink': 0.5,
                             'pad': 0.12})

    ax.set_title(title, fontsize=20)

# Define a function to plot each subplot for a scenario
def plot_scenario(ax, df_merge, us_gdf, column_name, title, ylabel, default_vmax):
    df_merge.crs = 'EPSG:4326'

    # Reproject GeoDataFrame us_gdf to match the CRS of DataFrame df
    us_gdf = us_gdf.to_crs(df_merge.crs)

    df_map = us_gdf.merge(df_merge, on="county_code", how='left')

    ax.axis('off')  # Remove the frame around the map

    # Formatting changes to the colorbar
    vmax = df_merge[column_name].max() if default_vmax is None else default_vmax

    # Focus around the contiguous US (excluding Alaska and Hawaii)
    ax.set_xlim(-125, -65)
    ax.set_ylim(24, 50)

    # Define the normalization and ticks for the colorbar
    norm = Normalize(vmin=0, vmax=vmax)
    ticks = np.linspace(0, vmax, 5)

    #Plot the geospatial data with normalization
    df_map.plot(column=column_name, cmap="Reds", linewidth=0.4, ax=ax, edgecolor=".4",
                missing_kwds={"color": "white"}, legend=True, vmin=0, vmax=vmax,
                legend_kwds={'label': ylabel, 'orientation': 'vertical', 'shrink': 0.2,
                             'pad': 0.12, 'ticks': ticks})

    ax.set_title(title, fontsize=20)


# Main plotting logic
def plot_scenarios(scenarios, df_capacities, df_demand_water, df_tech_cap_info, us_gdf, output_path, folder, save_filename, suptitle):
    num_def_tech_cap = len(df_tech_cap_info)
    print(num_def_tech_cap)


    for i, scenario in enumerate(scenarios):
        fig, axes = create_main_figure(num_def_tech_cap)
        scenario_title = scenario
        scenario_title = scenario_title.title()
        scenario_title = scenario_title.replace('Scenario_0', 'Net-Zero-Emissions-Scenario')
        scenario_title = scenario_title.replace('Scenario_', 'Cost-Optimal-Scenario')
        fig.suptitle(f'{suptitle} for Scenario: {scenario_title}', fontsize=24, y=0.75)
        print(scenario)
        if scenario != 'Cost-Optimal-Scenario':
            df_scenario = df_capacities[df_capacities['scenario'] == scenario]
        else:
            df_scenario = df_capacities.copy()

        for index, (technology, capacity_type, carrier, input_unit, unit_cap_demand, factor, max_value) in enumerate(df_tech_cap_info.itertuples(index=False)):
            print(technology, capacity_type, carrier, unit_cap_demand)
            if not pd.isna(technology):
                # Filter capacity data for current technology and capacity type
                df_tech_cap = df_scenario[(df_scenario['technology'] == technology) & (df_scenario['capacity_type'] == capacity_type)]
            else:
                df_tech_cap = df_scenario[(df_scenario['carrier'] == carrier)]

            if 'scenario' in df_tech_cap.columns:
                df_merge = df_tech_cap.merge(df_demand_water[['scenario', 'county_code', 'demand']], on=['county_code', 'scenario'])
            else:
                df_merge = df_tech_cap.merge(df_demand_water[['county_code', 'demand']], on='county_code')

            if not pd.isna(technology):
                df_merge['capacity/demand'] = df_merge['capacity'] / df_merge['demand'] * 10**6 * factor
                column_name = 'capacity/demand'
                title = f'{technology} {capacity_type}'
            else:
                df_merge['flow_import/demand'] = df_merge['flow_import'] / df_merge['demand'] * 10**6 * factor
                column_name = 'flow_import/demand'
                title = f'{carrier}'


            title = title.title()
            title = title.replace('Scenario_0', 'Net-Zero-Emissions-Scenario')
            title = title.replace('Scenario_', 'Cost-Optimal-Scenario')
            title = title.replace('No_scearnio', 'Cost-Optimal-Scenario')
            title = title.replace('_', ' ')

            ylabel = f'{column_name} [{unit_cap_demand}]'
            # Plot the scenario on the appropriate subplot
            plot_scenario2(axes[index], df_merge, us_gdf, column_name, title, ylabel, max_value)

        # Ensure layout is tight and save or show the figure
        plt.tight_layout()
        #plt.subplots_adjust(top=0.50)  # Adjust the top parameter to fit the title closer
        save_path = os.path.join(output_path, folder, 'Figures')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_filename_temp = scenario + '_' + save_filename
        plt.savefig(os.path.join(save_path, save_filename_temp), bbox_inches='tight')
        print(f'Figure saved to {os.path.join(save_path, save_filename)}')
        #plt.show()
        plt.close()

def plot_stacked_procentage_BAU(dfs, output_path, folder, units, save_fig=True):
    # Calculate the percentage distribution
    dfs['total_cost'] = dfs[['cost_capex_total', 'cost_opex_total', 'cost_carbon_emissions_total', 'cost_carrier_total']].sum(axis=1)
    y1 = dfs['cost_capex_total'].iloc[0] / dfs['total_cost'].iloc[0] * 100
    y2 = dfs['cost_opex_total'].iloc[0]/ dfs['total_cost'].iloc[0] * 100

    y3 = dfs['cost_carrier_total'].iloc[0] / dfs['total_cost'].iloc[0] * 100
    y4 = dfs['cost_carbon_emissions_total'].iloc[0] / dfs['total_cost'].iloc[0] * 100
    output_unit_co2, df_converted, factor_co2 = get_best_unit(dfs, 'carbon_emissions_cumulative', units['co2'])
    x = df_converted['carbon_emissions_cumulative'].iloc[0].round(1)
    categories = (str(x))
    data_base = {
        'Capex': np.array([y1]),
        'Opex' : np.array([y2]),
        'Cost Carrier' : np.array([y3]),
        'Cost $\\mathrm{{CO_2}}$ Emissions' : np.array([y4])
    }


    width = 0.2  # the width of the bars
    colors = ['#64557B', '#F4D35E', '#62866C', '#CB7876']


    fig, ax = plt.subplots(figsize=(4, 6))


    # Initialize the bottom position for the stacked bars
    bottom = np.zeros(len(categories))

    # Plot each component with specified colors
    for (base, base_count), color in zip(data_base.items(), colors):
        p = ax.bar(categories, base_count, width, label=base, bottom=bottom, color=color)
        bottom += base_count

    ax.set_title('Percentage distribution of costs in the BAU scenario')
    ax.set_ylabel('Percentage')
    ax.set_xlabel(f'$\\mathrm{{CO_2}}$  Emissions [{output_unit_co2}]')
    # Add some space to the right and left of the bars
    ax.margins(x=0.2)

    # Place the legend in the upper right corner
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    if save_fig:
        save_file = f'cost_stacked_percentag_BAU.png'
        save_file_path = os.path.join(output_path, folder, 'Figures')    # Save the figure if required
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        plt.savefig(os.path.join(save_file_path, save_file), bbox_inches='tight')
    plt.show()


def plot_stacked_import_BAU(folder, output_path, df, units, save_fig=True):
    # Drop the row where the scenario is ''
    df = df[df['scenario'] != '']

    file_title = 'import_stacked_BAU'
    # Convert units for carbon emissions and the y-axis data
    output_unit_co2, df_converted, _ = get_best_unit(df, 'carbon_emissions_cumulative', units['co2'])
    columns = ['diesel', 'electricity']
    output_unit_y_axis, df_converted, _ = get_best_unit(df_converted, columns, units['energy'])

    # Sort df by carbon_emissions_cumulative
    df_converted = df_converted.sort_values(by='carbon_emissions_cumulative')
    # Extracting the data needed for the plot
    x = df_converted['carbon_emissions_cumulative'].iloc[0].round(1)
    y1 = df_converted['diesel'].iloc[0]
    y2 = df_converted['electricity'].iloc[0]

    cat1 = f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]: {x}'
    categories = (cat1)
    data_base = {
        'Diesel': np.array([y1]),
        'Electricity' : np.array([y2]),
    }


    width = 0.2  # the width of the bars
    colors = ['#64557B', '#F4D35E']


    fig, ax = plt.subplots(figsize=(4, 6))


    # Initialize the bottom position for the stacked bars
    bottom = np.zeros(len(categories))

    # Plot each component with specified colors
    for (base, base_count), color in zip(data_base.items(), colors):
        p = ax.bar(categories, base_count, width, label=base, bottom=bottom, color=color)
        bottom += base_count

    ax.set_title('Imported Energy Carriers in the BAU scenario')
    ax.set_ylabel(f'Imports [{output_unit_y_axis}]')
    ax.set_xlabel(f'$\\mathrm{{CO_2}}$  Emissions [{output_unit_co2}]')
    # Add some space to the right and left of the bars
    ax.margins(x=0.2)

    # Place the legend in the upper right corner
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)
    if save_fig:
        save_file_path = os.path.join(output_path, folder, 'Figures')    # Save the figure if required
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        plt.savefig(os.path.join(save_file_path, file_title), bbox_inches='tight')
    #plt.show()


def plot_stacked_tech_car(folder, output_path, df, units, point, save_fig=True):
    file_title = 'cost_stacked'

    # Convert units for carbon emissions and the y-axis data
    output_unit_co2, df_converted, factor_co2 = get_best_unit(df, 'co2_emissions', units['co2'])
    output_unit_y_axis, df_converted, factor_cost = get_best_unit(df_converted, 'cost', units['cost'])
    output_unit_y_axis = output_unit_y_axis.replace('USD', '$')
    output_unit_y_axis = output_unit_y_axis.replace('giga', 'billion')

    # Sort df by co2_emissions
    df_converted = df_converted.sort_values(by='co2_emissions')

    # Initialize lists for the y-values and labels
    y_values = []
    labels = []

    # Extract the data needed for the plot
    for tech_car in df_converted['technology/carrier'].unique():
        if df_converted[df_converted['technology/carrier'] == tech_car]['cost'].isnull().all() or df_converted[df_converted['technology/carrier'] == tech_car]['cost'].sum() == 0:
            continue
        if df_converted[df_converted['technology/carrier'] == tech_car]['cost'].sum() / df_converted['cost'].sum() < 0.01:
            continue
        y_values.append(df_converted[df_converted['technology/carrier'] == tech_car]['cost'].values)
        labels.append(tech_car)

    # Convert y_values to a 2D array for stackplot
    y_values = np.array(y_values)

    # Extract x-axis data
    df_co2_emissions = df_converted.groupby('scenario')['co2_emissions'].first().reset_index()
    x = df_co2_emissions['co2_emissions'].values
    x = np.unique(x)
    x = np.sort(x)

    # Calculate the percentual distribution
    total_cost = np.sum(y_values, axis=0)
    percent_distribution = (y_values / total_cost) * 100

    # Plotting the charts using Matplotlib and GridSpec
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, width_ratios=[3, 1])

    # Stacked area chart
    ax0 = plt.subplot(gs[0])
    if point.any():
        bau_co2_emissions = point[0] * factor_co2
        bau_cost = point[1] * factor_cost
        ax0.scatter(bau_co2_emissions, bau_cost, color='#32769B', zorder=5)
        ax0.text(bau_co2_emissions, bau_cost, 'BAU', fontsize=12, color='black', ha='right')
    ax0.stackplot(x, y_values, labels=labels, colors=Category10[10][:len(labels)])
    ax0.set_xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
    ax0.set_ylabel(f'Cost [{output_unit_y_axis}]')
    ax0.legend(loc='upper right')
    ax0.set_title('Stacked Area Chart of Costs vs $\\mathrm{{CO_2}}$ Emissions')

    # Percentual distribution plot
    ax1 = plt.subplot(gs[1])
    ax1.stackplot(x, percent_distribution, labels=labels, colors=Category10[10][:len(labels)])
    ax1.set_xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
    ax1.set_ylabel('Percentual Distribution [%]')
    #ax1.legend(loc='upper left')
    ax1.set_title('Percentual Distribution')

    plt.tight_layout()

    if save_fig:
        save_file_path = os.path.join(output_path, folder, 'Figures')
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        plt.savefig(os.path.join(save_file_path, file_title), bbox_inches='tight')
        print(f"Saving Pareto front for {file_title} as {save_file_path}\n")


def plot_stacked_tech_car2(folder, output_path, df, units, point, dfs_BAU, save_fig=True):
    file_title = 'cost_stacked'

    # Convert units for carbon emissions and the y-axis data
    output_unit_co2, df_converted, factor_co2 = get_best_unit(df, 'co2_emissions', units['co2'])
    output_unit_y_axis, df_converted, factor_cost = get_best_unit(df_converted, 'cost', units['cost'])
    output_unit_BAU, df_converted_BAU, factor_cost = get_best_unit(dfs_BAU, 'co2_emissions', units['co2'])
    output_unit_y_axis = output_unit_y_axis.replace('USD', '$')
    output_unit_y_axis = output_unit_y_axis.replace('giga', 'billion')

    # Sort df by co2_emissions
    df_converted = df_converted.sort_values(by='co2_emissions')

    # Initialize lists for the y-values and labels
    y_values = []
    labels = []

    # Extract the data needed for the plot
    for tech_car in df_converted['technology/carrier'].unique():
        if df_converted[df_converted['technology/carrier'] == tech_car]['cost'].isnull().all() or df_converted[df_converted['technology/carrier'] == tech_car]['cost'].sum() == 0:
            continue
        if df_converted[df_converted['technology/carrier'] == tech_car]['cost'].sum() / df_converted['cost'].sum() < 0.01:
            continue
        y_values.append(df_converted[df_converted['technology/carrier'] == tech_car]['cost'].values)
        labels.append(tech_car)

    # Convert y_values to a 2D array for stackplot
    y_values = np.array(y_values)

    # Extract x-axis data
    df_co2_emissions = df_converted.groupby('scenario')['co2_emissions'].first().reset_index()
    x = df_co2_emissions['co2_emissions'].values
    x = np.unique(x)
    x = np.sort(x)

    # Calculate the percentual distribution
    total_cost = np.sum(y_values, axis=0)
    percent_distribution = (y_values / total_cost) * 100

    y_values_BAU_dict = {}
    labels_BAU = []

    for tech_car in dfs_BAU['technology/carrier'].unique():
        if dfs_BAU[dfs_BAU['technology/carrier'] == tech_car]['cost'].isnull().all() or dfs_BAU[dfs_BAU['technology/carrier'] == tech_car]['cost'].sum() == 0:
            continue
        if dfs_BAU[dfs_BAU['technology/carrier'] == tech_car]['cost'].sum() / dfs_BAU['cost'].sum() < 0.01:
            continue
        y_values_BAU_dict[tech_car] = dfs_BAU[dfs_BAU['technology/carrier'] == tech_car]['cost'].values
        labels_BAU.append(tech_car)

    # Ensure the order of y_values_BAU matches the order of labels
    labels_BAU = [label for label in labels if label in labels_BAU]
    y_values_BAU = [y_values_BAU_dict[label] for label in labels_BAU]
    #

    y_values_BAU = np.array(y_values_BAU)
    total_cost_BAU = np.sum(y_values_BAU, axis=0)
    percent_distribution_BAU = (y_values_BAU / total_cost_BAU) * 100
    x_BAU = df_converted_BAU['co2_emissions'].iloc[0].round(1)
    categories = (f'{x_BAU} [{output_unit_BAU}]')
    data_base = {
        labels_BAU[i]: percent_distribution_BAU[i] for i in range(len(labels_BAU))
    }

    # Plotting the charts using Matplotlib and GridSpec
    fig = plt.figure(figsize=(21, 6))
    gs = GridSpec(1, 3, width_ratios=[3, 1, 1])

    # Stacked area chart
    ax0 = plt.subplot(gs[0])
    if point.any():
        bau_co2_emissions = point[0] * factor_co2
        bau_cost = point[1] * factor_cost
        ax0.scatter(bau_co2_emissions, bau_cost, color='#32769B', zorder=5)
        ax0.text(bau_co2_emissions, bau_cost, 'BAU', fontsize=12, color='black', ha='right')
    ax0.stackplot(x, y_values, labels=labels, colors=Category10[10][:len(labels)])
    ax0.set_xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
    ax0.set_ylabel(f'Cost [{output_unit_y_axis}]')
    ax0.legend(loc='upper right')
    ax0.set_title('Stacked Area Chart of Costs vs $\\mathrm{{CO_2}}$ Emissions')

    # Percentual distribution plot
    ax1 = plt.subplot(gs[1])
    ax1.stackplot(x, percent_distribution, labels=labels, colors=Category10[10][:len(labels)])
    ax1.set_xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
    ax1.set_ylabel('Percentual Distribution [%]')
    ax1.set_title('Percentual Distribution')

    # Percentual distribution plot for BAU scenario
    width = 0.2  # the width of the bars
    bottom = np.zeros(1)
    colors = Category10[10][:len(labels_BAU)]
    ax2 = plt.subplot(gs[2])
    # Plot each component with specified colors
    for (base, base_count), color in zip(data_base.items(), colors):
        p = ax2.bar(categories, base_count, width, label=base, bottom=bottom, color=color)
        bottom += base_count

    ax2.set_title('Percentage distribution of costs in the BAU scenario')
    ax2.set_ylabel('Percentage')
    ax2.set_xlabel(f'$\\mathrm{{CO_2}}$  Emissions [{output_unit_BAU}]')
    ax2.margins(x=0.2)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

    plt.tight_layout()

    if save_fig:
        save_file_path = os.path.join(output_path, folder, 'Figures')
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        plt.savefig(os.path.join(save_file_path, file_title), bbox_inches='tight')
        print(f"Saving Pareto front for {file_title} as {save_file_path}\n")



def plot_stacked_tech_car3(folder, output_path, df, units, point, dfs_BAU, save_fig=True):
    file_title = 'cost_stacked'

    # Convert units for carbon emissions and the y-axis data
    output_unit_co2, df_converted, factor_co2 = get_best_unit(df, 'co2_emissions', units['co2'])
    output_unit_y_axis, df_converted, factor_cost = get_best_unit(df_converted, 'cost', units['cost'])
    output_unit_BAU, df_converted_BAU, factor_co2_BAU = get_best_unit(dfs_BAU, 'co2_emissions', units['co2'])
    output_unit_y_axis = output_unit_y_axis.replace('USD', '$')
    output_unit_y_axis = output_unit_y_axis.replace('giga', 'billion')

    # Sort df by co2_emissions
    df_converted = df_converted.sort_values(by='co2_emissions')

    # Initialize lists for the y-values and labels
    y_values = []
    labels = []

    # Extract the data needed for the plot
    for tech_car in df_converted['technology/carrier'].unique():
        if df_converted[df_converted['technology/carrier'] == tech_car]['cost'].isnull().all() or df_converted[df_converted['technology/carrier'] == tech_car]['cost'].sum() == 0:
            continue
        if df_converted[df_converted['technology/carrier'] == tech_car]['cost'].sum() / df_converted['cost'].sum() < 0.01:
            continue
        y_values.append(df_converted[df_converted['technology/carrier'] == tech_car]['cost'].values)
        labels.append(tech_car)

    # Convert y_values to a 2D array for stackplot
    y_values = np.array(y_values)

    # Extract x-axis data
    df_co2_emissions = df_converted.groupby('scenario')['co2_emissions'].first().reset_index()
    x = df_co2_emissions['co2_emissions'].values
    x = np.unique(x)
    x = np.sort(x)

    # Calculate the percentual distribution
    total_cost = np.sum(y_values, axis=0)
    percent_distribution = (y_values / total_cost) * 100

    # Prepare data for the third plot (BAU scenario)
    y_values_BAU_dict = {}
    labels_BAU = []

    for tech_car in dfs_BAU['technology/carrier'].unique():
        if dfs_BAU[dfs_BAU['technology/carrier'] == tech_car]['cost'].isnull().all() or dfs_BAU[dfs_BAU['technology/carrier'] == tech_car]['cost'].sum() == 0:
            continue
        if dfs_BAU[dfs_BAU['technology/carrier'] == tech_car]['cost'].sum() / dfs_BAU['cost'].sum() < 0.01:
            continue
        y_values_BAU_dict[tech_car] = dfs_BAU[dfs_BAU['technology/carrier'] == tech_car]['cost'].values
        labels_BAU.append(tech_car)

    # Ensure the order of y_values_BAU matches the order of labels
    labels_BAU = [label for label in labels if label in labels_BAU]
    y_values_BAU = [y_values_BAU_dict[label] for label in labels_BAU]

    # Convert y_values_BAU to a 2D array
    y_values_BAU = np.array(y_values_BAU)
    total_cost_BAU = np.sum(y_values_BAU, axis=0)
    percent_distribution_BAU = (y_values_BAU / total_cost_BAU) * 100
    x_BAU = df_converted_BAU['co2_emissions'].iloc[0].round(1)
    categories = (f'{x_BAU} [{output_unit_BAU}]')
    data_base = {
        labels_BAU[i]: percent_distribution_BAU[i] for i in range(len(labels_BAU))
    }

    # Define colors based on labels
    color_dict = {label: Category10[10][i % 10] for i, label in enumerate(labels)}

    # Plotting the charts using Matplotlib and GridSpec
    fig = plt.figure(figsize=(21, 6))
    gs = GridSpec(1, 3, width_ratios=[3, 1, 1])

    # Stacked area chart
    ax0 = plt.subplot(gs[0])
    if point.any():
        bau_co2_emissions = point[0] * factor_co2
        bau_cost = point[1] * factor_cost
        ax0.scatter(bau_co2_emissions, bau_cost, color='#32769B', zorder=5)
        ax0.text(bau_co2_emissions, bau_cost, 'BAU', fontsize=12, color='black', ha='right')
    ax0.stackplot(x, y_values, labels=labels, colors=[color_dict[label] for label in labels])
    ax0.set_xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
    ax0.set_ylabel(f'Cost [{output_unit_y_axis}]')
    ax0.legend(loc='upper right')
    ax0.set_title('Stacked Area Chart of Costs vs $\\mathrm{{CO_2}}$ Emissions')

    # Percentual distribution plot
    ax1 = plt.subplot(gs[1])
    ax1.stackplot(x, percent_distribution, labels=labels, colors=[color_dict[label] for label in labels])
    ax1.set_xlabel(f'$\\mathrm{{CO_2}}$ Emissions [{output_unit_co2}]')
    ax1.set_ylabel('Percentual Distribution [%]')
    ax1.set_title('Percentual Distribution')

    # Percentual distribution plot for BAU scenario
    width = 0.2  # the width of the bars
    bottom = np.zeros(1)
    colors = [color_dict[label] for label in labels_BAU]
    ax2 = plt.subplot(gs[2])
    # Plot each component with specified colors
    for (base, base_count), color in zip(data_base.items(), colors):
        p = ax2.bar(categories, base_count, width, label=base, bottom=bottom, color=color)
        bottom += base_count

    ax2.set_title('Percentage distribution of costs in the BAU scenario')
    ax2.set_ylabel('Percentage')
    ax2.set_xlabel(f'$\\mathrm{{CO_2}}$  Emissions [{output_unit_BAU}]')
    ax2.margins(x=0.2)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

    plt.tight_layout()

    if save_fig:
        save_file_path = os.path.join(output_path, folder, 'Figures')
        if not os.path.exists(save_file_path):
            os.makedirs(save_file_path)
        plt.savefig(os.path.join(save_file_path, file_title), bbox_inches='tight')
        print(f"Saving Pareto front for {file_title} as {save_file_path}\n")

