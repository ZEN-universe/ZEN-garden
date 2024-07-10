"""
:Title:        ZEN-GARDEN check_PV
:Created:      June-2024
:Authors:      Jara Spate (jspaete@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich
"""
import os
import itertools
import pandas as pd
import math
import numpy as np
import re
from pprint import pprint

import zen_garden.postprocess.results.gdf_US_JS as gdf_US_JS
import zen_garden.postprocess.results.plot_results as plot_results
from zen_garden.postprocess.results.unit_handling_JS import EnergySystemUnits
from zen_garden.postprocess.results.results_JS2 import Results
from zen_garden.postprocess.results.results import Results as Results2
from zen_garden.postprocess.results.folder_structur_JS import create_folder, get_folder_path


def load_conversion_output_PV(res, scenario, node=None):
    """
    Load the conversion output of PV for a given scenario and node. The DataFrame is transposed and cleaned up.
    param: scenario: str: The scenario name
    param: node: str: The node name
    return: pd.DataFrame: The conversion output of PV
    """
    df_PV = res.get_full_ts("flow_conversion_output", "PV", scenario=scenario, node=node)
    # Reset index to ensure a clean state
    df_PV.reset_index(inplace=True)

    df_PV.drop(['carrier'], axis=1, inplace=True)

    #set node as the index
    df_PV.set_index('node', inplace=True)

    # Transpose the DataFrame
    df_PV_scen_node_transposed = df_PV.transpose()

    #delet the first row
    df_PV_scen_node_transposed = df_PV_scen_node_transposed[1:]

    # Reset index to turn the index into a column

    df_PV_scen_node_transposed.reset_index(inplace=True)
    # Rename columns
    df_PV_scen_node_transposed.rename(columns={'index': 'time'}, inplace=True)


    return df_PV_scen_node_transposed


def create_max_load_PV_adjusted_capacity(res1, res2, scenario, dir_max_load):
    """
    Create a DataFrame with the maximum load of the PV adjusted by the capacity of the PV. This represents the maximum possible production of PV.
    param res2: Results2 object
    param scenario: str, the scenario name
    param dir_max_load: str, the directory of the max_load_PV
    return: DataFrame with the maximum possible production of PV
    """
    # Load the max_load_PV
    max_load_PV = pd.read_csv(dir_max_load)

    # Get the system
    system = Results2.get_system(res2, scenario_name=scenario)

    # Filter for the total hours per year
    max_load_PV = max_load_PV[:system.total_hours_per_year]

    # Get the capacity of the PV and adjust it
    df_capacity_PV = res1.get_full_ts('capacity_addition', 'PV', scenario="scenario_")
    df_capacity_PV.reset_index(inplace=True)
    df_capacity_PV.rename(columns={'location': 'node', 0: 'capacity'}, inplace=True)
    df_capacity_PV.drop(['capacity_type'], axis=1, inplace=True)

    # Reformate max_load to merge with df_capacity_PV
    max_load_PV_melted = max_load_PV.melt(id_vars=['time'], var_name='node', value_name='load')

    # Merge max_load_PV with df_capacity_PV
    merged_df = pd.merge(max_load_PV_melted, df_capacity_PV, on='node')

    # Multiply the 'load' by 'capacity' to get the maximum possible production
    merged_df['adjusted_load'] = merged_df['load'] * merged_df['capacity']

    # Reformate the DataFrame to have the 'node' as columns
    result_df = merged_df.pivot(index='time', columns='node', values='adjusted_load').reset_index()

    return result_df

def create_df_comparison_in_out_PV(df_PV_in_adjusted, df_out_PV, save_csv=False):
    # Delete all columns in df_PV_in_adjusted and df_out_PV that are not present in both DataFrames
    common_columns = df_PV_in_adjusted.columns.intersection(df_out_PV.columns)
    df_PV_in_adjusted = df_PV_in_adjusted[common_columns]
    df_out_PV = df_out_PV[common_columns]

    # Set 'time' column as index for both DataFrames
    df_PV_in_adjusted.set_index('time', inplace=True)
    df_out_PV.set_index('time', inplace=True)

    # Sort columns in alphabetical order, with 'time' column in the first position
    df_out_PV = df_out_PV.reindex(sorted(df_out_PV.columns), axis=1)
    df_PV_in_adjusted = df_PV_in_adjusted.reindex(sorted(df_PV_in_adjusted.columns), axis=1)

    # Add 'input_' prefix to columns of df_PV_in_adjusted and 'output_' prefix to columns of df_out_PV
    df_PV_in_adjusted.columns = ['input_' + str(col) if col != 'time' else col for col in df_PV_in_adjusted.columns]
    df_out_PV.columns = ['output_' + str(col) if col != 'time' else col for col in df_out_PV.columns]

    # Merge df_PV_in_adjusted and df_out_PV, alternating the columns
    df_PV_max_load = pd.concat([df_PV_in_adjusted, df_out_PV], axis=1)

    # Generate a list of interleaved column names
    columns_max_load_PV = df_PV_in_adjusted.columns.tolist()
    columns_df_PV = df_out_PV.columns.tolist()
    assert len(columns_max_load_PV) == len(columns_df_PV), "DataFrames must have the same number of columns to interleave."
    interleaved_columns = [col for pair in zip(columns_max_load_PV, columns_df_PV) for col in pair]

    # Reindex the DataFrame to have interleaved columns
    df_PV_max_load = df_PV_max_load[interleaved_columns]

    if save_csv:
        df_PV_max_load.to_csv('comparsion_in_out_PV.csvs')

    return df_PV_max_load

def plot_comparison_PV(res_basic, res2, dir_max_load, scenario, node, start_hour, duration, save_csv=False):
    # Load the flow_conversion_output of the PV technology
    df_out_PV = load_conversion_output_PV(res_basic, scenario=scenario,node=None)
    # Load the max_load and adjust it with the installed capacity to get the maximum possible production per hour and node
    df_PV_in_adjusted = create_max_load_PV_adjusted_capacity(res_basic, res2, scenario="scenario_", dir_max_load=dir_max_load)
    # Create the comparison file
    df_PV_max_load = create_df_comparison_in_out_PV(df_PV_in_adjusted, df_out_PV, save_csv)
    #plot the dataset
    plot_results.plot_comparison_PV_in_out(df_PV_max_load, node, start_hour, duration)
