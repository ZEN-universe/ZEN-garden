# File: state_mapping.py
"""
:Title: State Mapping Module
:Created: April-2024
:Authors: Jara Späte (jspaete@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

This module changes the state information in the DataFrame according to the specified mapping direction.
The mapping is based on a JSON file that contains the mappings between full state names, state abbreviations, and state numbers.
The module also prints out the states that were not successfully mapped.
The available mapping directions are 'full_to_abbr', 'full_to_number', and 'number_to_abbr'.

Example mappings:
- "full_to_abbr": "Alabama": "AL",
- "full_to_number": "Alabama": 1,
- "number_to_abbr": "1": "AL",

"""

import json
<<<<<<< HEAD
import os

DIRECTORY = '../zen_garden/postprocess/results/'
FILENAME = 'state_mappings_JS.json'
FILE_DIR = os.path.join(DIRECTORY, FILENAME)
=======


>>>>>>> 14e32f15 (Add postprocess results and unit handling scripts)
def print_unmapped_states(df, column_name, column_name_mapped):
    """
    Print out the states that were not successfully mapped.

    :param df: DataFrame with the data
    :param column_name: Name of the column containing state information
    :return: None
    """
    unmapped_states = df[df[column_name_mapped].isna()][column_name].unique()
    print("States not mapped:")
    for state in unmapped_states:
        print(state)

def get_state_mappings(mapping_direction):
    """
    Get the state mappings from the JSON file based on the specified mapping direction.

    :param mapping_direction: Direction of mapping (e.g., 'full_to_abbr')
    :return: Dictionary containing the state mappings
    :raises ValueError: If mapping_direction is not available
    """
<<<<<<< HEAD
    with open(FILE_DIR, 'r') as file:
=======
    with open('state_mappings.json', 'r') as file:
>>>>>>> 14e32f15 (Add postprocess results and unit handling scripts)
        state_mappings = json.load(file)

    available_mappings = ['full_to_abbr', 'full_to_number', 'number_to_abbr']
    if mapping_direction not in available_mappings:
        raise ValueError(f"Mapping direction {mapping_direction} not available. Please choose from {available_mappings}")

    return state_mappings[mapping_direction]

def mapping(df, column_name, mapping_direction):
    """
    Map states in the DataFrame according to the specified mapping direction.

    :param df: DataFrame containing state information
    :param column_name: Name of the column containing state information
    :param mapping_direction: Direction of mapping (e.g., 'full_to_abbr')
    :return: DataFrame with mapped states
    :raises ValueError: If mapping_direction is not available
    """
<<<<<<< HEAD
    with open(FILE_DIR, 'r') as file:
=======
    with open('state_mappings.json', 'r') as file:
>>>>>>> 14e32f15 (Add postprocess results and unit handling scripts)
        state_mappings = json.load(file)

    available_mappings = ['full_to_abbr', 'full_to_number', 'number_to_abbr']
    if mapping_direction not in available_mappings:
        raise ValueError(f"Mapping direction {mapping_direction} not available. Please choose from {available_mappings}")

    state_map = state_mappings[mapping_direction]
    column_name_mapped = column_name + '_mapped'

    # Map the values and insert the mapped column at the first position
    df.insert(0, column_name_mapped, df[column_name].map(state_map))
    print_unmapped_states(df, column_name, column_name_mapped)
    df.dropna(subset=[column_name_mapped], inplace=True)
    df.drop(column_name, axis=1, inplace=True)
    return df

def reverse_mapping(df, column_in, column_out, mapping_direction):
    """
    Reverse map states in the DataFrame according to the specified mapping direction.

    param df: DataFrame containing state information
    param column_in: Name of the column containing state information
    param column_out: Name of the column containing the mapped state information
    param mapping_direction: Direction of mapping (e.g., 'full_to_abbr')
    return: DataFrame with mapped states
    """
    # get current directory
<<<<<<< HEAD
    with open(FILE_DIR, 'r') as file:
=======
    with open('results_models/state_mappings.json', 'r') as file:
>>>>>>> 14e32f15 (Add postprocess results and unit handling scripts)
        state_mappings = json.load(file)

    available_mappings = ['full_to_abbr', 'full_to_number', 'number_to_abbr']
    if mapping_direction not in available_mappings:
        raise ValueError(f"Mapping direction {mapping_direction} not available. Please choose from {available_mappings}")

    state_map = state_mappings[mapping_direction]

    state_map_reverse = {abbr: full for full, abbr in state_map.items()}


    # Map the values and insert the mapped column at the first position
    df.insert(0, column_out, df[column_in].map(state_map_reverse))
    print_unmapped_states(df, column_in, column_out)
    df.dropna(subset=[column_out], inplace=True)
    return df