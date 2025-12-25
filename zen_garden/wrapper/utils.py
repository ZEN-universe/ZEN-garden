import os
import shutil
from pathlib import Path
import pandas as pd
import json
import numpy as np
from zen_garden.postprocess.results.results import Results
from zen_garden.preprocess.unit_handling import UnitHandling


def ensure_dir_exists(path: Path):
    """
    Ensure that a directory exists. If it doesn't, create it.

    Args:
        path (Path): The directory path to check and create.
    """
    if not path.exists():
        path.mkdir(parents=True)


def copy_file(src: Path, dest: Path):
    """
    Copy a single file from the source to the destination.

    Args:
        src (Path): The source file path.
        dest (Path): The destination file path.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source file {src} not found.")
    shutil.copy(src, dest)


def copy_dir(src: Path, dest: Path):
    """
    Copy an entire directory from the source to the destination.

    Args:
        src (Path): The source directory path.
        dest (Path): The destination directory path.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source directory {src} not found.")
    shutil.copytree(src, dest)


def remove_existing_dir(dest: Path):
    """
    Delete directory and all subdirectories if they exist.

    Args:
        dest (Path): Directory to be deleted.
    """
    # create new dataset for operational scenarios
    if os.path.exists(dest):
        shutil.rmtree(dest)


def copy_dataset(old_dataset: Path, new_dataset: Path, scenarios=None):
    """
    Copy the entire dataset from the old directory to a new directory.

    Args:
        old_dataset (Path): The path to the old dataset.
        new_dataset (Path): The path to the new dataset.
        scenarios (str, optional): A specific scenario file to copy. Defaults 
            to None.
    """
    remove_existing_dir(new_dataset)
    ensure_dir_exists(new_dataset)
    copy_dir(old_dataset / "energy_system", new_dataset / "energy_system")
    copy_dir(old_dataset / "set_carriers", new_dataset / "set_carriers")
    copy_dir(old_dataset / "set_technologies",
             new_dataset / "set_technologies")
    copy_file(old_dataset / "system.json", new_dataset / "system.json")

    if scenarios:
        copy_file(old_dataset / scenarios, new_dataset / "scenarios.json")


def load_results(out_dir: Path, scenario: str) -> dict:
    """
    Load simulation results from the specified directory and scenario.

    Args:
        out_dir (Path): Directory where the results are stored.
        scenario (str): Name of the scenario to load results for.

    Returns:
        dict: A dictionary containing various results data, such as capacity 
            addition, nodes, edges, and technologies.
    """
    r = Results(path=out_dir)

    assert 'capacity_addition' in r.get_component_names(
        'variable'), "Results have no variable named capacity addition"

    system = r.get_system()
    solver = r.get_solver()
    capacity_addition = r.get_total('capacity_addition',
                                    scenario_name=scenario)
    capacity_units = r.get_unit('capacity_addition', scenario_name=scenario)

    # Get conversion technologies excluding retrofitting
    set_conversion_not_retrofitting = list(
        set(system.set_conversion_technologies) -
        set(system.set_retrofitting_technologies))

    # Get edges from results
    edges = r.get_total('set_nodes_on_edges',
                        scenario_name=scenario).index.values

    # Reformat the results
    capacity_addition.columns.name = "year"
    capacity_addition = capacity_addition.stack().unstack("capacity_type")

    return {
        "capacity_addition": capacity_addition,
        "capacity_units": capacity_units,
        "system": system,
        "solver": solver,
        "nodes": system.set_nodes,
        "edges": edges,
        "technologies": {
            "set_conversion_technologies": set_conversion_not_retrofitting,
            "set_transport_technologies": system.set_transport_technologies,
            "set_storage_technologies": system.set_storage_technologies,
            "set_retrofitting_technologies":
            system.set_retrofitting_technologies
        }
    }


def get_element_location(element_name: str, raw_results: dict):
    """
    Get the location (nodes or edges) and the corresponding name for a given 
    element.

    Args:
        element_name (str): The name of the technology set (e.g., 
            'set_transport_technologies').
        raw_results (dict): The dictionary containing results data.

    Returns:
        tuple: A tuple containing the location (nodes or edges) and the 
            location name.
    """
    if element_name == "set_transport_technologies":
        location = raw_results["edges"]
        location_name = "edge"
    else:
        location = raw_results["nodes"]
        location_name = "node"
    return location, location_name


def get_element_folder(dataset_op: Path, element_name: str, tech: str) -> Path:
    """
    Get the folder path for a specific technology within a given element.

    Args:
        dataset_op (Path): The dataset output directory.
        element_name (str): The name of the technology set (e.g., 
            'set_conversion_technologies').
        tech (str): The name of the technology.

    Returns:
        Path: The path to the technology folder.
    """
    if element_name == "set_retrofitting_technologies":
        tech_folder_op = dataset_op / "set_technologies" / \
            "set_conversion_technologies" / element_name / tech
    else:
        tech_folder_op = dataset_op / "set_technologies" / element_name / tech
    return tech_folder_op


def format_capacity_addition(capacity_addition_tech: pd.DataFrame,
                             capacity_type: str,
                             suffix: str,
                             location_name: str) -> pd.DataFrame:
    """
    Format the capacity addition DataFrame for consistency in column names.

    Args:
        capacity_addition_tech (pd.DataFrame): The DataFrame with capacity 
            addition data.
        capacity_type (str): The type of capacity (e.g., 'power', 'energy').
        suffix (str): File suffix corresponding to capacity_type.
        location_name (str): The name of the location (either 'node' or 'edge').

    Returns:
        pd.DataFrame: The formatted DataFrame with the correct column names.
    """
    return capacity_addition_tech.rename(
        columns={
            "location": location_name,
            capacity_type: f"capacity_existing{suffix}",
            "year": "year_construction"
        })


def aggregate_capacity(capacity_existing: pd.DataFrame,
                       location_name: str) -> pd.DataFrame:
    """
    Aggregate capacity data by grouping it by location and year of construction.

    Args:
        capacity_existing (pd.DataFrame): The DataFrame with existing capacity 
            data.
        location_name (str): The name of the location (either 'node' or 'edge').

    Returns:
        pd.DataFrame: The aggregated DataFrame with summed capacities.
    """
    return capacity_existing.groupby([location_name, "year_construction"
                                      ]).sum().reset_index()


def save_capacity_existing(tech_folder_op: Path,
                           capacity_existing: pd.DataFrame, suffix: str):
    """
    Save the aggregated capacity data to a CSV file.

    Args:
        tech_folder_op (Path): The path to the technology folder.
        capacity_existing (pd.DataFrame): The aggregated capacity data.
        suffix (str): The suffix to append to the file name (e.g., '_energy').
    """
    capacity_existing.to_csv(tech_folder_op / f"capacity_existing{suffix}.csv",
                             mode='w',
                             header=True,
                             index=False)


def convert_to_original_units(capacity_addition_tech, capacity_units, capacity_type, unit_handling, tech, tech_folder_op, suffix):
    """
    Convert the capacity addition to the original units of the existing 
    capacity.

    Args:
        capacity_addition_tech (pd.DataFrame): The DataFrame containing capacity 
            addition data.
        capacity_units (pd.DataFrame): DataFrame containing unit information 
            for capacities.
        capacity_type (str): The type of capacity ('power' or 'energy').
        unit_handling (UnitHandling): The unit handling object for unit 
            conversions.
        tech (str): The technology name.
        tech_folder_op (Path): The path to the technology folder.
        suffix (str): Suffix to differentiate between 'power' and 'energy'.

    Returns:
        pd.DataFrame: The converted capacity addition data in the correct units.
    """
    # Get units for capacity addition
    capacity_addition_unit = capacity_units.loc[(tech, capacity_type)]

    # Ensure capacity addition is in base units
    if not np.isclose(unit_handling.get_unit_multiplier(capacity_addition_unit, tech), 1):
        raise AssertionError("Model output is not in base units")

    # Get capacity_existing units from attributes file
    fp_attributes = tech_folder_op / "attributes.json"
    with open(fp_attributes, 'r') as f:
        attributes = json.load(f)
        capacity_existing_unit = attributes[f'capacity_existing{suffix}']['unit']

    # Convert capacity addition to units of capacity_existing
    unit_multiplier = unit_handling.get_unit_multiplier(
        capacity_existing_unit, tech)
    capacity_addition_tech[f"capacity_existing{suffix}"] = capacity_addition_tech[
        f"capacity_existing{suffix}"] / unit_multiplier

    # Print output if necessary
    if not np.isclose(unit_multiplier, 1):
        print(
            f"Multiplying capacity addition (unit:{capacity_addition_unit}) "
            f"by a scale factor of {1/unit_multiplier} to convert to units "
            f"{capacity_existing_unit}"
        )

    return capacity_addition_tech


def round_capacity(results: dict, rounding_decimal_points: int, has_energy: bool) -> dict:
    """
    Round the capacities in the results to remove values below a certain 
    threshold.

    Args:
        results (dict): The dictionary containing results data.
        rounding_decimal_points (int): Number of decimal points after which to 
            round capacity values to zero. For example, if 
            ``rounding_decimal_points=6``, then all capacities below 10^-6 are 
            rounded to zero.
        has_energy (bool): Boolean whether the capacity addition has energy 
            column
    Returns:
        dict: The updated results dictionary with rounded capacity values.
    """
    capacity_addition = results["capacity_addition"]
    rounding_value = 10**(-rounding_decimal_points)
    idx_keep = capacity_addition["power"] > rounding_value

    if has_energy:
        idx_keep_energy = (
            capacity_addition["energy"] > rounding_value) | capacity_addition["energy"].isna()
        idx_keep = idx_keep | idx_keep_energy

    results["capacity_addition"] = capacity_addition.loc[idx_keep, :]

    return results


def add_capacity_additions(dataset_op: Path, results: dict, element_name: str,
                           capacity_type: str, unit_handling):
    """
    Transfer capacity additions from the results to the dataset for a given 
        element and capacity type.

    Args:
        dataset_op (Path): The output directory of the dataset.
        results (dict): The raw simulation results.
        element_name (str): The name of the technology set (e.g., 
            'set_conversion_technologies').
        capacity_type (str): The type of capacity ('power' or 'energy').
        unit_handling (UnitHandling): The unit handling object for unit 
            conversions.
    """
    print(f"Transferring capacity for {element_name}")
    location, location_name = get_element_location(element_name, results)
    elements = results["technologies"][element_name]
    capacity_addition = results["capacity_addition"]
    capacity_units = results["capacity_units"]

    for tech in elements:

        if tech not in capacity_addition.index.get_level_values(0):
            continue

        suffix = "" if capacity_type == "power" else "_energy"
        tech_folder_op = get_element_folder(dataset_op, element_name, tech)
        fp_capacity_existing = tech_folder_op / \
            f"capacity_existing{suffix}.csv"

        capacity_addition_tech = capacity_addition.loc[(
            tech, capacity_type)].reset_index()
        capacity_addition_tech = format_capacity_addition(
            capacity_addition_tech, capacity_type, suffix, location_name)

        capacity_addition_tech = convert_to_original_units(
            capacity_addition_tech, capacity_units, capacity_type,
            unit_handling, tech, tech_folder_op, suffix)

        # Read or initialize the 'capacity_existing' CSV
        if os.path.exists(fp_capacity_existing):
            capacity_existing = pd.read_csv(fp_capacity_existing,
                                            dtype={
                                                location_name:
                                                object,
                                                "year_construction":
                                                np.int64,
                                                f"capacity_existing{suffix}":
                                                np.float64
                                            })
            capacity_existing = pd.concat(
                [capacity_existing,
                 capacity_addition_tech]).reset_index(drop=True)
        else:
            capacity_existing = capacity_addition_tech

        # Aggregate capacity data
        capacity_existing = aggregate_capacity(capacity_existing,
                                               location_name)

        # Save updated data
        save_capacity_existing(tech_folder_op, capacity_existing, suffix)


def modify_json(file_path: Path, change_dict: dict):
    """
    Modify a JSON file according to a change dictionary.

    Args:
        file_path (Path): Path to the JSON file.
        change_dict (dict): Dictionary with attributes to change in the JSON 
            file.
    """
    with open(file_path, 'r+') as f:
        data = json.load(f)
        data.update(change_dict)  # Update dictionary with changes
        f.seek(0)  # Move cursor to the beginning of the file
        json.dump(data, f, indent=4)
        f.truncate()  # Remove leftover pieces if old file was longer


def capacity_addition_2_existing_capacity(out_dir: Path,
                                          dataset: Path,
                                          dataset_op: Path,
                                          scenario: str):
    """
    Add capacity additions from the simulation results to the existing 
    capacity dataset.

    Args:
        out_dir (Path): Directory of simulation outputs.
        dataset (Path): Original model dataset.
        dataset_op (Path): New model dataset to which to add capacity additions 
            as existing capacities.
        scenario (str): The scenario name to load.
        rounding_value (int, optional): Threshold for rounding capacity 
            additions to zero. Defaults to None.
    """
    # Load raw results
    results = load_results(out_dir, scenario)

    # Initialize unit handling class
    unit_handling = UnitHandling(
        dataset / "energy_system",
        results["solver"].rounding_decimal_points_units)

    has_energy = ("energy" in results["capacity_addition"].columns.values)
    # Round capacities below tolerance to zero
    results = round_capacity(results,
                             results["solver"].rounding_decimal_points_units,
                             has_energy
                             )

    # Add power capacity additions for different technology sets
    for element_name in results["technologies"].keys():
        add_capacity_additions(dataset_op, results, element_name,
                               "power", unit_handling)
    # add energy capacity additions if present
    if has_energy:
        add_capacity_additions(dataset_op, results, "set_storage_technologies",
                               "energy", unit_handling)
