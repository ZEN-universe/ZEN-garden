from zen_garden.postprocess.results.results import Results
from zen_garden.preprocess.unit_handling import UnitHandling
from pathlib import Path
import pandas as pd
import shutil
import os
import json
import numpy as np

def capacity_addition_2_existing_capacity(
    out_dir: Path | str ,
    dataset: Path | str,
    dataset_op: Path | str,
    scenario: str,
    rounding_value: int | None = None
):
    """
    out_dir ... directory of simulation_ouputs
    dataset_op ... new model dataset to which to add capacity additions as 
    existing capacities.
    """
    out_dir = Path(out_dir)
    dataset = Path(dataset)
    dataset_op = Path(dataset_op)

    # load results
    r = Results(path=out_dir)
    assert 'capacity_addition' in r.get_component_names(
        'variable'), "Results have no variable named capacity addition"
    capacity_addition_raw = r.get_total(
        'capacity_addition', 
        scenario_name=scenario
    )
    capacity_units_raw = r.get_unit(
        'capacity_addition',
        scenario_name=scenario
    )
    system = r.get_system()
    solver = r.get_solver()
    unit_handling = UnitHandling(dataset / "energy_system", solver.rounding_decimal_points_units)
    # add as capacities to input files of operations simulation
    #  Iterate over technology types: conversion, transport, storage
    technologies = {
        "set_conversion_technologies": system.set_conversion_technologies,
        "set_transport_technologies": system.set_transport_technologies,
        "set_storage_technologies": system.set_storage_technologies
    }
    for element_name, elements in technologies.items():
        print(f"Transferring capacity for {element_name}")
        if element_name == "set_transport_technologies":
            location = r.get_total('set_nodes_on_edges',
                                   scenario_name=scenario).index.values
            location_name = "edge"
        else:
            location = system.set_nodes
            location_name = "node"

        # reindex capacity addition so that it contains values for all combinations
        # of technologies and nodes
        index_full = pd.MultiIndex.from_product(
            [elements, ["power", "energy"], location],
            names=capacity_addition_raw.index.names)
        capacity_addition = capacity_addition_raw.reindex(
            index_full).sort_index()
        # Iterate over each technology in that type
        for tech in elements:
            if (element_name == "set_conversion_technologies"
                    and tech in system.set_retrofitting_technologies):
                tech_folder_op = (dataset_op / "set_technologies" /
                                  element_name /
                                  "set_retrofitting_technologies" / tech)
            else:
                tech_folder_op = (dataset_op / "set_technologies" /
                                  element_name / tech)
            
            capacity_addition_tech = capacity_addition.loc[(
                tech, "power")].stack(future_stack = True)
            capacity_addition_tech_energy = capacity_addition.loc[(
                tech, "energy")].stack(future_stack = True)

            # remove capacities below rounding threshold
            if rounding_value is not None:
                idx_keep = (
                    (capacity_addition_tech >= rounding_value) & 
                    ((capacity_addition_tech_energy >= rounding_value) 
                    | (capacity_addition_tech_energy.isna()))
                )
                capacity_addition_tech = capacity_addition_tech.loc[
                    idx_keep, :]
                capacity_addition_tech_energy = capacity_addition_tech_energy.loc[
                    idx_keep, :]
                
            dat_dict = {
                "power": capacity_addition_tech,
                "energy": capacity_addition_tech_energy
            }
            # Repeat for power capacity and energy capacity separately
            for (typ, capacity_addition_tech) in dat_dict.items():
                
                if typ == "power":
                    suffix = ""
                else:
                    suffix = "_energy"

                capacity_addition_tech = (
                    capacity_addition_tech.dropna().reset_index())

                # skip if no capacity additions
                if capacity_addition_tech.shape[0] == 0:
                    continue

                # rename columns
                capacity_addition_tech = (capacity_addition_tech.rename(
                    columns={
                        'location': location_name,
                        'level_1': 'year_construction',
                        0: 'capacity_existing' + suffix
                    }))
                
                # get units
                capacity_addition_unit = capacity_units_raw.loc[(tech, typ)]

                # load existing file
                fp_capacity_existing = (
                    tech_folder_op / ("capacity_existing" + suffix + ".csv"))
                fp_attributes = (tech_folder_op / "attributes.json")
                with open(fp_attributes, 'r') as f:
                    attributes = json.load(f)
                    capacity_existing_unit = (
                        attributes['capacity_existing']['unit'])
                
                # convert to base units
                unit_multiplier = (
                    unit_handling.get_unit_multiplier(capacity_existing_unit, tech)
                )
                capacity_addition_tech["capacity_existing" + suffix] = (
                    capacity_addition_tech["capacity_existing" + suffix]
                    /unit_multiplier 
                )

                if not np.isclose(unit_multiplier, 1):
                    print(
                        f"Multiplying capacity addition (unit:{capacity_addition_unit}) " \
                        f"by a scale factor of {1/unit_multiplier} to convert to units " \
                        f"{capacity_existing_unit}"
                    )


                if os.path.exists(fp_capacity_existing):
                    capacity_existing = pd.read_csv(fp_capacity_existing,
                                                    dtype={
                                                        "node":
                                                        object,
                                                        "year_construction":
                                                        np.int64,
                                                        "capacity_existing":
                                                        np.float64
                                                    })

                    # merge
                    capacity_existing = pd.concat(
                        [capacity_existing,
                         capacity_addition_tech]).reset_index(drop=True)
                else:
                    capacity_existing = capacity_addition_tech

                # sum capacities
                capacity_existing = capacity_existing.groupby(
                    by=[location_name, "year_construction"
                        ]).sum().reset_index()

                # save
                capacity_existing.to_csv(
                    tech_folder_op / ("capacity_existing" + suffix + ".csv"),
                    mode='w',
                    header=True,
                    index=False)


def modify_json(file_path, change_dict):
    """
    Modify a json file according to a change dictionary
    """
    with open(file_path, 'r+') as f:
        data = json.load(f)

        # write new attribute
        for attr, value in change_dict.items():
            data[attr] = value  # <--- set attribute value

        # write file
        f.seek(0)  # move cursor to beginning of file
        json.dump(data, f, indent=4)
        f.truncate()  # remove leftover pieces if old file was longer


def copy_dataset(old_dataset, new_dataset, scenarios=None):

    # create new dataset for operational scenarios
    if os.path.exists(new_dataset):
        shutil.rmtree(new_dataset)
    os.makedirs(new_dataset)
    shutil.copytree(
        Path(old_dataset) / "energy_system",
        Path(new_dataset) / "energy_system"
    )
    shutil.copytree(
        Path(old_dataset) / "set_carriers",
        Path(new_dataset) / "set_carriers"
    )
    shutil.copytree(
        Path(old_dataset) / "set_technologies",
        Path(new_dataset) / "set_technologies"
    )
    shutil.copyfile(
        Path(old_dataset) / "system.json", 
        Path(new_dataset) / "system.json"
    )
    if scenarios is not None:
        shutil.copyfile(
            Path(old_dataset) / scenarios,
            Path(new_dataset) / "scenarios.json"
        )
