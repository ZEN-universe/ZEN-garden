"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Alissa Ganter (aganter@ethz.ch),
              Davide Tonelli (davidetonelli@outlook.com),
              Jacob Mannhardt (jmannhardt@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Compilation  of the optimization problem.
"""
import importlib.util
import logging
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import shutil

import pkg_resources

from .model.optimization_setup import OptimizationSetup
from .postprocess.postprocess import Postprocess
from .preprocess.prepare import Prepare
from .utils import setup_logger, ScenarioDict

# we setup the logger here
setup_logger()


def main(config, dataset_path=None, job_index=None):
    """
    This function runs the compile.py script that was used in ZEN-Garden prior to the package build, it is executed
    in the __main__.py script

    :param config: A config instance used for the run
    :param dataset_path: If not None, used to overwrite the config.analysis["dataset"]
    :param job_index: The index of the scenario to run or a list of indices, if None, all scenarios are run in sequence
    """

    # print the version
    version = pkg_resources.require("zen_garden")[0].version
    logging.info(f"Running ZEN-Garden version: {version}")

    # prevent double printing
    logging.propagate = False

    # overwrite the path if necessary
    if dataset_path is not None:
        # logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis["dataset"] = dataset_path
    logging.info(f"Optimizing for dataset {config.analysis['dataset']}")
    # get the abs path to avoid working dir stuff
    config.analysis["dataset"] = os.path.abspath(config.analysis['dataset'])
    config.analysis["folder_output"] = os.path.abspath(config.analysis['folder_output'])

    ### System - load system configurations
    system_path = os.path.join(config.analysis['dataset'], "system.py")
    if not os.path.exists(system_path):
        raise FileNotFoundError(f"system.py not found in dataset: {config.analysis['dataset']}")
    spec = importlib.util.spec_from_file_location("module", system_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    system = module.system
    config.system.update(system)
    ### overwrite default system and scenario dictionaries
    if config.system["conduct_scenario_analysis"]:
        scenarios_path = os.path.abspath(os.path.join(config.analysis['dataset'], "scenarios.py"))
        if not os.path.exists(scenarios_path):
            raise FileNotFoundError(f"scenarios.py not found in dataset: {config.analysis['dataset']}")
        spec = importlib.util.spec_from_file_location("module", scenarios_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scenarios = module.scenarios
        config.scenarios.update(scenarios)
        # remove the default scenario if necessary
        if not config.system["run_default_scenario"] and "" in config.scenarios:
            del config.scenarios[""]

        # expand the scenarios
        config.scenarios = ScenarioDict.expand_lists(config.scenarios)

        # deal with the job array
        if job_index is not None:
            if isinstance(job_index, int):
                job_index = [job_index]
            else:
                job_index = list(job_index)
            logging.info(f"Running scenarios with job indices: {job_index}")


            # reduce the scenario and element to a single one
            scenarios = [list(config.scenarios.keys())[jx] for jx in job_index]
            elements = [list(config.scenarios.values())[jx] for jx in job_index]
        else:
            logging.info(f"Running all scenarios sequentially")
            scenarios = config.scenarios.keys()
            elements = config.scenarios.values()
    # Nothing to do with the scenarios
    else:
        scenarios = [""]
        elements = [{}]

    # get the name of the dataset
    model_name = os.path.basename(config.analysis["dataset"])
    if os.path.exists(out_folder := os.path.join(config.analysis["folder_output"], model_name)):
        logging.warning(f"The output folder '{out_folder}' already exists")
        if config.analysis["overwrite_output"]:
            logging.warning("Existing files will be overwritten!")

    # clean sub-scenarios if necessary
    if config.system["conduct_scenario_analysis"] and config.system["clean_sub_scenarios"]:
        # collect all paths that are in the scenario dict
        folder_dict = defaultdict(list)
        for key, value in config.scenarios.items():
            if value["sub_folder"] != "":
                folder_dict[f"scenario_{value['base_scenario']}"].append(f"scenario_{value['sub_folder']}")
                folder_dict[f"scenario_{value['base_scenario']}"].append(f"dict_all_sequence_time_steps_{value['sub_folder']}.h5")

        # compare to existing sub-scenarios
        for scenario_name, sub_folders in folder_dict.items():
            scenario_path = os.path.join(out_folder, scenario_name)
            if os.path.exists(scenario_path) and os.path.isdir(scenario_path):
                existing_sub_folder = os.listdir(scenario_path)
                for sub_folder in existing_sub_folder:
                    # delete the scenario subfolder
                    sub_folder_path = os.path.join(scenario_path, sub_folder)
                    if os.path.isdir(sub_folder_path) and sub_folder not in sub_folders:
                        logging.info(f"Removing sub-scenario {sub_folder}")
                        shutil.rmtree(sub_folder_path, ignore_errors=True)
                    # the time steps dict
                    if sub_folder.startswith("dict_all_sequence_time_steps") and sub_folder not in sub_folders:
                        logging.info(f"Removing time steps dict {sub_folder}")
                        os.remove(sub_folder_path)

    # iterate through scenarios
    for scenario, elements in zip(scenarios, elements):
        # create a dictionary with the paths to access the model inputs and check if input data exists
        prepare = Prepare(config)
        # check if all data inputs exist and remove non-existent
        prepare.check_existing_input_data()

        # FORMULATE THE OPTIMIZATION PROBLEM
        # add the elements and read input data
        optimization_setup = OptimizationSetup(config.analysis, prepare, scenario_dict=elements)
        # get rolling horizon years
        steps_optimization_horizon = optimization_setup.get_optimization_horizon()

        if scenario != "":
            additional_scenario_string = f"for scenario {scenario} "
        else:
            additional_scenario_string = ""
        #optimization_setup.restore_base_configuration(scenario,elements)  # per default scenario="" is used as base configuration. Use set_base_configuration(scenario, elements) if you want to change that
        #optimization_setup.overwrite_params(scenario, elements)
        # iterate through horizon steps
        for step_horizon in steps_optimization_horizon:
            if len(steps_optimization_horizon) == 1:
                logging.info(f"\n--- Conduct optimization for perfect foresight {additional_scenario_string}--- \n")
            else:
                logging.info(f"\n--- Conduct optimization for rolling horizon step {step_horizon} of {max(steps_optimization_horizon)} {additional_scenario_string}--- \n")
            # overwrite time indices
            optimization_setup.overwrite_time_indices(step_horizon)
            # create optimization problem
            optimization_setup.construct_optimization_problem()
            # SOLVE THE OPTIMIZATION PROBLEM
            optimization_setup.solve(config.solver)
            # break if infeasible
            if not optimization_setup.optimality:
                break
            # add newly capacity_addition of first year to existing capacity
            optimization_setup.add_new_capacity_addition(step_horizon)
            # add cumulative carbon emissions to previous carbon emissions
            optimization_setup.add_carbon_emission_cumulative(step_horizon)
            # EVALUATE RESULTS
            subfolder = Path("")
            scenario_name = None
            param_map = None
            output_scenarios = config.scenarios
            if config.system["conduct_scenario_analysis"]:
                # handle scenarios
                scenario_name = f"scenario_{scenario}"
                subfolder = Path(f"scenario_{elements['base_scenario']}")

                # set the scenarios
                if elements["sub_folder"] != "":
                    # get the param map
                    param_map = elements["param_map"]

                    # get the output scenarios
                    subfolder = subfolder.joinpath(f"scenario_{elements['sub_folder']}")
                    scenario_name = f"scenario_{elements['sub_folder']}"
                    output_scenarios = {}
                    for s, s_dict in config.scenarios.items():
                        if s_dict["base_scenario"] == elements["base_scenario"]:
                            out_dict = deepcopy(s_dict)
                            out_dict["base_scenario"] = s_dict["sub_folder"]
                            out_dict["sub_folder"] = ""
                            output_scenarios[s_dict["sub_folder"]] = out_dict
            # handle myopic foresight
            if len(steps_optimization_horizon) > 1:
                sf_string = str(subfolder)
                if sf_string != "":
                    sf_string += "_"
                sf_string += f"MF_{step_horizon}"
                subfolder = Path(sf_string)
            # write results
            _ = Postprocess(optimization_setup, scenarios=output_scenarios, subfolder=subfolder,
                            model_name=model_name, scenario_name=scenario_name, param_map=param_map)
    logging.info("Optimization finished")
    return optimization_setup
