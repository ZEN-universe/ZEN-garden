"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
              Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
==========================================================================================================================================================================="""
import os
import sys
import logging
import importlib.util
import pkg_resources

from shutil import rmtree

from .preprocess.prepare import Prepare
from .model.optimization_setup import OptimizationSetup
from .postprocess.postprocess import Postprocess


def main(config, dataset_path=None):
    """
    This function runs the compile.py script that was used in ZEN-Garden prior to the package build, it is executed
    in the __main__.py script
    :param config: A config instance used for the run
    :param dataset_path: If not None, used to overwrite the config.analysis["dataset"]
    """
    # SETUP LOGGER
    log_format = '%(asctime)s %(filename)s: %(message)s'
    log_path = os.path.join('outputs', 'logs')
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_path, 'valueChain.log'), level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logging.captureWarnings(True)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    # print the version
    version = pkg_resources.require("zen_garden")[0].version
    logging.info(f"Running ZEN-Garden version: {version}")

    # prevent double printing
    logging.propagate = False

    # overwrite the path if necessary
    if dataset_path is not None:
        logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis["dataset"] = dataset_path
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

    # create a dictionary with the paths to access the model inputs and check if input data exists
    prepare = Prepare(config)
    # check if all data inputs exist and remove non-existent
    prepare.check_existing_input_data()

    # FORMULATE THE OPTIMIZATION PROBLEM
    # add the elements and read input data
    optimization_setup = OptimizationSetup(config.analysis, prepare)
    # get rolling horizon years
    steps_optimization_horizon = optimization_setup.get_optimization_horizon()

    # get the name of the dataset
    model_name = os.path.basename(config.analysis["dataset"])
    if os.path.exists(out_folder := os.path.join(config.analysis["folder_output"], model_name)):
        logging.warning(f"The output folder '{out_folder}' already exists")
        if config.analysis["overwrite_output"]:
            logging.warning("Existing files will be overwritten!")

    # update input data
    for scenario, elements in config.scenarios.items():
        optimization_setup.restore_base_configuration(scenario,elements)  # per default scenario="" is used as base configuration. Use set_base_configuration(scenario, elements) if you want to change that
        optimization_setup.overwrite_params(scenario, elements)
        # iterate through horizon steps
        for step_horizon in steps_optimization_horizon:
            if len(steps_optimization_horizon) == 1:
                logging.info("\n--- Conduct optimization for perfect foresight --- \n")
            else:
                logging.info(f"\n--- Conduct optimization for rolling horizon step {step_horizon} of {max(steps_optimization_horizon)}--- \n")
            # overwrite time indices
            optimization_setup.overwrite_time_indices(step_horizon)
            # create optimization problem
            optimization_setup.construct_optimization_problem()
            # SOLVE THE OPTIMIZATION PROBLEM
            optimization_setup.solve(config.solver)
            # add newly built_capacity of first year to existing capacity
            optimization_setup.add_newly_built_capacity(step_horizon)
            # add cumulative carbon emissions to previous carbon emissions
            optimization_setup.add_carbon_emission_cumulative(step_horizon)
            # EVALUATE RESULTS
            subfolder = ""
            scenario_name = None
            if config.system["conduct_scenario_analysis"]:
                # handle scenarios
                subfolder += f"scenario_{scenario}"
                scenario_name = subfolder
            # handle myopic foresight
            if len(steps_optimization_horizon) > 1:
                if subfolder != "":
                    subfolder += f"_"
                subfolder += f"MF_{step_horizon}"
            # write results
            evaluation = Postprocess(optimization_setup, scenarios=config.scenarios, subfolder=subfolder,
                                     model_name=model_name, scenario_name=scenario_name)
    return optimization_setup
