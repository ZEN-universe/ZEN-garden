"""
This function runs ZEN garden,it is executed in the __main__.py script.
Compilation  of the optimization problem.
"""
import cProfile
import importlib.util
import logging
import os
from collections import defaultdict
import importlib

from .model.optimization_setup import OptimizationSetup
from .postprocess.postprocess import Postprocess
from .utils import setup_logger, InputDataChecks, StringUtils, ScenarioUtils, OptimizationError
from .preprocess.unit_handling import Scaling

# we setup the logger here
setup_logger()


def main(config, dataset_path=None, job_index=None, folder_output_path=None):
    """
    This function runs ZEN garden,
    it is executed in the __main__.py script

    :param config: A config instance used for the run
    :param dataset_path: If not None, used to overwrite the config.analysis.dataset
    :param job_index: The index of the scenario to run or a list of indices, if None, all scenarios are run in sequence
    :param folder_output_path: If not None, used to overwrite the config.analysis.folder_output
    """

    # print the version
    version = importlib.metadata.version("zen-garden")
    logging.info(f"Running ZEN-garden version: {version}")

    # prevent double printing
    logging.propagate = False

    # overwrite the path if necessary
    if dataset_path is not None:
        # logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis.dataset = dataset_path
    if folder_output_path is not None:
        config.analysis.folder_output = folder_output_path
    logging.info(f"Optimizing for dataset {config.analysis.dataset}")
    # get the abs path to avoid working dir stuff
    config.analysis.dataset = os.path.abspath(config.analysis.dataset)
    config.analysis.folder_output = os.path.abspath(config.analysis.folder_output)
    config.analysis.zen_garden_version = version
    ### SYSTEM CONFIGURATION
    input_data_checks = InputDataChecks(config=config, optimization_setup=None)
    input_data_checks.check_dataset()
    input_data_checks.read_system_file(config)
    input_data_checks.check_technology_selections()
    input_data_checks.check_year_definitions()
    # overwrite default system and scenario dictionaries
    scenarios,elements = ScenarioUtils.get_scenarios(config,job_index)
    # get the name of the dataset
    model_name, out_folder = StringUtils.setup_model_folder(config.analysis,config.system)
    # clean sub-scenarios if necessary
    ScenarioUtils.clean_scenario_folder(config,out_folder)
    ### ITERATE THROUGH SCENARIOS
    for scenario, scenario_dict in zip(scenarios, elements):
        # FORMULATE THE OPTIMIZATION PROBLEM
        # add the scenario_dict and read input data
        optimization_setup = OptimizationSetup(config, scenario_dict=scenario_dict, input_data_checks=input_data_checks)
        # get rolling horizon years
        steps_horizon = optimization_setup.get_optimization_horizon()
        # iterate through horizon steps
        for step in steps_horizon:
            StringUtils.print_optimization_progress(scenario,steps_horizon,step,system=config.system)
            # overwrite time indices
            optimization_setup.overwrite_time_indices(step)
            # create optimization problem
            optimization_setup.construct_optimization_problem()
            if config.solver.use_scaling:
                optimization_setup.scaling.run_scaling()
            elif config.solver.analyze_numerics:
                optimization_setup.scaling.analyze_numerics()
            # SOLVE THE OPTIMIZATION PROBLEM
            optimization_setup.solve()
            # break if infeasible
            if not optimization_setup.optimality:
                # write IIS
                optimization_setup.write_IIS()
                raise OptimizationError(optimization_setup.model.termination_condition)
            if config.solver.use_scaling:
                optimization_setup.scaling.re_scale()
            # save new capacity additions and cumulative carbon emissions for next time step
            optimization_setup.add_results_of_optimization_step(step)
            # EVALUATE RESULTS
            # create scenario name, subfolder and param_map for postprocessing
            scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                config = config,scenario = scenario,scenario_dict=scenario_dict,steps_horizon=steps_horizon,step=step
            )
            # write results
            Postprocess(optimization_setup, scenarios=config.scenarios, subfolder=subfolder,
                            model_name=model_name, scenario_name=scenario_name, param_map=param_map)
    logging.info("--- Optimization finished ---")
    return optimization_setup
