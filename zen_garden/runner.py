"""
This function runs ZEN garden,it is executed in the __main__.py script.
Compilation  of the optimization problem.
"""
import importlib.util
from pathlib import Path
import logging
import os
import importlib
from .optimization_setup import OptimizationSetup
from .postprocess.postprocess import Postprocess
from .utils import setup_logger, InputDataChecks, StringUtils, ScenarioUtils
import zen_garden.default_config as default_config
import json
import warnings

# we setup the logger here
setup_logger()


def run(config = "./config.json", dataset=None, job_index=None, 
               folder_output=None):
    """
    Run ZEN-garden.

    This function is the primary programmatic entry point for running
    ZEN-garden. When called, it reads the configuration, loads the model
    input data, constructs and solves the optimization problem, and saves
    the results.

    Args:
        config (str): Path to the configuration file (e.g. ``config.json``).
            If the file is located in the current working directory, the
            filename alone may be specified. Defaults to ``"./config.json"``.
        dataset (str): Path to the folder containing the input dataset
            (e.g. ``"./1_base_case"``). If located in the current working
            directory, the folder name alone may be used. Defaults to the
            ``dataset`` value specified in the configuration file.
        folder_output (str): Path to the folder where outputs will be saved.
            Defaults to ``"./outputs"``.
        job_index (list[int] | None): Indices of jobs (scenarios) to run.
            For example, ``job_index=[1]`` runs only the first scenario.
            Defaults to ``None`` (run all jobs).

    Returns:
        OptimizationSetup: The fully set up and solved optimization problem.

    Examples:
        >>> from zen_garden import run, download_example_dataset
        >>> download_example_dataset("1_base_case")
        >>> run("1_base_case")
    """

    # print the version
    version = importlib.metadata.version("zen-garden")
    logging.info(f"Running ZEN-garden version: {version}")

    # prevent double printing
    logging.propagate = False

    ### import the config
    if not os.path.exists(config):
        config = config.replace(".py", ".json")
    config_path, config_file = os.path.split(os.path.abspath(config))
    if config_file.endswith(".py"):
        spec = importlib.util.spec_from_file_location("module", Path(config_path) / config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.config
        warnings.warn(
            "Use of the `config.py` file is deprecated and will be removed " \
            "in ZEN-garden v3.0.0. Please switch to using a `config.json` " \
            "file instead.",
            DeprecationWarning,
            stacklevel=2
        )
    else:
        with open(Path(config_path) / config_file, "r") as f:
            config = default_config.Config(**json.load(f))

    # overwrite the path if necessary
    if dataset is not None:
        # logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis.dataset = dataset
    if folder_output is not None:
        config.analysis.folder_output = os.path.abspath(folder_output)
        config.solver.solver_dir = os.path.abspath(folder_output)
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
    scenarios, elements = ScenarioUtils.get_scenarios(config, job_index)
    # get the name of the dataset
    model_name, out_folder = StringUtils.setup_model_folder(config.analysis, config.system)
    # clean sub-scenarios if necessary
    ScenarioUtils.clean_scenario_folder(config, out_folder)
    ### ITERATE THROUGH SCENARIOS
    for scenario, scenario_dict in zip(scenarios, elements):
        # FORMULATE THE OPTIMIZATION PROBLEM
        # add the scenario_dict and read input data
        optimization_setup = OptimizationSetup(config, scenario_dict=scenario_dict, input_data_checks=input_data_checks)
        # get rolling horizon years
        steps_horizon = optimization_setup.get_optimization_horizon()
        # iterate through horizon steps
        for step in steps_horizon:
            # iterate through phases
            for phase in ['investment', 'operation']:
                #if operation phase, exclude capacity expansion
                if phase == 'operation' and not config.system.include_operation_only_phase:
                    continue
                StringUtils.print_optimization_progress(scenario, steps_horizon, step, system=config.system)
                if optimization_setup.system.include_operation_only_phase:
                    optimization_setup.set_phase_configurations(phase)
                # overwrite time indices
                optimization_setup.overwrite_time_indices(step)
                # create optimization problem
                optimization_setup.construct_optimization_problem()
                if optimization_setup.solver.use_scaling:
                    optimization_setup.scaling.run_scaling()
                elif optimization_setup.solver.analyze_numerics or optimization_setup.solver.run_diagnostics:
                    optimization_setup.scaling.analyze_numerics()
                # SOLVE THE OPTIMIZATION PROBLEM
                optimization_setup.solve()
                # break if infeasible
                if not optimization_setup.optimality:
                    # write IIS
                    optimization_setup.write_IIS(scenario)
                    logging.warning(f"Optimization: {optimization_setup.model.termination_condition}")
                    break
                if optimization_setup.solver.use_scaling:
                    optimization_setup.scaling.re_scale()
                # save new capacity additions and cumulative carbon emissions for next time step
                optimization_setup.add_results_of_optimization_step(step)
                # EVALUATE RESULTS
                # create scenario name, subfolder and param_map for postprocessing
                scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                    config=config, scenario=scenario, scenario_dict=scenario_dict, steps_horizon=steps_horizon, step=step)
                # write results
                Postprocess(optimization_setup, scenarios=config.scenarios, subfolder=subfolder,
                            model_name=model_name, scenario_name=scenario_name, param_map=param_map)
    logging.info("--- Optimization finished ---")
    return optimization_setup