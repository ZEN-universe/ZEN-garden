"""This function runs ZEN garden,it is executed in the __main__.py script.
Compilation  of the optimization problem.
"""

import importlib
import importlib.util
import json
import logging
import warnings
from importlib.metadata import version
from pathlib import Path

from zen_garden.default_config import Config
from zen_garden.optimization_setup import OptimizationSetup
from zen_garden.plugin_system.loader import register_plugins
from zen_garden.utils import InputDataChecks, ScenarioUtils, StringUtils, setup_logger

logger = logging.getLogger(__name__)


def import_config(config_path) -> Config:
    # Inform the user about the deprecation of `config.py` instead of renaming the file
    # if not os.path.exists(config_path):
    #     config = config_path.replace(".py", ".json")

    # Use pathlib instead of os.path
    config_path = Path(config_path)

    # Handle side-cases with early returns and early warnings
    if config_path.suffix == ".py":
        warnings.warn(
            "Use of the `config.py` file is deprecated and will be removed "
            "in ZEN-garden v3.0.0. Please switch to using a `config.json` "
            "file instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        spec = importlib.util.spec_from_file_location("module", config_path)
        assert spec is not None  # add more specific types for more information
        module = importlib.util.module_from_spec(spec)
        assert module is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module.config

    with open(config_path, "r") as f:
        json_config = json.load(f)

    return Config(**json_config)


def adjust_config_paths(
    config: Config, dataset, folder_output: str | None, config_path: str
) -> None:
    # overwrite the path if necessary
    if dataset is not None:
        # logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis.dataset = dataset

    config_dir = Path(config_path).parent

    if folder_output is not None:
        if not Path(folder_output).is_absolute():
            folder_output = str((config_dir / folder_output).resolve())
        config.analysis.folder_output = folder_output
        config.solver.solver_dir = folder_output
    logging.info(f"Optimizing for dataset {config.analysis.dataset}")
    # make all paths absolute to the config file path
    if not Path(config.analysis.dataset).is_absolute():
        config.analysis.dataset = str((config_dir / config.analysis.dataset).resolve())
    if not Path(config.analysis.folder_output).is_absolute():
        config.analysis.folder_output = str(
            (config_dir / config.analysis.folder_output).resolve()
        )
    if not Path(config.solver.solver_dir).is_absolute():
        config.solver.solver_dir = str(
            Path(config_dir / config.solver.solver_dir).resolve()
        )
    config.analysis.zen_garden_version = version("zen-garden")


def prepare_scenarios(config: Config, job_index: list[int] | None):
    scenarios, elements = ScenarioUtils.get_scenarios(config, job_index)
    model_name, out_folder = StringUtils.setup_model_folder(
        config.analysis, config.system
    )
    ScenarioUtils.clean_scenario_folder(config, out_folder)
    return zip(scenarios, elements, strict=False), model_name


def run(
    config="./config.json",
    dataset=None,
    job_index=None,
    folder_output: str | None = None,
    no_solve: bool = False,
    log_level: str | int = logging.INFO,
):
    """Run ZEN-garden.

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
        no_solve (bool): If ``True``, the optimization problem will be
            constructed but not solved. Defaults to ``False``.
        log_level (str | int): Logging level. Can be specified as a string
            (e.g. ``"INFO"``, ``"DEBUG"``, etc.) or as an integer
            (e.g. ``logging.INFO``, ``logging.DEBUG``, etc.).
            Defaults to ``logging.INFO``.

    Returns:
        OptimizationSetup: The fully set up and solved optimization problem.

    Examples:
        >>> from zen_garden import run, download_example_dataset
        >>> download_example_dataset("1_base_case")
        >>> run("1_base_case")
    """
    setup_logger(log_level)

    config_path = config
    config = import_config(config_path)
    register_plugins(config.plugins)
    adjust_config_paths(config, dataset, folder_output, config_path)

    ### SYSTEM CONFIGURATION
    input_data_checks = InputDataChecks(config=config)
    input_data_checks.check_dataset()
    input_data_checks.read_system_file(config)
    input_data_checks.check_technology_selections()
    input_data_checks.check_year_definitions()

    ## ITERATE THROUGH SCENARIOS
    scenarios, model_name = prepare_scenarios(config, job_index)
    optimization_setup = None
    for scenario, scenario_dict in scenarios:
        # FORMULATE THE OPTIMIZATION PROBLEM
        # add the scenario_dict and read input data
        optimization_setup = OptimizationSetup(config, scenario_dict, input_data_checks)
        # get rolling horizon years
        steps_horizon = optimization_setup.get_optimization_horizon()
        # iterate through horizon steps
        for step in steps_horizon:
            StringUtils.print_optimization_progress(
                scenario, steps_horizon, step, system=config.system
            )
            # overwrite time indices
            optimization_setup.overwrite_time_indices(step)
            # create optimization problem
            optimization_setup.construct_optimization_problem()
            optimization_setup.prepare_scaling()

            if no_solve:
                logger.info(
                    "Optimization problem constructed but not solved "
                    "(no_solve=True). Continue with next iteration."
                )
                continue

            # SOLVE THE OPTIMIZATION PROBLEM
            optimization_setup.solve(scenario)

            # break if infeasible
            if not optimization_setup.optimality:
                # write IIS
                optimization_setup.write_IIS(scenario)
                assert optimization_setup.zen_model is not None
                logger.warning(
                    f"Optimization: "
                    f"{optimization_setup.zen_model.lp_model.termination_condition}"
                )
                break

            optimization_setup.re_scale()
            optimization_setup.add_results_of_optimization_step(step)

            # EVALUATE RESULTS
            scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
                config=config,
                scenario=scenario,
                scenario_dict=scenario_dict,
                steps_horizon=steps_horizon,
                step=step,
            )
            optimization_setup.write_results(
                scenarios=config.scenarios,
                subfolder=subfolder,
                model_name=model_name,
                scenario_name=scenario_name,
                param_map=param_map,
            )

    logger.info("\n--- Optimization finished ---")
    return optimization_setup
