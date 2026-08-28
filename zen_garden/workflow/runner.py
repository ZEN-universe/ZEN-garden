"""This function runs ZEN garden,it is executed in the __main__.py script.
Compilation  of the optimization problem.
"""

import logging
from pathlib import Path

from zen_garden.config import Config
from zen_garden.input.scenario_utils import ScenarioUtils
from zen_garden.model.schema import ModelSchema
from zen_garden.plugin_system.events import Event, EventPublisher
from zen_garden.plugin_system.loader import register_plugins
from zen_garden.utils.string_utils import StringUtils
from zen_garden.utils.utils import setup_logger
from zen_garden.workflow.optimization_workflow import OptimizationWorkflow

logger = logging.getLogger(__name__)


def prepare_scenarios(config: Config, job_index: list[int] | None):
    """Prepare selected scenarios and return their iterator and model name."""
    scenarios, elements = ScenarioUtils.get_scenarios(config, job_index)
    model_name, out_folder = StringUtils.setup_model_folder(
        config.analysis, config.system
    )
    ScenarioUtils.clean_scenario_folder(config, out_folder)
    return zip(scenarios, elements, strict=False), model_name


def run(
    config: str | Path = "./config.json",
    dataset: str | Path | None = None,
    job_index: list[int] | None = None,
    folder_output: str | Path | None = None,
    no_solve: bool = False,
    log_level: str | int = logging.INFO,
):
    """Run ZEN-garden.

    This function is the primary programmatic entry point for running
    ZEN-garden. When called, it reads the configuration, loads the model
    input data, constructs and solves the optimization problem, and saves
    the results.

    Args:
        config_obj (str | Path): Path to the configuration file
            (e.g. ``config.json``).
            If the file is located in the current working directory, the
            filename alone may be specified. Defaults to ``"./config.json"``.
        dataset (str | Path | None): Path to the folder containing the input dataset
            (e.g. ``"./1_base_case"``). If located in the current working
            directory, the folder name alone may be used. Defaults to the
            ``dataset`` value specified in the configuration file.
        folder_output (str | Path | None): Path to the folder where outputs will be
            saved.
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

    # Load configurations
    config_obj = Config.from_file(
        config, dataset_path=dataset, folder_output=folder_output
    )
    config_obj.validate_configurations()

    # Initialize the model schema. The schema is a blueprint of the
    # optimization problem, including a list of all elements and their parameters,
    # variables, and constraints. The schema is entirely conceptual,
    # nothing has been instantiated yet. Plugins can modify the schema to add new
    # elements, parameters, variables, and constraints.
    model_schema = ModelSchema(config_obj)

    # Register plugins. Plugins can modify the model schema and add new elements,
    # parameters, variables, and constraints
    register_plugins(config_obj.plugins)

    # Give plugins a hook to inspect or modify the freshly created schema before
    # any scenario is run.
    EventPublisher.trigger(Event.after_model_schema_creation, model_schema)

    logging.info(f"Optimizing for dataset {config_obj.analysis.dataset}")

    ## ITERATE THROUGH SCENARIOS
    scenarios, model_name = prepare_scenarios(config_obj, job_index)
    optimization_workflow = None
    for scenario, scenario_dict in scenarios:
        # FORMULATE THE OPTIMIZATION PROBLEM
        # add the scenario_dict and read input data
        optimization_workflow = OptimizationWorkflow(model_schema, scenario_dict)
        optimization_workflow.load_data()
        optimization_workflow.aggregate_time_series()
        optimization_workflow.run_steps(scenario, model_name, no_solve)

    logger.info("\n--- Optimization finished ---")
    return optimization_workflow
