"""Class defining the optimization model.

The class takes as inputs the properties of the optimization problem. The
properties are saved in the dictionaries analysis and system which are passed
to the class. After initializing the model, the class adds carriers and
technologies to the model and returns it. The class also includes a method to
solve the optimization problem.
"""

import copy
import logging
from pathlib import Path

from zen_garden.model.time_steps import TimeStepsDicts
from zen_garden.model.zen_model import ZenModel
from zen_garden.optimization_step import OptimizationStep
from zen_garden.preprocess.time_series_aggregation import TimeSeriesAggregation
from zen_garden.preprocess.unit_handling import UnitHandling
from zen_garden.services.dataset_path_resolver import DatasetPathResolver
from zen_garden.services.element_registry import ElementRegistry
from zen_garden.services.input_repository import InputRepository
from zen_garden.services.network_topology import NetworkTopology
from zen_garden.services.parameter_loading_service import ParameterLoadingService
from zen_garden.services.scenario_dict import ScenarioDict
from zen_garden.services.service_container import ServiceContainer
from zen_garden.topology.model_schema import ModelSchema
from zen_garden.types import YearSpecificTs
from zen_garden.utils.input_data_checks import InputDataChecks

logger = logging.getLogger(__name__)


class OptimizationWorkflow:
    """Class defining the optimization model.

    The constructor only stores its inputs. The workflow then runs in explicit
    stages: :meth:`load_data` builds the services and loads all input data,
    :meth:`aggregate_time_series` conducts the time series aggregation, and
    :meth:`run_steps` builds and solves the optimization problem.
    """

    zen_model: ZenModel
    service_container: ServiceContainer

    def __init__(
        self,
        model_schema: ModelSchema,
        init_scenario_dict: dict,
    ):
        """Store the inputs for the optimization of the energy system.

        The constructor only records its arguments. Call :meth:`load_data` and
        :meth:`aggregate_time_series` (in that order) to build the services and
        load the input data before running :meth:`run_steps`.

        Args:
            model_schema (ModelSchema): Schema describing the optimization
                problem, exposing the canonical configuration.
            init_scenario_dict (dict): Dictionary defining the scenario, including
                data such as resources, demand, etc.

        """
        self.model_schema = model_schema
        self.init_scenario_dict = init_scenario_dict
        self.service_container = ServiceContainer("service_container")

    def load_data(self) -> None:
        """Build the services and load all input data for the optimization.

        Sets up the service container, resolves dataset paths, applies the
        scenario, validates the input data, registers all elements and loads
        every input parameter. Must be called before
        :meth:`aggregate_time_series` and :meth:`run_steps`.
        """
        # work on a private copy so the shared input schema is left untouched
        self.model_schema = copy.deepcopy(self.model_schema)
        self.service_container.register("model_schema", self.model_schema)

        self.dataset_path_resolver = self.service_container.build_and_register(
            "dataset_path_resolver", DatasetPathResolver
        )

        # dict to update elements according to scenario
        # WARNING: ScenarioDict::__init__ updates the config object!
        # Hence, input_data_checks must be initialized after ScenarioDict
        scenario_dict = ScenarioDict(
            self.init_scenario_dict,
            self.dataset_path_resolver,
            self.model_schema,
        )
        self.service_container.register("scenario_dict", scenario_dict)

        # Input data checks validate the dataset's folder structure and technology
        # data, and are registered for later injection into elements.
        # NOTE: created here (rather than passed in) because they depend on the
        # deep-copied model schema and the dataset path resolver built above.
        input_data_checks = InputDataChecks(model_schema=self.model_schema)
        input_data_checks.dataset_path_resolver = self.dataset_path_resolver
        # check if input data exists
        input_data_checks.check_primary_folder_structure()
        # check if all needed data inputs for the chosen technologies exist and
        # remove non-existent inputs
        # WARNING: This function modifies the config object!
        input_data_checks.check_existing_technology_data()
        self.service_container.register("input_data_checks", input_data_checks)

        # initiate dictionary for storing extra year data
        self.service_container.register("year_specific_ts", YearSpecificTs())

        # initiate dictionary for storing time steps
        time_steps = self.service_container.build_and_register(
            "time_steps", TimeStepsDicts
        )
        time_steps.sequence_time_steps_yearly = (
            self.model_schema.sequence_time_steps_yearly
        )

        # Initialize the global schema before creating any elements.
        energy_system_folder_path = Path(
            self.dataset_path_resolver.folder_of_set("energy_system")
        )
        unit_handling = UnitHandling(
            energy_system_folder_path, self.config.solver.rounding_decimal_points_units
        )
        self.service_container.register("unit_handling", unit_handling)

        self.service_container.build_and_register(
            "input_repository", InputRepository, folder_path=energy_system_folder_path
        )
        self.service_container.build_and_register("network_topology", NetworkTopology)
        element_registry = self.service_container.build_and_register(
            "element_registry", ElementRegistry
        )
        element_registry.register_elements()

        # check if all elements from the scenario_dict are in the model
        scenario_dict.check_if_all_elements_in_model(element_registry)

        # Store all input parameters using one schema-wide dependency graph.
        parameter_loading_service = self.service_container.build_and_register(
            "parameter_loading_service", ParameterLoadingService
        )
        parameter_loading_service.load_parameters()

        # conduct consistency checks of input units
        unit_handling.consistency_checks_input_units(self.config, element_registry)

    def aggregate_time_series(self) -> None:
        """Conduct the time series aggregation.

        Requires :meth:`load_data` to have been called first.
        """
        self.service_container.build_and_register(
            "time_series_aggregation", TimeSeriesAggregation
        )

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    @property
    def energy_system(self):
        """Return the canonical energy-system element from the schema."""
        return self.model_schema.energy_system

    def run_steps(
        self,
        scenario: str,
        model_name: str,
        no_solve: bool = False,
    ):
        """Run the optimization steps for the given scenario.

        :param scenario: The name of the scenario to run.
        :param model_name: The name of the optimization model.
        :param config: The configuration object containing the analysis, system,
            and solver dictionaries.
        :param no_solve: If True, the optimization problem will not be solved.
        """
        # get rolling horizon years
        optimized_time_steps, steps_horizon = self._get_optimization_horizon()
        steps_horizon_keys = list(steps_horizon.keys())

        # iterate through horizon steps
        for step in steps_horizon_keys:
            optimization_step = self.service_container.build(
                OptimizationStep,
                optimized_time_steps=optimized_time_steps,
                steps_horizon=steps_horizon,
            )
            if not optimization_step.run_step(
                scenario, step, model_name, steps_horizon_keys, no_solve
            ):
                break

    def _get_optimization_horizon(self):
        """Returns list of optimization horizon steps."""
        if not self.config.system.use_rolling_horizon:
            # if not using rolling horizon, the optimization horizon
            # is the entire time series
            optimized_time_steps = [0]
            steps_horizon = {0: self.model_schema.set_years}
            return optimized_time_steps, steps_horizon

        assert (
            self.config.system.years_in_rolling_horizon
            >= self.config.system.years_in_decision_horizon
        ), (
            "There must be at least the same number of years in the rolling"
            "horizon as the decision horizon. years_in_rolling_horizon"
            f"({self.config.system.years_in_rolling_horizon}) "
            "< years_in_decision_horizon "
            f"({self.config.system.years_in_decision_horizon})"
        )
        years_in_horizon = self.config.system.years_in_rolling_horizon
        time_steps_yearly = self.model_schema.set_years
        # skip years_in_decision_horizon years
        optimized_time_steps = [
            year
            for year in time_steps_yearly
            if (
                year % self.config.system.years_in_decision_horizon == 0
                or year == time_steps_yearly[-1]
            )
        ]
        steps_horizon = {
            year: list(
                range(
                    year,
                    min(year + years_in_horizon, max(time_steps_yearly) + 1),
                )
            )
            for year in optimized_time_steps
        }
        return optimized_time_steps, steps_horizon
