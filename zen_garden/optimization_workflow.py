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
from typing import TYPE_CHECKING

from zen_garden.elements.energy_system import EnergySystem
from zen_garden.model.config import Config
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

if TYPE_CHECKING:
    from zen_garden.default_config import Config as DefaultConfig
    from zen_garden.utils import InputDataChecks

logger = logging.getLogger(__name__)


class OptimizationWorkflow:
    """Class defining the optimization model.

    The class takes as inputs the properties of the optimization problem. The
    properties are saved in the dictionaries analysis and system which are
    passed to the class. After initializing the model, the class adds carriers
    and technologies to the model and returns it. The class also includes a \
    method to solve the optimization problem.
    """

    zen_model: ZenModel
    service_container: ServiceContainer

    def __init__(
        self,
        model_schema: ModelSchema,
        init_scenario_dict: dict,
        input_data_checks: "InputDataChecks",
    ):
        """Setup optimization of the energy system.

        This function sets up the optimization process for the energy system
        using the provided configuration, scenario data, and input data checks.

        Args:
            config (Config): Config object used to extract the analysis, system,
                and solver dictionaries.
            scenario_dict (dict): Dictionary defining the scenario, including
                data such as resources, demand, etc.
            input_data_checks (InputDataChecks): Input data checks object to
                verify the integrity of the input data.

        """
        self.service_container = ServiceContainer("service_container")
        self.model_schema = copy.deepcopy(model_schema)
        raw_config = self.model_schema.config

        # Copying is necessary, because the config object is modified,
        # e.g., in add_elements of ElementRegistry
        self.config = Config.from_setup(
            copy.deepcopy(raw_config.analysis),
            copy.deepcopy(raw_config.system),
            copy.deepcopy(raw_config.solver),
        )
        self.model_schema.config = self.config
        self.service_container.register("config", self.config)
        self.service_container.register("model_schema", self.model_schema)

        self.dataset_path_resolver = self.service_container.build_and_register(
            "dataset_path_resolver", DatasetPathResolver
        )

        # dict to update elements according to scenario
        # WARNING: ScenarioDict::__init__ updates the config object!
        # Hence, input_data_checks must be initialized after ScenarioDict
        scenario_dict = ScenarioDict(
            init_scenario_dict,
            self.dataset_path_resolver,
            self.config,
            self.model_schema.element_type_classes,
        )
        self.service_container.register("scenario_dict", scenario_dict)

        input_data_checks.config = self.config
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
        energy_systems = element_registry.all_elements_of_type(EnergySystem)
        assert len(energy_systems) == 1
        self.energy_system = energy_systems[0]
        self.service_container.register("energy_system", self.energy_system)

        # check if all elements from the scenario_dict are in the model
        scenario_dict.check_if_all_elements_in_model(element_registry)

        # Store all input parameters using one schema-wide dependency graph.
        parameter_loading_service = self.service_container.build_and_register(
            "parameter_loading_service", ParameterLoadingService
        )
        parameter_loading_service.load_parameters()

        # conduct consistency checks of input units
        unit_handling.consistency_checks_input_units(
            self.config, element_registry
        )

        # conduct time series aggregation
        self.service_container.build_and_register(
            "time_series_aggregation", TimeSeriesAggregation
        )

    def run_steps(
        self,
        scenario: str,
        model_name: str,
        config: "DefaultConfig",
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
                scenario, step, model_name, config, steps_horizon_keys, no_solve
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
