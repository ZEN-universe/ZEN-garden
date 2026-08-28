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
from zen_garden.preprocess.unit_converter import UnitConverter
from zen_garden.services.attribute_data_loader import AttributeDataLoader
from zen_garden.services.data_loading_service import DataLoadingService
from zen_garden.services.dataset_path_resolver import DatasetPathResolver
from zen_garden.services.element_factory import ElementFactory
from zen_garden.services.element_registry import ElementRegistry
from zen_garden.services.network_topology import NetworkTopology
from zen_garden.services.scenario_dict import ScenarioDict
from zen_garden.services.service_container import ServiceContainer
from zen_garden.topology.model_schema import ModelSchema
from zen_garden.types import YearSpecificTs
from zen_garden.utils.input_data_checks import InputDataChecks

logger = logging.getLogger(__name__)


class OptimizationWorkflow:
    """Class defining the optimization model.

    The constructor only stores its inputs. The workflow then runs in explicit
    stages: :meth:`load_data` wires the services and loads all input data,
    :meth:`aggregate_time_series` builds the time-step registry and conducts the
    time series aggregation, and :meth:`run_steps` builds and solves the
    optimization problem.
    """

    zen_model: ZenModel
    service_container: ServiceContainer
    dataset_path_resolver: DatasetPathResolver
    scenario_dict: ScenarioDict
    element_registry: ElementRegistry
    unit_converter: UnitConverter
    input_data_checks: InputDataChecks

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
        # work on a private copy so the shared input schema is left untouched
        self.model_schema = copy.deepcopy(model_schema)
        self.init_scenario_dict = init_scenario_dict
        self.service_container = ServiceContainer("service_container")

    def load_data(self) -> None:
        """Load all input data for the optimization.

        Runs in four steps: :meth:`_build_services` wires the service graph,
        :meth:`_validate_dataset` checks the dataset's folder structure,
        :meth:`_register_elements` instantiates every configured element, and
        :meth:`_load_parameters` loads every input parameter. Must be called
        before :meth:`aggregate_time_series` and :meth:`run_steps`.
        """
        self._build_services()
        self._validate_dataset()
        self._build_and_register_elements()
        self._load_parameters()

    def _build_services(self) -> None:
        """Wire the service graph the rest of the workflow depends on.

        This is the composition root: it only constructs and registers services
        (path resolver, scenario mapping, data loaders, network topology, ...).
        It does not read element data.
        """

        # Register service: model_schema; instance: the workflow's private schema copy.
        self.service_container.register("model_schema", self.model_schema)

        # Injected service: model_schema; explicit arguments: none.
        # Register the resulting DatasetPathResolver as dataset_path_resolver.
        self.dataset_path_resolver = self.service_container.build_and_register(
            "dataset_path_resolver", DatasetPathResolver
        )

        # dict to update elements according to scenario
        # WARNING: ScenarioDict::__init__ updates the config object!
        # Hence, input_data_checks must be initialized after ScenarioDict
        self.scenario_dict = ScenarioDict(
            self.init_scenario_dict,
            self.dataset_path_resolver,
            self.model_schema,
        )
        # Register service: scenario_dict; instance: the initialized scenario mapping.
        self.service_container.register("scenario_dict", self.scenario_dict)

        # Input data checks are used to validate the dataset and to resolve the
        # technology set (see _validate_dataset / _register_elements). Created
        # here because they depend on the deep-copied model schema and the
        # dataset path resolver built above, and are injected into elements.
        self.input_data_checks = InputDataChecks(model_schema=self.model_schema)
        self.input_data_checks.dataset_path_resolver = self.dataset_path_resolver
        self.service_container.register("input_data_checks", self.input_data_checks)

        # initiate dictionary for storing extra year data
        # Register service: year_specific_ts; instance: a new empty YearSpecificTs.
        self.service_container.register("year_specific_ts", YearSpecificTs())

        # Initialize the global schema before creating any elements.
        energy_system_folder_path = Path(
            self.dataset_path_resolver.folder_of_set("energy_system")
        )
        self.unit_converter = UnitConverter(
            energy_system_folder_path, self.config.solver.rounding_decimal_points_units
        )
        # Register service: unit_converter; instance: the dataset-specific unit handler.
        self.service_container.register("unit_converter", self.unit_converter)

        # Injected services: none; explicit argument: folder_path.
        # Register the resulting AttributeDataLoader as attribute_data_loader.
        self.service_container.build_and_register(
            "attribute_data_loader",
            AttributeDataLoader,
            folder_path=energy_system_folder_path,
        )
        # Injected services: model_schema, attribute_data_loader, input_data_checks,
        # unit_converter; explicit arguments: none. Register as network_topology.
        self.service_container.build_and_register("network_topology", NetworkTopology)
        # Injected services: model_schema, unit_converter; explicit arguments: none.
        # Register the resulting ElementRegistry as element_registry.
        self.element_registry = self.service_container.build_and_register(
            "element_registry", ElementRegistry
        )

    def _validate_dataset(self) -> None:
        """Validate the dataset's folder and file structure."""
        self.input_data_checks.check_primary_folder_structure()

    def _build_and_register_elements(self) -> None:
        """Resolve the technology set, then instantiate every configured element."""
        # Derive config.system.set_technologies from the per-type subsets and
        # fold nested subsets into their parents; ElementFactory reads these.
        # WARNING: this mutates the config object.
        self.input_data_checks.resolve_technology_set()

        # Injected services: service_container, model_schema, input_data_checks;
        # explicit arguments: none.
        self.service_container.build(ElementFactory).register_elements()

        # check if all elements from the scenario_dict are in the model
        self.scenario_dict.check_if_all_elements_in_model(self.element_registry)

    def _load_parameters(self) -> None:
        """Load every input parameter and validate the input units."""
        # Store all input parameters using one schema-wide dependency graph.
        # Injected service: model_schema; explicit arguments: none.
        # Register the resulting service as data_loading_service.
        data_loading_service = self.service_container.build_and_register(
            "data_loading_service", DataLoadingService
        )
        data_loading_service.load_parameters()

        # conduct consistency checks of input units
        self.unit_converter.consistency_checks_input_units(
            self.config, self.element_registry
        )

    def aggregate_time_series(self) -> None:
        """Build the time-step registry and conduct the time series aggregation.

        Requires :meth:`load_data` to have been called first.
        """
        # The time-step registry is owned by this phase: it is created here and
        # then populated by TimeSeriesAggregation.
        # Injected services and explicit arguments: none.
        time_steps = self.service_container.build_and_register(
            "time_steps", TimeStepsDicts
        )
        time_steps.sequence_time_steps_yearly = (
            self.model_schema.sequence_time_steps_yearly
        )

        # Injected services: model_schema, element_registry, time_steps,
        # year_specific_ts, attribute_data_loader; explicit arguments: none.
        # Register the resulting service as time_series_aggregation.
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
            # Injected services: service_container, model_schema, element_registry,
            # unit_converter, scenario_dict, time_steps; explicit arguments:
            # optimized_time_steps and steps_horizon.
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
