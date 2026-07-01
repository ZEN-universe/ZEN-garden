"""Class defining the optimization model.

The class takes as inputs the properties of the optimization problem. The
properties are saved in the dictionaries analysis and system which are passed
to the class. After initializing the model, the class adds carriers and
technologies to the model and returns it. The class also includes a method to
solve the optimization problem.
"""

import copy
import logging
import os
from collections import defaultdict
from pathlib import Path

from zen_garden.model import ELEMENT_TYPE_CLASSES
from zen_garden.model.config import Config
from zen_garden.model.context import Context
from zen_garden.model.energy_system import EnergySystem
from zen_garden.model.technology.technology import Technology
from zen_garden.model.zen_model import ZenModel
from zen_garden.postprocess.postprocess import Postprocess
from zen_garden.preprocess.scaling import Scaling
from zen_garden.preprocess.time_series_aggregation import TimeSeriesAggregation
from zen_garden.preprocess.unit_handling import UnitHandling
from zen_garden.services.element_registry import ElementRegistry
from zen_garden.services.model_construction_service import ModelConstructionService
from zen_garden.utils import (
    IISConstraintParser,
    InputDataChecks,
    ScenarioDict,
    StringUtils,
    resolve_dataset_paths,
)

logger = logging.getLogger(__name__)


class OptimizationSetup(object):
    """Class defining the optimization model.

    The class takes as inputs the properties of the optimization problem. The
    properties are saved in the dictionaries analysis and system which are
    passed to the class. After initializing the model, the class adds carriers
    and technologies to the model and returns it. The class also includes a \
    method to solve the optimization problem.
    """

    # dict of element classes, this dict is filled in the __init__ of the package

    def __init__(
        self, raw_config, scenario_dict: dict, input_data_checks: InputDataChecks
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
        # Copying is necessary, because the config object is modified,
        # e.g., in add_elements of ElementRegistry
        self.analysis = copy.deepcopy(raw_config.analysis)
        self.system = copy.deepcopy(raw_config.system)
        self.solver = copy.deepcopy(raw_config.solver)
        self.config = Config.from_setup(self.analysis, self.system, self.solver)

        self.input_data_checks = input_data_checks
        self.input_data_checks.optimization_setup = self
        self.input_data_checks.config = self.config
        # create a dictionary with the paths to access the model inputs
        # check if input data exists
        self.input_data_checks.check_primary_folder_structure()

        self.paths = resolve_dataset_paths(self.config)
        # self.element_list = self._build_element_list()

        # dict to update elements according to scenario
        self.scenario_dict = ScenarioDict(
            scenario_dict, ELEMENT_TYPE_CLASSES, self.paths, self.config
        )

        # empty dict of elements (will be filled with class_name: instance_list)
        self.dict_elements = defaultdict(list)
        # read the parameter change log
        self.parameter_change_log = {
            "min_full_load_hours_fraction": {
                "default_value": 0,  # only 0, 1, or 'inf' are allowed
                "unit": "min_load",
            },
            "capacity_lower_limit": {
                "default_value": 0,  # only 0, 1, or 'inf' are allowed
                "unit": "capacity_limit",
            },
            "capacity_lower_limit_energy": {
                "default_value": 0,  # only 0, 1, or 'inf' are allowed
                "unit": "capacity_limit_energy",
            },
            #    "new_parameter_name": {
            #       "default_value": 0, # only 0, 1, or 'inf' are allowed
            #       "unit": "existing_parameter_name_with_same_unit"
            #   }
        }

        # optimization model
        self.zen_model: ZenModel | None = None
        # self.model = None
        # # the components
        # self.variables = None
        # self.parameters = None
        # self.constraints = None
        # self.sets = None

        # initiate dictionary for storing extra year data
        self.year_specific_ts = {}

        self.context = Context.from_setup(
            paths=self.paths,
            element_classes=ELEMENT_TYPE_CLASSES,
            scenario_dict=self.scenario_dict,
            dict_elements=self.dict_elements,
            parameter_change_log=self.parameter_change_log,
            year_specific_ts=self.year_specific_ts,
        )
        self.input_data_checks.context = self.context
        # check if all needed data inputs for the chosen technologies exist
        # remove non-existent inputs
        self.input_data_checks.check_existing_technology_data()

        # Init the energy system
        self.unit_handling = UnitHandling(
            Path(self.context.paths["energy_system"]["folder"]),
            self.config.solver.rounding_decimal_points_units,
        )
        self.energy_system = EnergySystem(
            self.config,
            self.context,
            self.unit_handling,
        )
        self.element_registry = ElementRegistry(
            self.config,
            self.context,
            self.energy_system,
            self.input_data_checks,
            self.unit_handling,
        )

        # add Elements to optimization
        self.element_registry.add_elements()

        # check if all elements from the scenario_dict are in the model
        ScenarioDict.check_if_all_elements_in_model(
            self.context.scenario_dict, self.context.dict_elements
        )

        # The time series aggregation
        self.time_series_aggregation = None

        # set base scenario
        self.set_base_configuration()

        # read input data into elements
        self.read_input_csv()

        # conduct consistency checks of input units
        self.unit_handling.consistency_checks_input_units(
            self.config,
            self.context,
            self.energy_system,
            self.element_registry,
        )

        # conduct time series aggregation
        self.time_series_aggregation = TimeSeriesAggregation(
            self.energy_system,
            self.config,
            self.context,
            self.element_registry,
        )

    def read_input_csv(self):
        """Read the input and conducts the time series aggregation."""
        logger.info("\n--- Read input data of elements --- \n")
        self.energy_system.store_input_data()
        for element in self.context.dict_elements["Element"]:
            element_class = [
                k for k, v in ELEMENT_TYPE_CLASSES.items() if v == element.__class__
            ][0]
            logger.info(f"Create {element_class} {element.name}")
            element.store_input_data()

    def construct_optimization_problem(self) -> ZenModel:
        """Constructs the optimization problem."""
        # create empty ConcreteModel
        if self.config.solver.solver_dir is not None and not os.path.exists(
            self.config.solver.solver_dir
        ):
            os.makedirs(self.config.solver.solver_dir)

        service = ModelConstructionService(
            self.config,
            self.context,
            self.energy_system,
            self.element_registry,
            self.unit_handling,
        )
        self.zen_model = service.construct_model()

        self.scaling = Scaling(
            self.zen_model.lp_model,
            self.config.solver.scaling_algorithm,
            self.config.solver.scaling_include_rhs,
        )

        return self.zen_model

        # self.model = lp.Model(solver_dir=self.config.solver.solver_dir)
        # # we need to reset the components to not carry them over
        # self.sets = IndexSet()
        # self.variables = Variable(self, self.config, self.context)
        # self.parameters = Parameter(self)
        # self.constraints = Constraint(self.sets, self.model)
        # zen_model = ZenModel(self.config, self.context, self.energy_system)

        # define and construct components of self.model
        # Element.construct_model_components(self, zen_model)

        # self.energy_balance.construct_model_components(zen_model)
        # for element in self.context.element_classes:
        #     element.construct_model_components(zen_model)

        # Initiate scaling object
        # self.scaling = Scaling(
        #     self.model,
        #     self.config.solver.scaling_algorithm,
        #     self.config.solver.scaling_include_rhs,
        # )

    def get_optimization_horizon(self):
        """Returns list of optimization horizon steps."""
        # if using rolling horizon
        if self.config.system.use_rolling_horizon:
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
            self.years_in_horizon = self.config.system.years_in_rolling_horizon
            time_steps_yearly = self.energy_system.set_time_steps_yearly
            # skip years_in_decision_horizon years
            self.optimized_time_steps = [
                year
                for year in time_steps_yearly
                if (
                    year % self.config.system.years_in_decision_horizon == 0
                    or year == time_steps_yearly[-1]
                )
            ]
            self.steps_horizon = {
                year: list(
                    range(
                        year,
                        min(year + self.years_in_horizon, max(time_steps_yearly) + 1),
                    )
                )
                for year in self.optimized_time_steps
            }
        # if no rolling horizon
        else:
            self.years_in_horizon = len(self.energy_system.set_time_steps_yearly)
            self.optimized_time_steps = [0]
            self.steps_horizon = {0: self.energy_system.set_time_steps_yearly}
        return list(self.steps_horizon.keys())

    def get_decision_horizon(self, step_horizon):
        """Return the decision horizon.

        Returns the decision horizon for the optimization step, i.e., the time
        steps for which the decisions are saved.

        :param step_horizon: step of the rolling horizon
        :return: decision_horizon: list of time steps in the decision horizon
        """
        if step_horizon == self.optimized_time_steps[-1]:
            decision_horizon = [step_horizon]
        else:
            next_optimization_step = self.optimized_time_steps[
                self.optimized_time_steps.index(step_horizon) + 1
            ]
            decision_horizon = list(range(step_horizon, next_optimization_step))
        return decision_horizon

    def set_base_configuration(self, scenario="", elements=None):
        """Set base configuration.

        :param scenario: name of base scenario
        :param elements: elements in base configuration
        """
        if elements is None:
            elements = {}
        self.base_scenario = scenario
        self.base_configuration = elements

    def overwrite_time_indices(self, step_horizon):
        """Select subset of time indices, matching the step horizon.

        :param step_horizon: step of the rolling horizon
        """
        if self.config.system.use_rolling_horizon:
            time_steps_yearly_horizon = self.steps_horizon[step_horizon]
            base_time_steps_horizon = (
                self.energy_system.time_steps.decode_yearly_time_steps(
                    time_steps_yearly_horizon
                )
            )
            # overwrite aggregated time steps - operation
            set_time_steps_operation = self.energy_system.time_steps.encode_time_step(
                base_time_steps=base_time_steps_horizon, time_step_type="operation"
            )
            # overwrite aggregated time steps - storage
            set_time_steps_storage = self.energy_system.time_steps.encode_time_step(
                base_time_steps=base_time_steps_horizon, time_step_type="storage"
            )
            # copy invest time steps
            time_steps_operation = set_time_steps_operation.squeeze().tolist()
            time_steps_storage = set_time_steps_storage.squeeze().tolist()
            if isinstance(time_steps_operation, int):
                time_steps_operation = [time_steps_operation]
                time_steps_storage = [time_steps_storage]
            self.energy_system.time_steps.time_steps_operation = time_steps_operation
            self.energy_system.time_steps.time_steps_storage = time_steps_storage
            # overwrite base time steps and yearly base time steps
            new_base_time_steps_horizon = base_time_steps_horizon.squeeze().tolist()
            if not isinstance(new_base_time_steps_horizon, list):
                new_base_time_steps_horizon = [new_base_time_steps_horizon]
            self.energy_system.set_base_time_steps = new_base_time_steps_horizon
            self.energy_system.set_time_steps_yearly = time_steps_yearly_horizon

    def solve(self):
        """Create model instance by assigning parameter values and initializing sets."""
        assert (
            self.zen_model is not None
        ), "The optimization model has not been constructed yet."

        solver_name = self.config.solver.name
        # remove options that are None
        solver_options = {
            key: self.config.solver.solver_options[key]
            for key in self.config.solver.solver_options
            if self.config.solver.solver_options[key] is not None
        }

        logger.info(f"\n--- Solve model instance using {solver_name} ---\n")
        # disable logger temporarily
        logging.disable(logging.WARNING)

        if solver_name == "gurobi":
            self.zen_model.lp_model.solve(
                solver_name=solver_name,
                io_api=self.config.solver.io_api,
                keep_files=self.config.solver.keep_files,
                sanitize_zeros=True,
                # remaining kwargs are passed to the solver
                **solver_options,
            )
        else:
            self.zen_model.lp_model.solve(
                solver_name=solver_name,
                io_api=self.config.solver.io_api,
                keep_files=self.config.solver.keep_files,
                sanitize_zeros=True,
            )
        # enable logger
        logging.disable(logging.NOTSET)
        if self.zen_model.lp_model.termination_condition == "optimal":
            self.optimality = True
        elif self.zen_model.lp_model.termination_condition == "suboptimal":
            logger.warning("The optimization is suboptimal")
            self.optimality = True
        else:
            self.optimality = False

    def write_IIS(self, scenario=""):
        """Write an ILP file to print the IIS if infeasible and using Gurobi."""
        assert (
            self.zen_model is not None
        ), "The optimization model has not been constructed yet."

        if (
            self.zen_model.lp_model.termination_condition == "infeasible"
            and self.config.solver.name == "gurobi"
        ):
            output_folder = StringUtils.get_output_folder(self.config.analysis)
            ilp_file = os.path.join(
                output_folder,
                f"infeasible_model_IIS{f'_{scenario}' if scenario else ''}.ilp",
            )
            logger.info(f"Writing parsed IIS to {ilp_file}")
            parser = IISConstraintParser(ilp_file, self.zen_model.lp_model)
            parser.write_parsed_output()

    def add_results_of_optimization_step(self, step_horizon):
        """Adds capacity additions and carbon emissions to the next optimization step.

        This function takes the capacity additions and carbon emissions of the
        current optimization step and adds them to the existing capacity and
        existing emissions of the next optimization step. Values from the
        currently simulated year are added as existing capacities and
        emissions for future steps.

        Args:
            step_horizon (int): The year index of the current optimization step.
                In myopic foresight, capacities and emissions from this step are
                added to existing capacities and emissions.

        Returns:
            None

        """
        decision_horizon = self.get_decision_horizon(step_horizon)
        # add newly capacity_addition of first year to existing capacity
        self.add_new_capacity_addition(decision_horizon)
        # add cumulative carbon emissions to previous carbon emissions
        self.add_carbon_emission_cumulative(decision_horizon)

    def add_new_capacity_addition(self, decision_horizon):
        """Adds the newly built capacity to the existing capacity.

        This function adds installed capacities from the current optimization
        step to existing capacities in the model. It also adds costs from the
        installed capacities to existing capacity investment. Capacity values whose
        magnitude is below that specified by the solver setting
        "rounding_decimal_points_capacity" are set to zero.

        Args:
            decision_horizon (list or int): A list of the years to transfer installed
                capacities to existing capacities.

        Returns:
            None

        """
        assert (
            self.zen_model is not None
        ), "The optimization model has not been constructed yet."
        capacity_addition = (
            self.zen_model.lp_model.solution["capacity_addition"].to_series().dropna()
        )
        invest_capacity = (
            self.zen_model.lp_model.solution["capacity_investment"].to_series().dropna()
        )
        cost_capex_overnight = (
            self.zen_model.lp_model.solution["cost_capex_overnight"]
            .to_series()
            .dropna()
        )

        if self.config.solver.round_parameters:
            rounding_value = 10 ** (
                -self.config.solver.rounding_decimal_points_capacity
            )
        else:
            rounding_value = 0
        capacity_addition[capacity_addition <= rounding_value] = 0
        invest_capacity[invest_capacity <= rounding_value] = 0
        cost_capex_overnight[cost_capex_overnight <= rounding_value] = 0

        for tech in self.element_registry.get_all_elements(Technology):
            if not isinstance(tech, Technology):
                raise TypeError(
                    f"Element {tech.name} is not of type Technology, "
                    f"but of type {type(tech)}"
                )
            # new capacity
            capacity_addition_tech = capacity_addition.loc[tech.name].unstack()
            capacity_investment = invest_capacity.loc[tech.name].unstack()
            cost_capex_tech = cost_capex_overnight.loc[tech.name].unstack()
            tech.add_new_capacity_addition_tech(
                capacity_addition_tech, cost_capex_tech, decision_horizon
            )
            tech.add_new_capacity_investment(capacity_investment, decision_horizon)

    def add_carbon_emission_cumulative(self, decision_horizon):
        """Adds current emissions to existing emissions.

        This function adds carbon emissions from the current optimization
        step to the existing carbon emissions.

        Args:
            decision_horizon (list or int): A list of the years to transfer
                emissions to existing emissions.

        Returns:
            None

        """
        assert (
            self.zen_model is not None
        ), "The optimization model has not been constructed yet."
        interval_between_years = self.config.system.interval_between_years
        last_year = decision_horizon[-1]
        carbon_emissions_cumulative = (
            self.zen_model.lp_model.solution["carbon_emissions_cumulative"]
            .loc[last_year]
            .item()
        )
        carbon_emissions_annual = (
            self.zen_model.lp_model.solution["carbon_emissions_annual"]
            .loc[last_year]
            .item()
        )
        self.energy_system.carbon_emissions_cumulative_existing = (
            carbon_emissions_cumulative
            + carbon_emissions_annual * (interval_between_years - 1)
        )

    def write_results(
        self,
        scenarios,
        subfolder: tuple[Path, Path] | Path,
        model_name,
        scenario_name,
        param_map,
    ):
        """Write results of the optimization to files.

        This function writes the results of the optimization to files in the
        specified subfolder. It also saves the optimization setup and unit handling
        objects for future use.

        Args:
            subfolder (str): The subfolder where the results will be saved.
            model_name (str): The name of the optimization model.
            scenario_name (str): The name of the scenario being optimized.
            param_map (dict): A dictionary mapping parameter names to their values.
        """
        assert (
            self.zen_model is not None
        ), "The optimization model has not been constructed yet."
        Postprocess(
            self.config,
            self.context,
            self.unit_handling,
            self.zen_model,
            self.energy_system,
            self.scaling,
            optimized_time_steps=self.optimized_time_steps,
            scenarios=scenarios,
            model_name=model_name,
            subfolder=subfolder,
            scenario_name=scenario_name,
            param_map=param_map,
        )
