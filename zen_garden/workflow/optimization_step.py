"""Class defining the optimization model.

The class takes as inputs the properties of the optimization problem. The
properties are saved in the dictionaries analysis and system which are passed
to the class. After initializing the model, the class adds carriers and
technologies to the model and returns it. The class also includes a method to
solve the optimization problem.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from zen_garden.elements.technology import Technology
from zen_garden.model.construction_service import ModelConstructionService
from zen_garden.model.optimization_model import OptimizationModel
from zen_garden.postprocess.postprocess import Postprocess
from zen_garden.utils import IISConstraintParser, StringUtils
from zen_garden.workflow.scaling import Scaling

if TYPE_CHECKING:
    from zen_garden.input.scenario_dict import ScenarioDict
    from zen_garden.input.unit_converter import UnitConverter
    from zen_garden.model.element_registry import ElementRegistry
    from zen_garden.model.schema import ModelSchema
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.service_container import ServiceContainer

logger = logging.getLogger(__name__)


class OptimizationStep:
    """Class defining the optimization model.

    The class takes as inputs the properties of the optimization problem. The
    properties are saved in the dictionaries analysis and system which are
    passed to the class. After initializing the model, the class adds carriers
    and technologies to the model and returns it. The class also includes a \
    method to solve the optimization problem.
    """

    optimization_model: "OptimizationModel"
    service_container: "ServiceContainer"

    def __init__(
        self,
        service_container: "ServiceContainer",
        model_schema: "ModelSchema",
        element_registry: "ElementRegistry",
        unit_converter: "UnitConverter",
        scenario_dict: "ScenarioDict",
        time_steps: "TimeStepsDicts",
        optimized_time_steps: list[int],
        steps_horizon: dict[int, list[int]],
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
        self.service_container = service_container
        self.model_schema = model_schema
        self.element_registry = element_registry
        self.unit_converter = unit_converter
        self.scenario_dict = scenario_dict
        self.time_steps = time_steps

        self.optimized_time_steps = optimized_time_steps
        self.steps_horizon = steps_horizon

        # Injected services: service_container, model_schema; explicit arguments: none.
        # Register the resulting OptimizationModel as optimization_model.
        self.optimization_model = self.service_container.build_and_register(
            "optimization_model", OptimizationModel
        )

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    @property
    def energy_system(self):
        """Return the canonical energy-system element from the schema."""
        return self.model_schema.energy_system

    def run_step(
        self,
        scenario: str,
        step: int,
        model_name: str,
        steps_horizon_keys: list[int],
        no_solve: bool = False,
    ) -> bool:
        """Run the optimization step.

        :param scenario: The scenario to run the optimization for.
        :param steps_horizon: The steps of the rolling horizon.
        :param step: The current step of the rolling horizon.
        :param no_solve: If True, the optimization problem will be constructed
            but not solved. Defaults to False.
        :return: True if the optimization was successful, False otherwise
        """
        StringUtils.print_optimization_progress(
            scenario, steps_horizon_keys, step, system=self.config.system
        )
        # overwrite time indices
        self.overwrite_time_indices(step)
        # create optimization problem
        self.construct_optimization_problem()
        self.prepare_scaling()

        if no_solve:
            logger.info(
                "Optimization problem constructed but not solved "
                "(no_solve=True). Continue with next iteration."
            )
            return True

        # SOLVE THE OPTIMIZATION PROBLEM
        self.solve(scenario)

        # break if infeasible
        if not self.optimality:
            # write IIS
            self.write_IIS(scenario)
            assert self.optimization_model is not None
            condition = self.optimization_model.lp_model.termination_condition
            logger.warning(f"Optimization: {condition}")
            return False

        self.re_scale()
        self.add_results_of_optimization_step(step)

        # EVALUATE RESULTS
        scenario_name, subfolder, param_map = StringUtils.generate_folder_path(
            config=self.model_schema.config,
            scenario=scenario,
            scenario_dict=self.scenario_dict,
            steps_horizon=steps_horizon_keys,
            step=step,
        )
        self.write_results(
            scenarios=self.model_schema.config.scenarios,
            subfolder=subfolder,
            model_name=model_name,
            scenario_name=scenario_name,
            param_map=param_map,
        )

        return True

    def construct_optimization_problem(self) -> OptimizationModel:
        """Constructs the optimization problem."""
        # create empty ConcreteModel
        if self.config.solver.solver_dir is not None and not os.path.exists(
            self.config.solver.solver_dir
        ):
            os.makedirs(self.config.solver.solver_dir)

        # Injected services: service_container, model_schema; explicit arguments: none.
        self.service_container.build(ModelConstructionService).construct_model()

        # Injected services: none; explicit arguments: lp_model, algorithm,
        # include_rhs. Register the resulting Scaling instance as scaling.
        self.scaling = self.service_container.build_and_register(
            "scaling",
            Scaling,
            config=self.config,
            lp_model=self.optimization_model.lp_model,
            algorithm=self.config.solver.scaling_algorithm,
            include_rhs=self.config.solver.scaling_include_rhs,
        )

        return self.optimization_model

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

    def overwrite_time_indices(self, step_horizon):
        """Select subset of time indices, matching the step horizon.

        :param step_horizon: step of the rolling horizon
        """
        if not self.config.system.use_rolling_horizon:
            return

        time_steps_yearly_horizon = self.steps_horizon[step_horizon]
        base_time_steps_horizon = self.time_steps.decode_yearly_time_steps(
            time_steps_yearly_horizon
        )
        # overwrite aggregated time steps - operation
        set_time_steps_operation = self.time_steps.encode_time_step(
            base_time_steps=base_time_steps_horizon, time_step_type="operation"
        )
        # overwrite aggregated time steps - storage
        set_time_steps_storage = self.time_steps.encode_time_step(
            base_time_steps=base_time_steps_horizon, time_step_type="storage"
        )
        # copy invest time steps
        assert isinstance(set_time_steps_operation, np.ndarray)
        assert isinstance(set_time_steps_storage, np.ndarray)
        time_steps_operation = set_time_steps_operation.squeeze().tolist()
        time_steps_storage = set_time_steps_storage.squeeze().tolist()
        if isinstance(time_steps_operation, int):
            time_steps_operation = [time_steps_operation]
            time_steps_storage = [time_steps_storage]
        self.time_steps.time_steps_operation = time_steps_operation
        self.time_steps.time_steps_storage = time_steps_storage
        # overwrite base time steps and yearly base time steps
        new_base_time_steps_horizon = base_time_steps_horizon.squeeze().tolist()
        if not isinstance(new_base_time_steps_horizon, list):
            new_base_time_steps_horizon = [new_base_time_steps_horizon]
        self.model_schema.set_hours_all_years = new_base_time_steps_horizon
        self.model_schema.set_years = time_steps_yearly_horizon

    def prepare_scaling(self):
        """Prepare scaling of the optimization problem."""
        if self.config.solver.use_scaling:
            self.scaling.run_scaling()
        elif self.config.solver.analyze_numerics or self.config.solver.run_diagnostics:
            self.scaling.analyze_numerics()

    def re_scale(self):
        """Re-scale the optimization problem after solving."""
        if self.config.solver.use_scaling:
            self.scaling.re_scale()

    def solve(self, scenario: str = "base"):
        """Create model instance by assigning parameter values and initializing sets."""
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
            self.optimization_model.lp_model.solve(
                solver_name=solver_name,
                io_api=self.config.solver.io_api,
                keep_files=self.config.solver.keep_files,
                sanitize_zeros=True,
                # remaining kwargs are passed to the solver
                **solver_options,
            )
        else:
            self.optimization_model.lp_model.solve(
                solver_name=solver_name,
                io_api=self.config.solver.io_api,
                keep_files=self.config.solver.keep_files,
                sanitize_zeros=True,
            )
        # enable logger
        logging.disable(logging.NOTSET)
        if self.optimization_model.lp_model.termination_condition == "optimal":
            self.optimality = True
        elif self.optimization_model.lp_model.termination_condition == "suboptimal":
            logger.warning("The optimization is suboptimal")
            self.optimality = True
        else:
            self.optimality = False

    def write_IIS(self, scenario=""):
        """Write an ILP file to print the IIS if infeasible and using Gurobi."""
        if not self.config.solver.name == "gurobi":
            return
        if (
            self.optimization_model.lp_model.termination_condition
            == "infeasible_or_unbounded"
            and (
                "solver_options" not in self.config.solver
                or "DualReductions" not in self.config.solver.solver_options
            )
        ):
            logger.warning(
                "The optimization problem is infeasible or unbounded. "
                "When using Gurobi, consider setting the solver option "
                "'DualReductions' to 0 to get a more informative termination condition"
                "and."
            )
        if self.optimization_model.lp_model.termination_condition == "infeasible":
            output_folder = StringUtils.get_output_folder(self.config.analysis)
            ilp_file = os.path.join(
                output_folder,
                f"infeasible_model_IIS{f'_{scenario}' if scenario else ''}.ilp",
            )
            logger.info(f"Writing parsed IIS to {ilp_file}")
            parser = IISConstraintParser(ilp_file, self.optimization_model.lp_model)
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
        if not self.config.system.use_rolling_horizon:
            return

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
        capacity_addition = (
            self.optimization_model.lp_model.solution["capacity_addition"]
            .to_series()
            .dropna()
        )
        invest_capacity = (
            self.optimization_model.lp_model.solution["capacity_investment"]
            .to_series()
            .dropna()
        )
        cost_capex_overnight = (
            self.optimization_model.lp_model.solution["cost_capex_overnight"]
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

        for tech in self.element_registry.all_elements_of_type(Technology):
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
        interval_between_years = self.config.system.interval_between_years
        last_year = decision_horizon[-1]
        carbon_emissions_cumulative = (
            self.optimization_model.lp_model.solution["carbon_emissions_cumulative"]
            .loc[last_year]
            .item()
        )
        carbon_emissions_annual = (
            self.optimization_model.lp_model.solution["carbon_emissions_annual"]
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
        Postprocess(
            self.model_schema,
            self.unit_converter,
            self.optimization_model,
            self.scaling,
            self.time_steps,
            optimized_time_steps=self.optimized_time_steps,
            scenarios=scenarios,
            model_name=model_name,
            subfolder=subfolder,
            param_map=param_map,
        ).save_results(scenario_name)
