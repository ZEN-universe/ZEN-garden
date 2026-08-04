"""Constructor for the EnergySystem."""

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from typing_extensions import override

from zen_garden.constraints.energy_system import ENERGY_SYSTEM_CONSTRAINTS
from zen_garden.model_constructors.model_constructor import ModelConstructor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EnergySystemConstructor(ModelConstructor):
    @override
    def has_elements(self) -> bool:
        """Check if the energy system has elements."""
        return True

    @override
    def construct_sets(self):
        """Constructs the pe.Sets of the class <EnergySystem>."""
        logger.info("Constructing sets for EnergySystem")

        # construct pe.Sets of the class <EnergySystem>
        # nodes
        self.zen_model.add_set(
            name="set_nodes",
            data=self.energy_system.set_nodes,
            doc="Set of nodes",
        )
        # edges
        self.zen_model.add_set(
            name="set_edges",
            data=self.energy_system.set_edges,
            doc="Set of edges",
        )
        # nodes on edges
        self.zen_model.add_set(
            name="set_nodes_on_edges",
            data=self.energy_system.set_nodes_on_edges,
            doc="Set of nodes that constitute an edge. "
            "Edge connects first node with second node.",
            index_set="set_edges",
        )
        # carriers
        self.zen_model.add_set(
            name="set_carriers",
            data=self.energy_system.set_carriers,
            doc="Set of carriers",
        )
        # technologies
        self.zen_model.add_set(
            name="set_technologies",
            data=self.energy_system.set_technologies,
            doc="set_technologies",
        )
        # all elements
        data = list(
            set(self.zen_model.sets["set_technologies"])
            | set(self.zen_model.sets["set_carriers"])
        )
        self.zen_model.add_set(name="set_elements", data=data, doc="Set of elements")
        # set set_elements to indexing_sets
        self.zen_model.indexing_sets.append("set_elements")
        # time-steps
        self.zen_model.add_set(
            name="set_hours_all_years",
            data=self.energy_system.set_hours_all_years,
            doc="Set of base time-steps",
        )
        # yearly time steps
        self.zen_model.add_set(
            name="set_years",
            data=self.energy_system.set_years,
            doc="Set of yearly time-steps",
        )
        # yearly time steps of entire optimization horizon
        self.zen_model.add_set(
            name="set_years_entire_horizon",
            data=self.energy_system.set_years_entire_horizon,
            doc="Set of yearly time-steps of entire optimization horizon",
        )
        # operational time steps
        self.zen_model.add_set(
            name="set_time_steps_operation",
            data=self.energy_system.time_steps.time_steps_operation,
            doc="Set of operational time steps",
        )
        # storage time steps
        self.zen_model.add_set(
            name="set_time_steps_storage",
            data=self.energy_system.time_steps.time_steps_storage,
            doc="Set of storage level time steps",
        )

    @override
    def construct_params(self):
        """Constructs the pe.Params of the class <EnergySystem>."""
        logger.info("Constructing parameters for EnergySystem")

        # operational time step duration
        self.add_parameter(
            name="time_steps_operation_duration",
            set_time_steps="set_time_steps_operation",
            doc="Parameter which specifies the duration of each operational time step",
        )
        # storage time step duration
        self.add_parameter(
            name="time_steps_storage_duration",
            set_time_steps="set_time_steps_storage",
            doc="Parameter which specifies the duration of each storage time step",
        )
        # discount rate
        self.add_parameter(
            name="discount_rate",
            doc="Parameter which specifies the discount rate of the energy system",
        )
        # carbon emissions limit
        self.add_parameter(
            name="carbon_emissions_annual_limit",
            set_time_steps="set_years",
            doc="Parameter which specifies the total limit on carbon emissions",
        )
        # carbon emissions budget
        self.add_parameter(
            name="carbon_emissions_budget",
            doc="Parameter which specifies the total budget of carbon emissions "
            "until the end of the entire time horizon",
        )
        # carbon emissions budget
        self.add_parameter(
            name="carbon_emissions_cumulative_existing",
            doc="Parameter which specifies the total previous carbon emissions",
        )
        # carbon price
        self.add_parameter(
            name="price_carbon_emissions",
            set_time_steps="set_years",
            doc="Parameter which specifies the yearly carbon price",
        )
        # carbon price of budget overshoot
        self.add_parameter(
            name="price_carbon_emissions_budget_overshoot",
            doc="Parameter which specifies the carbon price for budget overshoot",
        )
        # carbon price of annual overshoot
        self.add_parameter(
            name="price_carbon_emissions_annual_overshoot",
            doc="Parameter which specifies the carbon price for annual overshoot",
        )
        # carbon price of overshoot
        self.add_parameter(
            name="market_share_unbounded",
            doc="Parameter which specifies the unbounded market share",
        )
        # knowledge depreciation rate
        self.add_parameter(
            name="knowledge_depreciation_rate",
            doc="Parameter which specifies the knowledge depreciation rate",
        )
        # knowledge spillover rate
        self.add_parameter(
            name="knowledge_spillover_rate",
            doc="Parameter which specifies the knowledge spillover rate",
        )

    @override
    def construct_vars(self):
        """Constructs the pe.Vars of the class <EnergySystem>."""
        logger.info("Constructing variables for EnergySystem")

        # carbon emissions
        self.zen_model.add_variable(
            name="carbon_emissions_annual",
            index_sets=self.zen_model.sets["set_years"],
            doc="annual carbon emissions of energy system",
            unit_category={"emissions": 1},
        )
        # cumulative carbon emissions
        self.zen_model.add_variable(
            name="carbon_emissions_cumulative",
            index_sets=self.zen_model.sets["set_years"],
            doc="cumulative carbon emissions of energy system over time for each year",
            unit_category={"emissions": 1},
        )
        # carbon emission overshoot
        self.zen_model.add_variable(
            name="carbon_emissions_budget_overshoot",
            index_sets=self.zen_model.sets["set_years"],
            bounds=(0, np.inf),
            doc="overshoot carbon emissions of energy system "
            "at the end of the time horizon",
            unit_category={"emissions": 1},
        )
        # carbon emission overshoot
        self.zen_model.add_variable(
            name="carbon_emissions_annual_overshoot",
            index_sets=self.zen_model.sets["set_years"],
            bounds=(0, np.inf),
            doc="overshoot of the annual carbon emissions limit of energy system",
            unit_category={"emissions": 1},
        )
        # cost of carbon emissions
        self.zen_model.add_variable(
            name="cost_carbon_emissions_total",
            index_sets=self.zen_model.sets["set_years"],
            doc="total cost of carbon emissions of energy system",
            unit_category={"money": 1},
        )
        # costs
        self.zen_model.add_variable(
            name="cost_total",
            index_sets=self.zen_model.sets["set_years"],
            doc="total cost of energy system",
            unit_category={"money": 1},
        )
        # net_present_cost
        self.zen_model.add_variable(
            name="net_present_cost",
            index_sets=self.zen_model.sets["set_years"],
            doc="net_present_cost of energy system",
            unit_category={"money": 1},
        )

    @override
    def construct_constraints(self):
        """Constructs the constraints of the class <EnergySystem>."""
        logger.info("Constructing constraints for EnergySystem")

        for EnergySystemConstraint in ENERGY_SYSTEM_CONSTRAINTS:
            constraint = EnergySystemConstraint(
                self.config, self.zen_model, self.energy_system, self.time_steps
            )
            constraint.build()

    @override
    def construct_objective(self):
        """Constructs the pe.Objective of the class <EnergySystem>."""
        logger.info("Constructing objective for EnergySystem")

        # get selected objective rule
        if self.config.analysis.objective == "total_cost":
            objective = self.objective_total_cost()
        elif self.config.analysis.objective == "total_carbon_emissions":
            objective = self.objective_total_carbon_emissions()
        else:
            raise KeyError(f"Objective type {self.config.analysis.objective} not known")

        # get selected objective sense
        sense = self.config.analysis.sense
        assert sense in ["min", "max"], f"Objective sense {sense} not known"

        # construct objective
        self.zen_model.lp_model.add_objective(objective, sense=sense)

    @override
    def _initialize_component(
        self,
        component_name: str,
        index_names: list[str] | None,
        capacity_types: bool = False,
        set_time_steps: str | None = None,
    ):
        """Initialize a modeling component by extracting the stored input data.

        Args:
            component_name: name of modeling component
            index_names: names of index sets, only if calling_class is not EnergySystem
            set_time_steps: time steps, only if calling_class is EnergySystem
        """
        component = getattr(self.energy_system, component_name)
        dict_of_units = {}
        if component_name in self.energy_system.units:
            dict_of_units = self.energy_system.units[component_name]

        if index_names is not None:
            index_list = index_names
        elif set_time_steps is not None:
            index_list = [set_time_steps]
        else:
            index_list = []

        if set_time_steps:
            component_data = component[self.zen_model.sets[set_time_steps]]
        elif type(component) is float:
            component_data = component
        else:
            component_data = component.squeeze()

        return component_data, index_list, dict_of_units

    def _ensure_pd_series_multi_index(self, component_data):
        """Convert pd.Series index to pd.MultiIndex.

        :param component_data: extracted data as pd.Series
        :return: component_data: extracted data as pd.Series with MultiIndex
        """
        if isinstance(component_data, pd.Series) and not isinstance(
            component_data.index, pd.MultiIndex
        ):
            component_data.index = pd.MultiIndex.from_product(
                [component_data.index.to_list()]
            )
        return component_data

    # Objective rules
    # ---------------

    def objective_total_cost(self):
        """Objective function to minimize the total net present cost.

        .. math::
            J = \\sum_{y\\in\\mathcal{Y}} NPC_y

        :param model: optimization model
        :return: net present cost objective function
        """
        return self.zen_model.variables["net_present_cost"].sum("set_years")

    def objective_total_carbon_emissions(self):
        """Objective function to minimize total emissions.

        .. math::
            J = E^{\\mathrm{cum}}_Y

        :math:`E^{\\mathrm{cum}}_Y`: cumulative carbon emissions at the end of
        the time horizon

        :param model: optimization model
        :return: total carbon emissions objective function
        """
        return (
            self.zen_model.variables["carbon_emissions_cumulative"]
            .at[self.zen_model.sets["set_years"][-1]]
            .to_linexpr()
        )
