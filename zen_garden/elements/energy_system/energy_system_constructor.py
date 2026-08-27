"""Constructor for the EnergySystem."""

import logging
from typing import TYPE_CHECKING

import pandas as pd
from typing_extensions import override

from zen_garden.elements.energy_system import EnergySystem
from zen_garden.elements.energy_system.constraints import ENERGY_SYSTEM_CONSTRAINTS
from zen_garden.elements.model_constructor import ModelConstructor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EnergySystemConstructor(ModelConstructor):
    element_class = EnergySystem
    constraints = ENERGY_SYSTEM_CONSTRAINTS


    @override
    def construct_vars(self):
        """Constructs the pe.Vars of the class <EnergySystem>."""
        logger.info("Constructing variables for EnergySystem")

        for variable in self.variables:

            if variable.name in [
                "carbon_emissions_annual",
                "carbon_emissions_cumulative",
                "carbon_emissions_budget_overshoot",
                "carbon_emissions_annual_overshoot",
                "cost_carbon_emissions_total",
                "cost_total",
                "net_present_cost",
            ]:
                # Exceptional bounds, masks or indices
                index_sets = self.zen_model.sets["set_years"]
                bounds = variable.get_bounds()
            else:
                # Standard behavior
                index_sets = self.create_custom_set(variable.indices)
                bounds = variable.get_bounds()

            self.zen_model.add_variable(
                name=variable.name,
                index_sets=index_sets,
                binary=variable.binary,
                bounds=bounds,
                doc=variable.doc,
                unit_category=variable.unit_category,
            )

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
