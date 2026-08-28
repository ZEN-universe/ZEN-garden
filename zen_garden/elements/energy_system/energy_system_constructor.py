"""Constructor for the EnergySystem."""

import logging

from typing_extensions import override

from zen_garden.elements.energy_system import EnergySystem
from zen_garden.elements.model_constructor import ModelConstructor

logger = logging.getLogger(__name__)


class EnergySystemConstructor(ModelConstructor):
    element_class = EnergySystem

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
