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
        """Set the optimization objective from a registered expression.

        The objective candidates are built as energy-system expressions (see
        ``zen_garden.elements.energy_system.expressions``); ``config.analysis``
        selects which one to use and with which sense.
        """
        logger.info("Constructing objective for EnergySystem")

        objective_name = self.config.analysis.objective
        if objective_name not in self.zen_model.expressions:
            raise KeyError(f"Objective type {objective_name} not known")

        sense = self.config.analysis.sense
        assert sense in ["min", "max"], f"Objective sense {sense} not known"

        self.zen_model.lp_model.add_objective(
            self.zen_model.expressions[objective_name], sense=sense
        )
