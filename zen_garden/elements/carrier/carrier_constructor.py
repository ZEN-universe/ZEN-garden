"""Constructor for the Carrier elements."""

import logging

from typing_extensions import override

from zen_garden.elements.carrier import Carrier
from zen_garden.elements.carrier.constraints import CARRIER_CONSTRAINTS
from zen_garden.elements.model_constructor import ModelConstructor

logger = logging.getLogger(__name__)


class CarrierConstructor(ModelConstructor):
    element_class = Carrier
    constraints = CARRIER_CONSTRAINTS

    @override
    def construct_vars(self):
        logger.info("Constructing variables for Carrier")

        for variable in self.variables:
            if variable.name in [
                "cost_carrier_total",
                "carbon_emissions_carrier_total",
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
