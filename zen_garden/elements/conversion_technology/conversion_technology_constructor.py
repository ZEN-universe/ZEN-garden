"""Constructor for the ConversionTechnology elements."""

import logging

from typing_extensions import override

from zen_garden.elements.conversion_technology import (
    ConversionTechnology,
)
from zen_garden.elements.conversion_technology.constraints import LinearCapexConstraint
from zen_garden.elements.model_constructor import ModelConstructor

logger = logging.getLogger(__name__)


class ConversionTechnologyConstructor(ModelConstructor):
    element_class = ConversionTechnology

    @override
    def construct_constraints(self):
        logger.info("Constructing constraints for ConversionTechnology")

        for ConversionTechnologyConstraint in self.constraints:
            self.service_container.build(ConversionTechnologyConstraint).build()

        # capex (built last, after the generic conversion constraints)
        self.service_container.build(LinearCapexConstraint).build()
