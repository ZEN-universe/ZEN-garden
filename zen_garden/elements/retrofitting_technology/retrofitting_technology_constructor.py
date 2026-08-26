"""Constructor for the RetrofittingTechnology elements."""

import logging

import numpy as np
from typing_extensions import override

from zen_garden.elements.model_constructor import ModelConstructor
from zen_garden.elements.retrofitting_technology import (
    RetrofittingTechnology,
)
from zen_garden.elements.retrofitting_technology.constraints import (
    RETROFITTING_TECHNOLOGY_CONSTRAINTS,
)

logger = logging.getLogger(__name__)


class RetrofittingTechnologyConstructor(ModelConstructor):
    element_class = RetrofittingTechnology
    constraints = RETROFITTING_TECHNOLOGY_CONSTRAINTS
    parameters = RetrofittingTechnology.own_parameters
    variables = RetrofittingTechnology.variables
    sets = RetrofittingTechnology.own_sets

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class
        :class:`zen_garden.elements.retrofitting_technology.RetrofittingTechnology`.

        :return: True if there are elements, False otherwise
        """
        return np.size(self.config.system["set_retrofitting_technologies"]) > 0

    @override
    def construct_vars(self):
        logger.info("Constructing variables for RetrofittingTechnology")

        for variable in self.variables:
            if variable.name in []:
                # Exceptional bounds, masks or indices
                index_sets = None
                bounds = None
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