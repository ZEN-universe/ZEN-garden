"""Constructor for the RetrofittingTechnology elements."""

import logging

from typing_extensions import override

from zen_garden.elements.model_constructor import ModelConstructor
from zen_garden.elements.retrofitting_technology import (
    RetrofittingTechnology,
)

logger = logging.getLogger(__name__)


class RetrofittingTechnologyConstructor(ModelConstructor):
    element_class = RetrofittingTechnology
    # Optional, self-contained type: only build when retrofitting technologies
    # are configured.
    always_construct = False

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
