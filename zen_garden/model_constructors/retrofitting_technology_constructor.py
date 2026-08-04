"""Constructor for the RetrofittingTechnology elements."""

import logging

import numpy as np
from typing_extensions import override

from zen_garden.constraints.retrofitting_technology import (
    RETROFITTING_TECHNOLOGY_CONSTRAINTS,
)
from zen_garden.elements.retrofitting_technology import RetrofittingTechnology
from zen_garden.model_constructors.model_constructor import ModelConstructor

logger = logging.getLogger(__name__)


class RetrofittingTechnologyConstructor(ModelConstructor):
    element_class = RetrofittingTechnology
    constraints = RETROFITTING_TECHNOLOGY_CONSTRAINTS

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class
        :class:`zen_garden.elements.retrofitting_technology.RetrofittingTechnology`.

        :return: True if there are elements, False otherwise
        """
        return np.size(self.config.system["set_retrofitting_technologies"]) > 0

    @override
    def construct_sets(self):
        logger.info("Constructing sets for RetrofittingTechnology")
        # get base technologies
        retrofit_base_technology = self.element_registry.get_attribute_of_all_elements(
            self.element_class, "retrofit_base_technology"
        )

        # retrofitting base technologies
        self.zen_model.add_set(
            name="set_retrofitting_base_technologies",
            data=retrofit_base_technology,
            doc="set of base technologies for a specific retrofitting technology. "
            "Indexed by set_retrofitting_technologies",
            index_set="set_retrofitting_technologies",
        )

    @override
    def construct_params(self):
        logger.info("Constructing parameters for RetrofittingTechnology")

        # slope of linearly modeled capex
        self.add_parameter(
            name="retrofit_flow_coupling_factor",
            index_names=[
                "set_retrofitting_technologies",
                "set_nodes",
                "set_time_steps_operation",
            ],
            capacity_types=False,
            doc="Parameter which specifies the flow coupling between the retrofitting "
            "technologies and its base technology",
        )

    @override
    def construct_vars(self):
        logger.info("Constructing variables for RetrofittingTechnology")
