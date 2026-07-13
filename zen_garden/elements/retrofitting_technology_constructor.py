import logging
from typing import override

import numpy as np

from zen_garden.elements.element_constructor import ElementConstructor
from zen_garden.elements.retrofitting_technology import RetrofittingTechnology
from zen_garden.elements.retrofitting_technology_rules import (
    RetrofittingTechnologyRules,
)

logger = logging.getLogger(__name__)


class RetrofittingTechnologyConstructor(ElementConstructor):
    element_class = RetrofittingTechnology

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class <Carrier>.

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
        self.zen_model.sets.add_set(
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

    @override
    def construct_constraints(self):
        logger.info("Constructing constraints for RetrofittingTechnology")

        # add pwa constraints
        rules = RetrofittingTechnologyRules(
            self.config, self.zen_model, self.energy_system, self.time_steps
        )

        # flow coupling of retrofitting technology and its base technology
        rules.constraint_retrofit_flow_coupling()
