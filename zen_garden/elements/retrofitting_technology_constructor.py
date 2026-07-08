import logging
from typing import TYPE_CHECKING, override

import numpy as np

from zen_garden.elements.element_constructor import ElementConstructor
from zen_garden.elements.retrofitting_technology import RetrofittingTechnology
from zen_garden.elements.retrofitting_technology_rules import (
    RetrofittingTechnologyRules,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class RetrofittingTechnologyConstructor(ElementConstructor):
    element_class = RetrofittingTechnology

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class <Carrier>.

        :return: True if there are elements, False otherwise
        """
        return np.size(self.config.system["set_retrofitting_technologies"]) > 0

    def construct_sets(self):
        """Constructs the pe.Sets of the class <RetrofittingTechnology>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
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

    def construct_params(self):
        """Constructs the pe.Params of the class <RetrofittingTechnology>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
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

    def construct_vars(self):
        """Constructs the pe.Vars of the class <RetrofittingTechnology>."""
        logger.info("Constructing variables for RetrofittingTechnology")

    def construct_constraints(self):
        """Constructs the Constraints of the class <RetrofittingTechnology>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing constraints for RetrofittingTechnology")

        # add pwa constraints
        rules = RetrofittingTechnologyRules(
            self.config, self.zen_model, self.energy_system
        )

        # flow coupling of retrofitting technology and its base technology
        rules.constraint_retrofit_flow_coupling()
