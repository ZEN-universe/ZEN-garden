"""Class defining the parameters, variables, and constraints of the retrofitting
technologies. The class takes the abstract optimization model as an input and adds
parameters, variables, and constraints of the retrofitting technologies.
"""

import itertools
import logging
from typing import TYPE_CHECKING, override

import numpy as np
import pandas as pd

from zen_garden.model.config import Config
from zen_garden.model.context import Context
from zen_garden.model.element import ElementConstructor
from zen_garden.model.generic_rule import GenericRule
from zen_garden.model.technology.conversion_technology import ConversionTechnology
from zen_garden.model.zen_model import ZenModel
from zen_garden.utils import align_like

if TYPE_CHECKING:
    from zen_garden.model.energy_system import EnergySystem

logger = logging.getLogger(__name__)


class RetrofittingTechnology(ConversionTechnology):
    """Class defining retrofitting technologies."""

    # set label
    label = "set_retrofitting_technologies"
    location_type = "set_nodes"

    def store_carriers(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().store_carriers()

    def store_input_data(self):
        """Retrieves and stores input data for element as attributes.

        Each Child class overwrites method to store different attributes.
        """
        # get attributes from class <Technology>
        super().store_input_data()
        # get retrofit base technology
        self.retrofit_base_technology = (
            self.data_input.extract_retrofit_base_technology()
        )
        # get flow_coupling factor and capex
        self.retrofit_flow_coupling_factor = self.data_input.extract_input_data(
            "retrofit_flow_coupling_factor",
            index_sets=["set_nodes", "set_time_steps"],
            unit_category={},
        )


class RetrofittingTechnologyConstructor(ElementConstructor):
    element_class = RetrofittingTechnology

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class <Carrier>.

        :return: True if there are elements, False otherwise
        """
        return np.size(self.config.system["set_retrofitting_technologies"]) > 0

    def construct_sets(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Sets of the class <RetrofittingTechnology>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing sets for RetrofittingTechnology")
        # get base technologies
        retrofit_base_technology = self.element_registry.get_attribute_of_all_elements(
            self.element_class, "retrofit_base_technology"
        )

        # retrofitting base technologies
        zen_model.sets.add_set(
            name="set_retrofitting_base_technologies",
            data=retrofit_base_technology,
            doc="set of base technologies for a specific retrofitting technology. "
            "Indexed by set_retrofitting_technologies",
            index_set="set_retrofitting_technologies",
        )

    def construct_params(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Params of the class <RetrofittingTechnology>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing parameters for RetrofittingTechnology")

        # slope of linearly modeled capex
        self.add_parameter(
            zen_model,
            energy_system,
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

    def construct_vars(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the pe.Vars of the class <RetrofittingTechnology>."""
        logger.info("Constructing variables for RetrofittingTechnology")

    def construct_constraints(self, zen_model: ZenModel, energy_system: "EnergySystem"):
        """Constructs the Constraints of the class <RetrofittingTechnology>.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        logger.info("Constructing constraints for RetrofittingTechnology")

        # add pwa constraints
        rules = RetrofittingTechnologyRules(self.config, self.context)

        # flow coupling of retrofitting technology and its base technology
        rules.constraint_retrofit_flow_coupling(zen_model, energy_system)


class RetrofittingTechnologyRules(GenericRule):
    """Rules for the RetrofittingTechnology class."""

    def __init__(self, config: Config, context: Context):
        """Inits the rules for a given EnergySystem.

        :param optimization_setup: The OptimizationSetup the element is part of
        """
        super().__init__(config, context)

    def constraint_retrofit_flow_coupling(
        self, zen_model: ZenModel, energy_system: "EnergySystem"
    ):
        """Couples reference flow variables based on modeling technique.

        .. math::
            \\text{if reference carrier in input carriers}
            \\underline{G}_{i,n,t}^\\mathrm{r} = G^\\mathrm{d,approximation}_{i,n,t}
        .. math::
            \\text{if reference carrier in output carriers}
            \\overline{G}_{i,n,t}^\\mathrm{r} = G^\\mathrm{d,approximation}_{i,n,t}

        """
        flow_conversion_input = zen_model.lp_model.variables["flow_conversion_input"]
        flow_conversion_output = zen_model.lp_model.variables["flow_conversion_output"]
        rc_in = pd.Series(
            {
                (t, c): (
                    True if c in zen_model.sets["set_reference_carriers"][t] else False
                )
                for t, c in itertools.product(
                    zen_model.sets["set_conversion_technologies"],
                    zen_model.sets["set_input_carriers"].superset,
                )
            }
        )
        rc_out = pd.Series(
            {
                (t, c): (
                    True if c in zen_model.sets["set_reference_carriers"][t] else False
                )
                for t, c in itertools.product(
                    zen_model.sets["set_conversion_technologies"],
                    zen_model.sets["set_output_carriers"].superset,
                )
            }
        )
        rc_in.index.names = ["set_conversion_technologies", "set_input_carriers"]
        rc_out.index.names = ["set_conversion_technologies", "set_output_carriers"]
        rc_in = align_like(rc_in.to_xarray(), flow_conversion_input)
        rc_out = align_like(rc_out.to_xarray(), flow_conversion_output)
        term_flow_reference = flow_conversion_input.where(rc_in).sum(
            "set_input_carriers"
        ) + flow_conversion_output.where(rc_out).sum("set_output_carriers")
        retrofit_base_technologies = pd.Series(
            {
                t: rt
                for t in zen_model.sets["set_conversion_technologies"]
                if t in zen_model.sets["set_retrofitting_base_technologies"]
                for rt in zen_model.sets["set_retrofitting_base_technologies"][t]
            },
            name="set_conversion_technologies",
        )
        retrofit_base_technologies.index.name = "set_conversion_technologies"
        retrofit_flow_coupling = (
            zen_model.parameters.retrofit_flow_coupling_factor.rename(
                {"set_retrofitting_technologies": "set_conversion_technologies"}
            )
        )
        term_flow_retrofit = self.map_and_expand(
            term_flow_reference, retrofit_base_technologies
        )
        term_flow_base = term_flow_reference.sel(
            {
                "set_conversion_technologies": zen_model.sets[
                    "set_retrofitting_technologies"
                ]
            }
        )
        lhs = term_flow_base - retrofit_flow_coupling * term_flow_retrofit
        rhs = 0
        constraints = lhs <= rhs

        zen_model.constraints.add_constraint(
            "constraint_retrofit_flow_coupling", constraints
        )
