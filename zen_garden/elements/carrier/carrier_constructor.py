"""Constructor for the Carrier elements."""

import logging

import numpy as np
from typing_extensions import override

from zen_garden.elements.carrier import Carrier
from zen_garden.elements.carrier.constraints import CARRIER_CONSTRAINTS
from zen_garden.elements.model_constructor import ModelConstructor

logger = logging.getLogger(__name__)


class CarrierConstructor(ModelConstructor):
    element_class = Carrier
    constraints = CARRIER_CONSTRAINTS
    parameters = Carrier.own_parameters

    @override
    def has_elements(self) -> bool:
        """Checks if there are any elements of the class
        :class:`zen_garden.elements.carrier.Carrier`.

        :return: True if there are elements, False otherwise
        """
        return True

    @override
    def construct_sets(self):
        logger.info("Constructing sets for Carrier")

    @override
    def construct_vars(self):
        logger.info("Constructing variables for Carrier")

        # flow of imported carrier
        self.zen_model.add_variable(
            name="flow_import",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=(0.0, np.inf),
            doc="node- and time-dependent carrier import from the grid",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # flow of exported carrier
        self.zen_model.add_variable(
            name="flow_export",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=(0.0, np.inf),
            doc="node- and time-dependent carrier export from the grid",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # carrier import/export cost
        self.zen_model.add_variable(
            name="cost_carrier",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            doc="node- and time-dependent carrier cost due to import and export",
            unit_category={"money": 1, "time": -1},
        )
        # total carrier import/export cost
        self.zen_model.add_variable(
            name="cost_carrier_total",
            index_sets=self.zen_model.sets["set_years"],
            doc="total carrier cost due to import and export",
            unit_category={"money": 1},
        )
        # carbon emissions
        self.zen_model.add_variable(
            name="carbon_emissions_carrier",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            doc="carbon emissions of importing and exporting carrier",
            unit_category={"emissions": 1, "time": -1},
        )
        # carbon emissions carrier
        self.zen_model.add_variable(
            name="carbon_emissions_carrier_total",
            index_sets=self.zen_model.sets["set_years"],
            doc="total carbon emissions of importing and exporting carrier",
            unit_category={"emissions": 1},
        )
        # shed demand
        self.zen_model.add_variable(
            name="shed_demand",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=(0.0, np.inf),
            doc="shed demand of carrier",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # cost of shed demand
        self.zen_model.add_variable(
            name="cost_shed_demand",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=(0.0, np.inf),
            doc="shed demand of carrier",
            unit_category={"money": 1, "time": -1},
        )
