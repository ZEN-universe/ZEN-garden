"""Constructor for the Carrier elements."""

import logging
from typing import override

import numpy as np

from zen_garden.elements.carrier import Carrier
from zen_garden.elements.carrier_rules import CarrierRules
from zen_garden.elements.element_constructor import ElementConstructor
from zen_garden.model.components.zen_index import ZenIndex

logger = logging.getLogger(__name__)


class CarrierConstructor(ElementConstructor):
    element_class = Carrier

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
    def construct_params(self):
        logger.info("Constructing parameters for Carrier")

        # demand of carrier
        self.add_parameter(
            name="demand",
            index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the carrier demand",
        )
        # availability of carrier
        self.add_parameter(
            name="availability_import",
            index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the maximum energy that can be imported "
            "from outside the system boundaries",
        )
        # availability of carrier
        self.add_parameter(
            name="availability_export",
            index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the maximum energy that can be exported "
            "to outside the system boundaries",
        )
        # availability of carrier
        self.add_parameter(
            name="availability_import_yearly",
            index_names=["set_carriers", "set_nodes", "set_years"],
            doc="Parameter which specifies the maximum energy that can be imported "
            "from outside the system boundaries for the entire year",
        )
        # availability of carrier
        self.add_parameter(
            name="availability_export_yearly",
            index_names=["set_carriers", "set_nodes", "set_years"],
            doc="Parameter which specifies the maximum energy that can be exported "
            "to outside the system boundaries for the entire year",
        )
        # import price
        self.add_parameter(
            name="price_import",
            index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the import carrier price",
        )
        # export price
        self.add_parameter(
            name="price_export",
            index_names=["set_carriers", "set_nodes", "set_time_steps_operation"],
            doc="Parameter which specifies the export carrier price",
        )
        # demand shedding price
        self.add_parameter(
            name="price_shed_demand",
            index_names=["set_carriers"],
            doc="Parameter which specifies the price to shed demand",
        )
        # carbon intensity carrier import
        self.add_parameter(
            name="carbon_intensity_carrier_import",
            index_names=["set_carriers", "set_nodes", "set_years"],
            doc="Parameter which specifies the carbon intensity of carrier import",
        )
        # carbon intensity carrier export
        self.add_parameter(
            name="carbon_intensity_carrier_export",
            index_names=["set_carriers", "set_nodes", "set_years"],
            doc="Parameter which specifies the carbon intensity of carrier export",
        )

    @override
    def construct_vars(self):
        logger.info("Constructing variables for Carrier")

        variables = self.zen_model.variables
        sets = self.zen_model.sets

        # flow of imported carrier
        variables.add_variable(
            name="flow_import",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=(0.0, np.inf),
            doc="node- and time-dependent carrier import from the grid",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # flow of exported carrier
        variables.add_variable(
            name="flow_export",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=(0.0, np.inf),
            doc="node- and time-dependent carrier export from the grid",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # carrier import/export cost
        variables.add_variable(
            name="cost_carrier",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            doc="node- and time-dependent carrier cost due to import and export",
            unit_category={"money": 1, "time": -1},
        )
        # total carrier import/export cost
        variables.add_variable(
            name="cost_carrier_total",
            index_sets=sets["set_years"],
            doc="total carrier cost due to import and export",
            unit_category={"money": 1},
        )
        # carbon emissions
        variables.add_variable(
            name="carbon_emissions_carrier",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            doc="carbon emissions of importing and exporting carrier",
            unit_category={"emissions": 1, "time": -1},
        )
        # carbon emissions carrier
        variables.add_variable(
            name="carbon_emissions_carrier_total",
            index_sets=sets["set_years"],
            doc="total carbon emissions of importing and exporting carrier",
            unit_category={"emissions": 1},
        )
        # shed demand
        variables.add_variable(
            name="shed_demand",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=(0.0, np.inf),
            doc="shed demand of carrier",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        # cost of shed demand
        variables.add_variable(
            name="cost_shed_demand",
            index_sets=self.create_custom_set(
                ["set_carriers", "set_nodes", "set_time_steps_operation"],
            ),
            bounds=(0.0, np.inf),
            doc="shed demand of carrier",
            unit_category={"money": 1, "time": -1},
        )

    @override
    def construct_constraints(self):
        logger.info("Constructing constraints for Carrier")

        rules = CarrierRules(
            self.config, self.zen_model, self.energy_system, self.time_steps
        )

        # limit import/export flow by availability
        rules.constraint_availability_import_export()

        # limit import/export flow by availability for each year
        rules.constraint_availability_import_export_yearly()

        # cost for carrier
        rules.constraint_cost_carrier()

        # cost and limit for shed demand
        rules.constraint_cost_limit_shed_demand()

        # total cost for carriers
        rules.constraint_cost_carrier_total()

        # carbon emissions
        rules.constraint_carbon_emissions_carrier()

        # carbon emissions carrier
        rules.constraint_carbon_emissions_carrier_total()

        # energy balance
        index_values, index_names = self.create_custom_set(
            ["set_carriers", "set_nodes", "set_time_steps_operation"]
        )
        rules.constraint_nodal_energy_balance(
            ZenIndex(index_values, index_names),
            index_names[:1],
        )
