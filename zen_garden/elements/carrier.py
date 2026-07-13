"""Class defining a generic energy carrier.

The class takes as inputs the abstract optimization model. The class adds parameters,
variables and constraints of a generic carrier and returns the abstract optimization
model.
"""

import logging

import numpy as np

from zen_garden.elements.element import Element

logger = logging.getLogger(__name__)


class Carrier(Element):
    """Class defining a generic energy carrier."""

    # set label
    label = "set_carriers"
    # empty list of elements
    list_of_elements = []

    def store_input_data(self):
        """Retrieves and stores input data for element as attributes. Each Child class
        overwrites method to store different attributes.
        """
        # set attributes of carrier
        # raw import
        self.raw_time_series["demand"] = self.data_input.extract_input_data(
            "demand",
            index_sets=["set_nodes", "set_time_steps"],
            time_steps="set_base_time_steps_yearly",
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.raw_time_series["availability_import"] = (
            self.data_input.extract_input_data(
                "availability_import",
                index_sets=["set_nodes", "set_time_steps"],
                time_steps="set_base_time_steps_yearly",
                unit_category={"energy_quantity": 1, "time": -1},
            )
        )
        self.raw_time_series["availability_export"] = (
            self.data_input.extract_input_data(
                "availability_export",
                index_sets=["set_nodes", "set_time_steps"],
                time_steps="set_base_time_steps_yearly",
                unit_category={"energy_quantity": 1, "time": -1},
            )
        )
        self.raw_time_series["price_export"] = self.data_input.extract_input_data(
            "price_export",
            index_sets=["set_nodes", "set_time_steps"],
            time_steps="set_base_time_steps_yearly",
            unit_category={"money": 1, "energy_quantity": -1},
        )
        self.raw_time_series["price_import"] = self.data_input.extract_input_data(
            "price_import",
            index_sets=["set_nodes", "set_time_steps"],
            time_steps="set_base_time_steps_yearly",
            unit_category={"money": 1, "energy_quantity": -1},
        )
        # non-time series input data
        self.availability_import_yearly = self.data_input.extract_input_data(
            "availability_import_yearly",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"energy_quantity": 1},
        )
        self.availability_export_yearly = self.data_input.extract_input_data(
            "availability_export_yearly",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"energy_quantity": 1},
        )
        self.carbon_intensity_carrier_import = self.data_input.extract_input_data(
            "carbon_intensity_carrier_import",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"emissions": 1, "energy_quantity": -1},
        )
        self.carbon_intensity_carrier_export = self.data_input.extract_input_data(
            "carbon_intensity_carrier_export",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"emissions": 1, "energy_quantity": -1},
        )
        self.price_shed_demand = self.data_input.extract_input_data(
            "price_shed_demand",
            index_sets=[],
            unit_category={"money": 1, "energy_quantity": -1},
        )

    def overwrite_time_steps(self, base_time_steps):
        """Overwrites set_time_steps_operation.

        :param base_time_steps: base time steps of the energy system
        """
        set_time_steps_operation = self.time_steps.encode_time_step(
            base_time_steps=base_time_steps, time_step_type="operation"
        )
        assert isinstance(set_time_steps_operation, np.ndarray)
        self.set_time_steps_operation = set_time_steps_operation.squeeze().tolist()
