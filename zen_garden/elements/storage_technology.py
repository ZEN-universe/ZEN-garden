"""Class defining the parameters, variables and constraints that hold for all storage
technologies. The class takes the abstract optimization model as an input, and returns
the parameters, variables and constraints that hold for the storage technologies.
"""

import logging
from typing import override

from zen_garden.elements.technology import Technology

logger = logging.getLogger(__name__)


class StorageTechnology(Technology):
    """Class defining storage technologies."""

    # set label
    label = "set_storage_technologies"
    location_type = "set_nodes"

    @override
    def _initialize(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().initialize_reference_carrier()

    def store_input_data(self):
        """Retrieves and stores input data for element as attributes.

        Each Child class overwrites method to store different attributes.
        """
        # get attributes from class <Technology>
        super().store_input_data()
        # set attributes for parameters of child class <StorageTechnology>
        self.efficiency_charge = self.data_input.extract_input_data(
            "efficiency_charge",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={},
        )
        self.efficiency_discharge = self.data_input.extract_input_data(
            "efficiency_discharge",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={},
        )
        self.self_discharge = self.data_input.extract_input_data(
            "self_discharge", index_sets=["set_nodes"], unit_category={}
        )
        # extract existing energy capacity
        self.capacity_addition_min_energy = self.data_input.extract_input_data(
            "capacity_addition_min_energy",
            index_sets=[],
            unit_category={"energy_quantity": 1},
        )
        self.capacity_addition_max_energy = self.data_input.extract_input_data(
            "capacity_addition_max_energy",
            index_sets=[],
            unit_category={"energy_quantity": 1},
        )
        self.capacity_limit_energy = self.data_input.extract_input_data(
            "capacity_limit_energy",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"energy_quantity": 1},
        )
        self.capacity_lower_limit_energy = self.data_input.extract_input_data(
            "capacity_lower_limit_energy",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"energy_quantity": 1},  # Note: No "time": -1 for energy!
        )
        self.capacity_existing_energy = self.data_input.extract_input_data(
            "capacity_existing_energy",
            index_sets=["set_nodes", "set_technologies_existing"],
            unit_category={"energy_quantity": 1},
        )
        self.capacity_investment_existing_energy = self.data_input.extract_input_data(
            "capacity_investment_existing_energy",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"energy_quantity": 1},
        )
        self.energy_to_power_ratio_min = self.data_input.extract_input_data(
            "energy_to_power_ratio_min", index_sets=[], unit_category={"time": 1}
        )
        self.energy_to_power_ratio_max = self.data_input.extract_input_data(
            "energy_to_power_ratio_max", index_sets=[], unit_category={"time": 1}
        )
        self.capex_specific_storage = self.data_input.extract_input_data(
            "capex_specific_storage",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"money": 1, "energy_quantity": -1, "time": -1},
        )
        self.capex_specific_storage_energy = self.data_input.extract_input_data(
            "capex_specific_storage_energy",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"money": 1, "energy_quantity": -1},
        )
        self.opex_specific_fixed = self.data_input.extract_input_data(
            "opex_specific_fixed",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"money": 1, "energy_quantity": -1, "time": 1},
        )
        self.opex_specific_fixed_energy = self.data_input.extract_input_data(
            "opex_specific_fixed_energy",
            index_sets=["set_nodes", "set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
            unit_category={"money": 1, "energy_quantity": -1},
        )
        self.convert_to_fraction_of_capex()
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()
        self.capex_capacity_existing_energy = (
            self.calculate_capex_of_capacities_existing(storage_energy=True)
        )
        # add flow_storage_inflow time series
        self.raw_time_series["flow_storage_inflow"] = (
            self.data_input.extract_input_data(
                "flow_storage_inflow",
                index_sets=["set_nodes", "set_time_steps"],
                time_steps="set_base_time_steps_yearly",
                unit_category={"energy_quantity": 1, "time": -1},
            )
        )

    def convert_to_fraction_of_capex(self):
        """Converts the capex and fixed opex to fraction of capex.

        this method converts the total capex to fraction of capex, depending on
        how many hours per year are calculated.
        """
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        self.opex_specific_fixed_energy = (
            self.opex_specific_fixed_energy * fraction_year
        )
        self.capex_specific_storage = self.capex_specific_storage * fraction_year
        self.capex_specific_storage_energy = (
            self.capex_specific_storage_energy * fraction_year
        )

    def calculate_capex_of_single_capacity(
        self, capacity, index, storage_energy=False, **kwargs
    ):
        """This method calculates the annualized capex of a single existing capacity.

        :param capacity: capacity of storage technology
        :param index: index of capacity
        :param storage_energy: boolean if energy capacity or power capacity
        :return: capex of single capacity
        """
        if storage_energy:
            absolute_capex = (
                self.capex_specific_storage_energy[index[0]].iloc[0] * capacity
            )
        else:
            absolute_capex = self.capex_specific_storage[index[0]].iloc[0] * capacity
        return absolute_capex

    @override
    def calculate_capex_of_capacities_existing(self, storage_energy=False):
        capacities_existing = (
            self.capacity_existing_energy if storage_energy else self.capacity_existing
        )
        return capacities_existing.to_frame().apply(
            lambda _capacity_existing: self.calculate_capex_of_single_capacity(
                _capacity_existing.squeeze(),
                _capacity_existing.name,
                storage_energy=storage_energy,
            ),
            axis=1,
        )
