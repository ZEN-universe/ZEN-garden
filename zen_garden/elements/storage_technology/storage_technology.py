"""Class defining storage technologies."""

import logging
from typing import ClassVar

from typing_extensions import override

from zen_garden.elements.storage_technology.parameters import (
    STORAGE_TECHNOLOGY_PARAMETERS,
)
from zen_garden.elements.technology import Technology
from zen_garden.topology.generic_parameter import GenericParameter

logger = logging.getLogger(__name__)


class StorageTechnology(Technology):
    """Class defining storage technologies."""

    # set label
    label = "set_storage_technologies"
    location_type = "set_nodes"
    own_parameters: ClassVar[list[type[GenericParameter]]] = (
        STORAGE_TECHNOLOGY_PARAMETERS
    )

    @override
    def _initialize(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().initialize_reference_carrier()

    def postprocess_input_data(self) -> None:
        """Materialize persistent existing-capacity cost state."""
        self.convert_to_fraction_of_capex()
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()
        self.capex_capacity_existing_energy = (
            self.calculate_capex_of_capacities_existing(storage_energy=True)
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
