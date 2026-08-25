"""Class defining conversion technologies."""

import logging
from typing import ClassVar, cast

from typing_extensions import override

from zen_garden.elements.conversion_technology.parameters import (
    CONVERSION_TECHNOLOGY_PARAMETERS,
)
from zen_garden.elements.technology import Technology
from zen_garden.topology.generic_parameter import GenericParameter

logger = logging.getLogger(__name__)


class ConversionTechnology(Technology):
    """Class defining conversion technologies."""

    # set label
    label = "set_conversion_technologies"
    location_type = "set_nodes"
    own_parameters: ClassVar[list[type[GenericParameter]]] = (
        CONVERSION_TECHNOLOGY_PARAMETERS
    )

    @override
    def _initialize(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().initialize_reference_carrier()
        # define input and output carrier
        self.input_carrier = cast(
            list[str], self.data_input.extract_carriers(carrier_type="input_carrier")
        )
        self.output_carrier = cast(
            list[str], self.data_input.extract_carriers(carrier_type="output_carrier")
        )
        self.energy_system.set_technology_of_carrier(
            self.name, self.input_carrier + self.output_carrier
        )
        # check if reference carrier in input and output carriers and
        #   set technology to correspondent carrier
        self.input_data_checks.check_carrier_configuration(
            input_carrier=self.input_carrier,
            output_carrier=self.output_carrier,
            reference_carrier=self.reference_carrier,
            name=self.name,
        )

    def postprocess_input_data(self) -> None:
        """Materialize persistent existing-capacity cost state."""
        self.convert_to_fraction_of_capex()
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()

    def convert_to_fraction_of_capex(self):
        """This method retrieves the total capex and converts it to annualized capex."""

        # annualize cost_capex_overnight
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        self.capex_specific_conversion = self.capex_specific_conversion * fraction_year

    def calculate_capex_of_single_capacity(self, capacity, index, **kwargs):
        """This method calculates the annualized capex of a single existing capacity.

        :param capacity: existing capacity of technology
        :param index: index of capacity specifying node and time
        :return: annualized capex of a single existing capacity
        """
        if capacity == 0:
            return 0
        capex = self.capex_specific_conversion[index[0]].iloc[0] * capacity

        return capex
