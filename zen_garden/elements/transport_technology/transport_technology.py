"""Class defining the transport technologies."""

import logging
from typing import ClassVar

import numpy as np
from typing_extensions import override

from zen_garden.elements.technology import Technology
from zen_garden.elements.transport_technology.parameters import (
    TRANSPORT_TECHNOLOGY_PARAMETERS,
)
from zen_garden.topology.generic_parameter import GenericParameter

logger = logging.getLogger(__name__)


class TransportTechnology(Technology):
    # set label
    label = "set_transport_technologies"
    location_type = "set_edges"
    dict_reversed_edges: dict[str, str] = {}
    own_parameters: ClassVar[list[type[GenericParameter]]] = (
        TRANSPORT_TECHNOLOGY_PARAMETERS
    )

    @override
    def _initialize(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().initialize_reference_carrier()

    def postprocess_input_data(self) -> None:
        """Materialize persistent existing-capacity cost state."""
        self.convert_to_fraction_of_capex()
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()

    def convert_to_fraction_of_capex(self):
        """Converts total capex to fraction of capex.

        this method converts the total capex to fraction of capex, depending on how
        many hours per year are calculated.
        """
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        self.capex_specific_transport = self.capex_specific_transport * fraction_year
        self.capex_per_distance_transport = (
            self.capex_per_distance_transport * fraction_year
        )

    def calculate_capex_of_single_capacity(self, capacity, index, **kwargs):
        """This method calculates the capex of a single existing capacity.

        :param capacity: capacity of transport technology
        :param index: index of capacity
        :return: capex of single capacity
        """
        if np.isnan(self.capex_specific_transport[index[0]].iloc[0]) and np.isnan(
            self.capex_per_distance_transport[index[0]].iloc[0]
        ):
            return 0
        elif self.config.system.double_capex_transport and capacity != 0:
            return (
                self.capex_specific_transport[index[0]].iloc[0] * capacity
                + self.capex_per_distance_transport[index[0]].iloc[0]
                * self.distance[index[0]]
            )
        else:
            return self.capex_specific_transport[index[0]].iloc[0] * capacity

    ### --- getter/setter classmethods
    def set_reversed_edge(self, edge, reversed_edge):
        """Maps the reversed edge to an edge.

        :param edge: edge
        :param reversed_edge: reversed edge
        """
        self.dict_reversed_edges[edge] = reversed_edge

    def get_reversed_edge(self, edge):
        """Get the reversed edge corresponding to an edge.

        :param edge: edge
        :return: reversed edge
        """
        return self.dict_reversed_edges[edge]
