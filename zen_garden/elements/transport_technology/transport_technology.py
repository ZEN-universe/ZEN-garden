"""Class defining the transport technologies."""

import logging
from typing import ClassVar

import numpy as np
import pandas as pd
from typing_extensions import override

from zen_garden.elements.technology import Technology
from zen_garden.elements.transport_technology.constraints import (
    TRANSPORT_TECHNOLOGY_CONSTRAINTS,
)
from zen_garden.elements.transport_technology.expressions import (
    TRANSPORT_TECHNOLOGY_EXPRESSIONS,
)
from zen_garden.elements.transport_technology.parameters import (
    TRANSPORT_TECHNOLOGY_PARAMETERS,
)
from zen_garden.elements.transport_technology.variables import (
    TRANSPORT_TECHNOLOGY_VARIABLES,
)
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.component_types.parameter import GenericParameter
from zen_garden.model.component_types.variable import GenericVariable

logger = logging.getLogger(__name__)


class TransportTechnology(Technology):
    # set label
    label = "set_transport_technologies"
    location_type = "set_edges"
    dict_reversed_edges: dict[str, str] = {}
    own_parameters: ClassVar[list[type[GenericParameter]]] = (
        TRANSPORT_TECHNOLOGY_PARAMETERS
    )
    variables: ClassVar[list[type[GenericVariable]]] = TRANSPORT_TECHNOLOGY_VARIABLES
    expressions: ClassVar[list[type[GenericExpression]]] = (
        TRANSPORT_TECHNOLOGY_EXPRESSIONS
    )
    constraints: ClassVar[list[type[GenericConstraint]]] = (
        TRANSPORT_TECHNOLOGY_CONSTRAINTS
    )

    @override
    def _initialize(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().initialize_reference_carrier()

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

    @override
    def calculate_capex_of_capacities(self, capacities, storage_energy=False):
        """Vectorized version of calculate_capex_of_single_capacity.

        :param capacities: existing capacities indexed by edge and vintage
        :param storage_energy: unused for transport technologies
        :return: capex of existing capacities
        """
        edges = capacities.index.get_level_values(0)
        capacity = capacities.to_numpy(dtype=float)
        capex_specific = self._first_value_per_location(
            self.capex_specific_transport, edges
        )
        capex_per_distance = self._first_value_per_location(
            self.capex_per_distance_transport, edges
        )
        capex = capex_specific * capacity
        if self.config.system.double_capex_transport:
            distance = self._first_value_per_location(self.distance, edges)
            capex = np.where(
                capacity != 0, capex + capex_per_distance * distance, capex
            )
        # no capex if neither specific nor per-distance capex is defined
        capex = np.where(
            np.isnan(capex_specific) & np.isnan(capex_per_distance), 0.0, capex
        )
        return pd.Series(capex, index=capacities.index)

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
