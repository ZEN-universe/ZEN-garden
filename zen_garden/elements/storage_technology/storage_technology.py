"""Class defining storage technologies."""

import logging
from typing import ClassVar

import pandas as pd
from typing_extensions import override

from zen_garden.elements.storage_technology.constraints import (
    STORAGE_TECHNOLOGY_CONSTRAINTS,
)
from zen_garden.elements.storage_technology.expressions import (
    STORAGE_TECHNOLOGY_EXPRESSIONS,
)
from zen_garden.elements.storage_technology.parameters import (
    STORAGE_TECHNOLOGY_PARAMETERS,
)
from zen_garden.elements.storage_technology.variables import (
    STORAGE_TECHNOLOGY_VARIABLES,
)
from zen_garden.elements.technology import Technology
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.component_types.parameter import GenericParameter
from zen_garden.model.component_types.variable import GenericVariable

logger = logging.getLogger(__name__)


class StorageTechnology(Technology):
    """Class defining storage technologies."""

    # set label
    label = "set_storage_technologies"
    location_type = "set_nodes"
    own_parameters: ClassVar[list[type[GenericParameter]]] = (
        STORAGE_TECHNOLOGY_PARAMETERS
    )
    variables: ClassVar[list[type[GenericVariable]]] = STORAGE_TECHNOLOGY_VARIABLES
    expressions: ClassVar[list[type[GenericExpression]]] = (
        STORAGE_TECHNOLOGY_EXPRESSIONS
    )
    constraints: ClassVar[list[type[GenericConstraint]]] = (
        STORAGE_TECHNOLOGY_CONSTRAINTS
    )

    @override
    def _initialize(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().initialize_reference_carrier()

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
    def calculate_capex_of_capacities(self, capacities, storage_energy=False):
        """Vectorized version of calculate_capex_of_single_capacity.

        :param capacities: existing capacities indexed by node and vintage
        :param storage_energy: boolean if energy capacity or power capacity
        :return: capex of existing capacities
        """
        capex_specific_storage = (
            self.capex_specific_storage_energy
            if storage_energy
            else self.capex_specific_storage
        )
        capex_specific = self._first_value_per_location(
            capex_specific_storage, capacities.index.get_level_values(0)
        )
        return pd.Series(
            capex_specific * capacities.to_numpy(dtype=float), index=capacities.index
        )
