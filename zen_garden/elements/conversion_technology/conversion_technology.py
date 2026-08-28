"""Class defining conversion technologies."""

import logging
from typing import ClassVar, cast

from typing_extensions import override

from zen_garden.elements.conversion_technology.constraints import (
    CONVERSION_TECHNOLOGY_CONSTRAINTS,
)
from zen_garden.elements.conversion_technology.expressions import (
    CONVERSION_TECHNOLOGY_EXPRESSIONS,
)
from zen_garden.elements.conversion_technology.parameters import (
    CONVERSION_TECHNOLOGY_PARAMETERS,
)
from zen_garden.elements.conversion_technology.sets import (
    CONVERSION_TECHNOLOGY_SETS,
)
from zen_garden.elements.conversion_technology.variables import (
    CONVERSION_TECHNOLOGY_VARIABLES,
)
from zen_garden.elements.technology import Technology
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.component_types.parameter import GenericParameter
from zen_garden.model.component_types.set import GenericSet
from zen_garden.model.component_types.variable import GenericVariable

logger = logging.getLogger(__name__)


class ConversionTechnology(Technology):
    """Class defining conversion technologies."""

    # set label
    label = "set_conversion_technologies"
    location_type = "set_nodes"
    own_parameters: ClassVar[list[type[GenericParameter]]] = (
        CONVERSION_TECHNOLOGY_PARAMETERS
    )
    variables: ClassVar[list[type[GenericVariable]]] = CONVERSION_TECHNOLOGY_VARIABLES
    own_sets: ClassVar[list[type[GenericSet]]] = CONVERSION_TECHNOLOGY_SETS
    expressions: ClassVar[list[type[GenericExpression]]] = (
        CONVERSION_TECHNOLOGY_EXPRESSIONS
    )
    constraints: ClassVar[list[type[GenericConstraint]]] = (
        CONVERSION_TECHNOLOGY_CONSTRAINTS
    )

    @override
    def _initialize(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().initialize_reference_carrier()
        # define input and output carrier
        self.input_carrier = cast(
            list[str],
            self.element_data_loader.extract_carriers(carrier_type="input_carrier"),
        )
        self.output_carrier = cast(
            list[str],
            self.element_data_loader.extract_carriers(carrier_type="output_carrier"),
        )
        self.model_schema.set_technology_of_carrier(
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
