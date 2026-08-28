"""Class defining a generic energy carrier."""

import logging
from typing import ClassVar

from zen_garden.elements.carrier.constraints import CARRIER_CONSTRAINTS
from zen_garden.elements.carrier.expressions import CARRIER_EXPRESSIONS
from zen_garden.elements.carrier.parameters import CARRIER_PARAMETERS
from zen_garden.elements.carrier.variables import CARRIER_VARIABLES
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.component_types.parameter import GenericParameter
from zen_garden.model.component_types.variable import GenericVariable
from zen_garden.model.element import Element

logger = logging.getLogger(__name__)


class Carrier(Element):
    """Class defining a generic energy carrier."""

    # set label
    name = "Carrier"
    label = "set_carriers"
    # empty list of elements
    list_of_elements: list[str] = []
    own_parameters: ClassVar[list[type[GenericParameter]]] = CARRIER_PARAMETERS
    variables: ClassVar[list[type[GenericVariable]]] = CARRIER_VARIABLES
    expressions: ClassVar[list[type[GenericExpression]]] = CARRIER_EXPRESSIONS
    constraints: ClassVar[list[type[GenericConstraint]]] = CARRIER_CONSTRAINTS
