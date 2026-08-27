"""Class defining a generic energy carrier."""

import logging
from typing import ClassVar

from zen_garden.elements.carrier.parameters import CARRIER_PARAMETERS
from zen_garden.elements.carrier.variables import CARRIER_VARIABLES
from zen_garden.elements.element import Element
from zen_garden.topology.generic_parameter import GenericParameter
from zen_garden.topology.generic_variable import GenericVariable

logger = logging.getLogger(__name__)


class Carrier(Element):
    """Class defining a generic energy carrier."""

    # set label
    name = "Carrier"
    label = "set_carriers"
    # empty list of elements
    list_of_elements: list[str] = []
    own_parameters: ClassVar[list[type[GenericParameter]]] = CARRIER_PARAMETERS
    # Todo: Add the constraints here?
    parameters: ClassVar[list[type[GenericParameter]]] = CARRIER_PARAMETERS
    variables: ClassVar[list[type[GenericVariable]]] = CARRIER_VARIABLES
