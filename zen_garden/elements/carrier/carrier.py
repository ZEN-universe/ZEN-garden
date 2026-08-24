"""Class defining a generic energy carrier."""

import logging
from typing import ClassVar

from zen_garden.elements.carrier.parameters import CARRIER_PARAMETERS
from zen_garden.elements.element import Element
from zen_garden.topology.generic_parameter import GenericParameter

logger = logging.getLogger(__name__)


class Carrier(Element):
    """Class defining a generic energy carrier."""

    # set label
    name = "Carrier"
    label = "set_carriers"
    # empty list of elements
    list_of_elements: list[str] = []
    parameters: ClassVar[list[type[GenericParameter]]] = CARRIER_PARAMETERS
