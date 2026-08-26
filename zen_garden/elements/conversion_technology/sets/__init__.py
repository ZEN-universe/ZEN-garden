"""Conversion-technology set specifications."""

from zen_garden.topology.generic_set import GenericSet

from .set_dependent_carriers import SetDependentCarriers
from .set_input_carriers import SetInputCarriers
from .set_output_carriers import SetOutputCarriers

CONVERSION_TECHNOLOGY_SETS: list[type[GenericSet]] = [
    SetInputCarriers,
    SetOutputCarriers,
    SetDependentCarriers,
]
__all__ = [
    "CONVERSION_TECHNOLOGY_SETS",
    "SetDependentCarriers",
    "SetInputCarriers",
    "SetOutputCarriers",
]
