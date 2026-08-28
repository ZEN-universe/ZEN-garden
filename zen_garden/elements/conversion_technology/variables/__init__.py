"""Conversion technology variables."""

from zen_garden.model.component_types.variable import GenericVariable

from .flow_conversion_input import FlowConversionInput
from .flow_conversion_output import FlowConversionOutput

CONVERSION_TECHNOLOGY_VARIABLES: list[type[GenericVariable]] = [
    FlowConversionInput,
    FlowConversionOutput,
]

__all__ = [
    "FlowConversionInput",
    "FlowConversionOutput",
    "CONVERSION_TECHNOLOGY_VARIABLES",
]
