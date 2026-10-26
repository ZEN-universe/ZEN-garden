"""Conversion technology expressions."""

from zen_garden.model.component_types.expression import GenericExpression

from .conversion_dependent_flow import ConversionDependentFlow
from .conversion_overnight_capex import ConversionOvernightCapex
from .conversion_reference_flow import ConversionReferenceFlow

CONVERSION_TECHNOLOGY_EXPRESSIONS: list[type[GenericExpression]] = [
    ConversionReferenceFlow,
    ConversionDependentFlow,
    ConversionOvernightCapex,
]

__all__ = [
    "CONVERSION_TECHNOLOGY_EXPRESSIONS",
    "ConversionDependentFlow",
    "ConversionOvernightCapex",
    "ConversionReferenceFlow",
]
