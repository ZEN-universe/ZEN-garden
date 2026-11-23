"""Conversion-technology set specifications."""

from zen_garden.model.component_types.set import SetDeclaration

from .relation_conversion_technology_dependent_carrier import (
    RelationConversionTechnologyDependentCarrier,
)
from .relation_conversion_technology_input_carrier import (
    RelationConversionTechnologyInputCarrier,
)
from .relation_conversion_technology_output_carrier import (
    RelationConversionTechnologyOutputCarrier,
)

CONVERSION_TECHNOLOGY_SETS: list[SetDeclaration] = [
    RelationConversionTechnologyInputCarrier,
    RelationConversionTechnologyOutputCarrier,
    RelationConversionTechnologyDependentCarrier,
]
__all__ = [
    "CONVERSION_TECHNOLOGY_SETS",
    "RelationConversionTechnologyDependentCarrier",
    "RelationConversionTechnologyInputCarrier",
    "RelationConversionTechnologyOutputCarrier",
]
