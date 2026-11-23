"""Retrofitting-technology set specifications."""

from zen_garden.model.component_types.set import SetDeclaration

from .relation_retrofitting_technology_base_technology import (
    RelationRetrofittingTechnologyBaseTechnology,
)

RETROFITTING_TECHNOLOGY_SETS: list[SetDeclaration] = [
    RelationRetrofittingTechnologyBaseTechnology
]
__all__ = [
    "RETROFITTING_TECHNOLOGY_SETS",
    "RelationRetrofittingTechnologyBaseTechnology",
]
