"""Retrofitting technology variables."""

from zen_garden.model.component_types.variable import GenericVariable

RETROFITTING_TECHNOLOGY_VARIABLES: list[type[GenericVariable]] = []

__all__ = [
    "RETROFITTING_TECHNOLOGY_VARIABLES",
]
