"""Retrofitting-technology set specifications."""

from zen_garden.model.component_types.set import GenericSet

from .set_retrofitting_base_technologies import SetRetrofittingBaseTechnologies

RETROFITTING_TECHNOLOGY_SETS: list[type[GenericSet]] = [SetRetrofittingBaseTechnologies]
__all__ = ["RETROFITTING_TECHNOLOGY_SETS", "SetRetrofittingBaseTechnologies"]
