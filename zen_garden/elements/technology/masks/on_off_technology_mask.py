"""Mask for technologies configured for on/off operation."""

from zen_garden.model.component_types.mask import GenericMask


class OnOffTechnologyMask(GenericMask):
    """Technology-location-time tuples configured for on/off operation."""

    name = "on_off_technology"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_mask(cls, model_constructor, registry):
        sets = model_constructor.optimization_model.sets
        resolved = sets.resolve_indices(
            [
                "set_technologies",
                "set_location",
                "set_years",
                "set_representative_hours",
            ]
        )
        technologies = sets.on_off_technologies()
        technology_mask = resolved.structural_mask.coords["set_technologies"].isin(
            list(technologies)
        )
        return resolved.structural_mask & technology_mask
