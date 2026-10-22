"""Mask for technologies configured for on/off operation."""

from zen_garden.model.component_types.mask import GenericMask


class OnOffTechnologyMask(GenericMask):
    """Technology-location-time tuples configured for on/off operation."""

    name = "on_off_technology"

    @classmethod
    def build(cls, model_constructor, registry):
        index_values, index_names = model_constructor.create_custom_set(
            [
                "set_technologies",
                "set_on_off",
                "set_location",
                "set_time_steps_operation",
            ]
        )
        index_names.pop(1)
        return model_constructor.optimization_model.sets.indices_to_mask(
            index_values, index_names, (0, 0)
        )[0]
