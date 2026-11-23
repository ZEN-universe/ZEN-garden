from zen_garden.elements.energy_system.sets.set_carriers import SetCarriers
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.model.component_types.set import GenericRelationSet


class RelationConversionTechnologyInputCarrier(GenericRelationSet):
    """Input carriers belonging to each conversion technology."""

    name = "relation_conversion_technology_input_carrier"
    dimensions = (SetConversionTechnologies, SetCarriers)
    apply_by_default = False

    @classmethod
    def get_combinations(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class,
            "input_carrier",
        )
