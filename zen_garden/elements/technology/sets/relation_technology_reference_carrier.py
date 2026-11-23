from zen_garden.elements.energy_system.sets.set_carriers import SetCarriers
from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.model.component_types.set import GenericRelationSet


class RelationTechnologyReferenceCarrier(GenericRelationSet):
    """Reference carrier belonging to each technology."""

    name = "relation_technology_reference_carrier"
    dimensions = (SetTechnologies, SetCarriers)
    apply_by_default = False

    @classmethod
    def get_combinations(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class,
            "reference_carrier",
        )
