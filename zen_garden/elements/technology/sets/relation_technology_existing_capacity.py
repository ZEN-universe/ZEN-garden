from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.technology.sets.set_existing_capacity_ids import (
    SetExistingCapacityIds,
)
from zen_garden.model.component_types.set import GenericRelationSet


class RelationTechnologyExistingCapacity(GenericRelationSet):
    """Existing-capacity identifiers belonging to each technology."""

    name = "relation_technology_existing_capacity"
    dimensions = (SetTechnologies, SetExistingCapacityIds)

    @classmethod
    def get_combinations(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class,
            "set_technologies_existing",
        )
