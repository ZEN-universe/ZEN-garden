from zen_garden.elements.energy_system.sets.set_capacity_types import SetCapacityTypes
from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.set import GenericRelationSet


class RelationTechnologyCapacityType(GenericRelationSet):
    """Meaningful capacity categories for each technology."""

    name = "relation_technology_capacity_type"
    dimensions = (SetTechnologies, SetCapacityTypes)

    @classmethod
    def get_combinations(cls, model_constructor):
        sets = model_constructor.optimization_model.sets
        storage_technologies = set(sets[SetStorageTechnologies])
        return [
            (technology, capacity_type)
            for technology in sets[SetTechnologies]
            for capacity_type in sets[SetCapacityTypes]
            if capacity_type != "energy" or technology in storage_technologies
        ]
