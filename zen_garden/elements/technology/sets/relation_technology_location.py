from zen_garden.elements.energy_system.sets.set_location import SetLocation
from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.elements.technology.sets.set_retrofitting_technologies import (
    SetRetrofittingTechnologies,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.elements.technology.sets.set_transport_technologies import (
    SetTransportTechnologies,
)
from zen_garden.model.component_types.set import GenericRelationSet


class RelationTechnologyLocation(GenericRelationSet):
    """Locations at which each technology can be constructed."""

    name = "relation_technology_location"
    dimensions = (SetTechnologies, SetLocation)

    @classmethod
    def get_combinations(cls, model_constructor):
        sets = model_constructor.optimization_model.sets
        node_technologies = list(
            dict.fromkeys(
                [
                    *sets[SetConversionTechnologies],
                    *sets[SetStorageTechnologies],
                    *sets[SetRetrofittingTechnologies],
                ]
            )
        )
        combinations = [
            (technology, node)
            for technology in node_technologies
            for node in model_constructor.network_topology.set_nodes
        ]
        combinations.extend(
            (technology, edge)
            for technology in sets[SetTransportTechnologies]
            for edge in model_constructor.network_topology.set_edges
        )
        return combinations
