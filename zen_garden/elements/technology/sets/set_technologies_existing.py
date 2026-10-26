from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.model.component_types.set import GenericSet


class SetTechnologiesExisting(GenericSet):
    name = "set_technologies_existing"
    symbol = r"v \in \mathcal{V}_{h}"
    doc = "Set of existing technology capacities."
    index_set = SetTechnologies

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class, cls.name
        )
