from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.model.component_types.set import GenericSet


class SetReferenceCarriers(GenericSet):
    name = "set_reference_carriers"
    symbol = r"c_{h}^{\mathrm{ref}} \in \mathcal{C}"
    doc = "Reference carrier corresponding to a technology."
    index_set = SetTechnologies

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class, "reference_carrier"
        )
