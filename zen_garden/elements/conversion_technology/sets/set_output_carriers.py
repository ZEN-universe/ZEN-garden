from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.model.component_types.set import GenericSet


class SetOutputCarriers(GenericSet):
    name = "set_output_carriers"
    symbol = r"c^{\mathrm{out}} \in \mathcal{C}_{h}^{\mathrm{out}}"
    doc = "Set of carriers that are an output to a specific conversion technology."
    index_set = SetConversionTechnologies

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class, "output_carrier"
        )
