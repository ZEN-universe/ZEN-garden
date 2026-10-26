from zen_garden.elements.technology.sets.set_retrofitting_technologies import (
    SetRetrofittingTechnologies,
)
from zen_garden.model.component_types.set import GenericSet


class SetRetrofittingBaseTechnologies(GenericSet):
    name = "set_retrofitting_base_technologies"
    symbol = r"h^{\mathrm{base}} \in \mathcal{H}_{h^{\mathrm{retro}}}^{\mathrm{base}}"
    doc = "Set of base technologies for a specific retrofitting technology."
    index_set = SetRetrofittingTechnologies

    @classmethod
    def get_data(cls, model_constructor):
        return model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class, "retrofit_base_technology"
        )
