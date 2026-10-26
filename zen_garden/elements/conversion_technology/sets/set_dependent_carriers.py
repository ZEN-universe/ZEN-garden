from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.model.component_types.set import GenericSet


class SetDependentCarriers(GenericSet):
    name = "set_dependent_carriers"
    symbol = r"c^{\mathrm{dep}} \in \mathcal{C}_{h}^{\mathrm{dep}}"
    doc = (
        "Set of carriers that are dependent on the reference carrier of a "
        "specific conversion technology."
    )
    index_set = SetConversionTechnologies

    @classmethod
    def get_data(cls, model_constructor):
        registry, element_class = (
            model_constructor.element_registry,
            model_constructor.element_class,
        )
        inputs = registry.get_attribute_of_all_elements(element_class, "input_carrier")
        outputs = registry.get_attribute_of_all_elements(
            element_class, "output_carrier"
        )
        references = registry.get_attribute_of_all_elements(
            element_class, "reference_carrier"
        )
        dependent = {}
        for technology in inputs:
            dependent[technology] = inputs[technology] + outputs[technology]
            dependent[technology].remove(references[technology][0])
        return dependent
