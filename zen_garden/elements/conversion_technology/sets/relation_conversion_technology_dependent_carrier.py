from zen_garden.elements.energy_system.sets.set_carriers import SetCarriers
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.model.component_types.set import GenericRelationSet


class RelationConversionTechnologyDependentCarrier(GenericRelationSet):
    """Dependent carriers belonging to each conversion technology."""

    name = "relation_conversion_technology_dependent_carrier"
    dimensions = (SetConversionTechnologies, SetCarriers)
    apply_by_default = False

    @classmethod
    def get_combinations(cls, model_constructor):
        registry = model_constructor.element_registry
        element_class = model_constructor.element_class
        inputs = registry.get_attribute_of_all_elements(element_class, "input_carrier")
        outputs = registry.get_attribute_of_all_elements(
            element_class, "output_carrier"
        )
        references = registry.get_attribute_of_all_elements(
            element_class, "reference_carrier"
        )
        dependent = {}
        for technology in inputs:
            carriers = inputs[technology] + outputs[technology]
            carriers.remove(references[technology][0])
            dependent[technology] = carriers
        return dependent
