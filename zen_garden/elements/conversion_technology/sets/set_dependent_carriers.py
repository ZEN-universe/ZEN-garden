from zen_garden.topology.generic_set import GenericSet


class SetDependentCarriers(GenericSet):
    name, doc, index_set = (
        "set_dependent_carriers",
        "Non-reference carriers indexed by conversion technology",
        "set_conversion_technologies",
    )

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
