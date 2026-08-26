"""Conversion-technology set specifications."""

from zen_garden.topology.generic_set import GenericSet


class SetInputCarriers(GenericSet):
    name = "set_input_carriers"
    doc = "Input carriers indexed by conversion technology"
    index_set = "set_conversion_technologies"

    @classmethod
    def get_data(cls, constructor):
        return constructor.element_registry.get_attribute_of_all_elements(
            constructor.element_class, "input_carrier"
        )


class SetOutputCarriers(GenericSet):
    name = "set_output_carriers"
    doc = "Output carriers indexed by conversion technology"
    index_set = "set_conversion_technologies"

    @classmethod
    def get_data(cls, constructor):
        return constructor.element_registry.get_attribute_of_all_elements(
            constructor.element_class, "output_carrier"
        )


class SetDependentCarriers(GenericSet):
    name = "set_dependent_carriers"
    doc = "Non-reference carriers indexed by conversion technology"
    index_set = "set_conversion_technologies"

    @classmethod
    def get_data(cls, constructor):
        registry = constructor.element_registry
        element_class = constructor.element_class
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


CONVERSION_TECHNOLOGY_SETS: list[type[GenericSet]] = [
    SetInputCarriers,
    SetOutputCarriers,
    SetDependentCarriers,
]
