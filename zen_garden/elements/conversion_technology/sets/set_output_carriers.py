from zen_garden.topology.generic_set import GenericSet


class SetOutputCarriers(GenericSet):
    name, doc, index_set = (
        "set_output_carriers",
        "Output carriers indexed by conversion technology",
        "set_conversion_technologies",
    )

    @classmethod
    def get_data(cls, constructor):
        return constructor.element_registry.get_attribute_of_all_elements(
            constructor.element_class, "output_carrier"
        )
