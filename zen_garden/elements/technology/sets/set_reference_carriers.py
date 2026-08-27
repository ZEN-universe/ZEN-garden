from zen_garden.topology.generic_set import GenericSet


class SetReferenceCarriers(GenericSet):
    name, doc, index_set = (
        "set_reference_carriers",
        "Reference carriers indexed by technology",
        "set_technologies",
    )

    @classmethod
    def get_data(cls, constructor):
        return constructor.element_registry.get_attribute_of_all_elements(
            constructor.element_class, "reference_carrier"
        )
