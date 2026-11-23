from zen_garden.model.component_types.set import GenericDomainSet


class SetExistingCapacityIds(GenericDomainSet):
    """Canonical identifiers for existing technology-capacity vintages."""

    name = "set_existing_capacity_ids"
    symbol = r"v \in \mathcal{V}"
    doc = "Set of identifiers used for existing technology capacities."

    @classmethod
    def get_data(cls, model_constructor):
        capacities = model_constructor.element_registry.get_attribute_of_all_elements(
            model_constructor.element_class,
            "set_technologies_existing",
        )
        return list(
            dict.fromkeys(
                identifier
                for identifiers in capacities.values()
                for identifier in identifiers
            )
        )
