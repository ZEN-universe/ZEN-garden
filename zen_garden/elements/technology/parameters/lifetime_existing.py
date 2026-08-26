from zen_garden.topology.generic_parameter import GenericParameter


class LifetimeExisting(GenericParameter):
    """Parameter specifying the remaining lifetime of an existing technology."""

    name = "lifetime_existing"
    indices = ("set_technologies", "set_location", "set_technologies_existing")
    doc = "Parameter specifying the remaining lifetime of an existing technology"
    unit_category = {}
    dependencies = ["lifetime"]

    @classmethod
    def store_input_data(cls, element):
        """Extract remaining lifetime for existing capacity vintages."""
        value = element.data_input.extract_lifetime_existing(
            "capacity_existing", index_sets=cls._input_indices(element)
        )
        cls._store_value(element, cls.name, value)
