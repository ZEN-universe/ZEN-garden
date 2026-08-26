from zen_garden.topology.generic_parameter import GenericParameter


class CapexSpecificStorage(GenericParameter):
    """Specific capex of storage technologies."""

    name = "capex_specific_storage"
    indices = (
        "set_storage_technologies",
        "set_capacity_types",
        "set_nodes",
        "set_years",
    )
    doc = "Specific capex of storage technologies"
    unit_category = {"money": 1, "energy_quantity": -1, "time": 1}
    capacity_types = True

    @classmethod
    def store_input_data(cls, element):
        """Load storage power and energy capex inputs."""
        indices = ["set_nodes", "set_years"]
        cls._store_value(
            element,
            cls.name,
            element.data_input.extract_input_data(
                cls.name,
                indices,
                {"money": 1, "energy_quantity": -1, "time": -1},
            ),
        )
        cls._store_value(
            element,
            f"{cls.name}_energy",
            element.data_input.extract_input_data(
                f"{cls.name}_energy",
                indices,
                {"money": 1, "energy_quantity": -1},
            ),
        )
