from zen_garden.model.component_types.parameter import (
    GenericParameter,
    ParameterInputData,
)


class CapexSpecificStorage(GenericParameter):
    """Specific capex of storage technologies."""

    name = "capex_specific_storage"
    symbol = r"\kappa^{\mathrm{cap}}_{h,q,y}"
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_years",
    )
    doc = "Specific capex of power- and energy-rated storage capacity."
    unit_category = {"money": 1, "energy_quantity": -1, "time": 1}

    @classmethod
    def get_input_data(cls, element):
        """Load storage power-capacity CAPEX."""
        indices = ["set_nodes", "set_years"]
        value = element.element_data_loader.extract_input_data(
            cls.name,
            indices,
            {"money": 1, "energy_quantity": -1, "time": -1},
        )
        return ParameterInputData(value, element.units.get(cls.name))
