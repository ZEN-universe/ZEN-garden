from zen_garden.topology.generic_parameter import GenericParameter


class CapexCapacityExisting(GenericParameter):
    """Total outstanding capex of an existing technology."""

    name = "capex_capacity_existing"
    indices = (
        "set_technologies",
        "set_capacity_types",
        "set_location",
        "set_technologies_existing",
    )
    doc = "Total outstanding capex of an existing technology"
    unit_category = {"money": 1}
    capacity_types = True
    dependencies = [
        "capacity_existing",
        "opex_specific_fixed",
        "capex_specific_conversion",
        "capex_specific_storage",
        "capex_specific_transport",
        "capex_per_distance_transport",
    ]

    @classmethod
    def store_input_data(cls, element):
        """Annualize costs and materialize persistent existing-capacity capex."""
        fraction_year = element.calculate_fraction_of_year()
        annualized_attributes = (
            "opex_specific_fixed",
            "opex_specific_fixed_energy",
            "capex_specific_conversion",
            "capex_specific_storage",
            "capex_specific_storage_energy",
            "capex_specific_transport",
            "capex_per_distance_transport",
        )
        for attribute in annualized_attributes:
            if hasattr(element, attribute):
                value = getattr(element, attribute)
                setattr(element, attribute, value * fraction_year)

        element.capex_capacity_existing = (
            element.calculate_capex_of_capacities_existing()
        )
        if hasattr(element, "capacity_existing_energy"):
            element.capex_capacity_existing_energy = (
                element.calculate_capex_of_capacities_existing(storage_energy=True)
            )
