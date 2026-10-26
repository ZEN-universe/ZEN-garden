from zen_garden.model.component_types.parameter import GenericParameter


class OpexSpecificVariable(GenericParameter):
    """Variable specific opex."""

    name = "opex_specific_variable"
    symbol = r"\kappa^{\mathrm{op,var}}_{h,y}"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )
    doc = "Parameter which specifies the variable specific opex."
    unit_category = {"money": 1, "energy_quantity": -1}
    time_series = True
