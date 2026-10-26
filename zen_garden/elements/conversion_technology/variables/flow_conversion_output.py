from zen_garden.elements.conversion_technology.variables.flow_conversion_input import (
    FlowConversionInput,
)


class FlowConversionOutput(FlowConversionInput):
    """Variable for carrier output of conversion technologies."""

    name = "flow_conversion_output"
    symbol = r"F^{\mathrm{conv,out}}_{h,c^{\mathrm{out}},n,t}"
    indices = [
        "set_conversion_technologies",
        "set_output_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Carrier output of conversion technologies."
    unit_category = {"energy_quantity": 1, "time": -1}
