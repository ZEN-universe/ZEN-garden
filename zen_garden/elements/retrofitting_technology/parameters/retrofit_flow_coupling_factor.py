from zen_garden.elements.conversion_technology.parameters.conversion_factor import (
    ConversionFactor,
)
from zen_garden.model.component_types.parameter import GenericParameter


class RetrofitFlowCouplingFactor(GenericParameter):
    """Flow coupling between a retrofitting technology and its base technology."""

    name = "retrofit_flow_coupling_factor"
    symbol = r"\eta^{\mathrm{retro}}_{h,n,t}"
    indices = (
        "set_retrofitting_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    doc = (
        "Parameter which specifies the flow coupling between a retrofitting "
        "technology and its base technology."
    )
    unit_category = {}
    time_series = True
    dependencies = (ConversionFactor,)
