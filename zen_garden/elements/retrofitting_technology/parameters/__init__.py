"""retrofitting technology parameters."""

from zen_garden.model.component_types.parameter import GenericParameter

from .retrofit_flow_coupling_factor import RetrofitFlowCouplingFactor

RETROFITTING_TECHNOLOGY_PARAMETERS: list[type[GenericParameter]] = [
    RetrofitFlowCouplingFactor,
]

__all__ = [
    "RetrofitFlowCouplingFactor",
    "RETROFITTING_TECHNOLOGY_PARAMETERS",
]
