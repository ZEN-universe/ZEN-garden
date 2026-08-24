"""Retrofitting technology constraints."""

from zen_garden.topology.generic_constraint import GenericConstraint

from .retrofit_flow_coupling_constraint import (
    RetrofitFlowCouplingConstraint,
)

RETROFITTING_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    RetrofitFlowCouplingConstraint,
]

__all__ = ["RetrofitFlowCouplingConstraint"]
