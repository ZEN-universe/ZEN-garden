"""Retrofitting Technology constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint
from zen_garden.elements.retrofitting_technology.constraints.retrofit_flow_coupling_constraint import (
    RetrofitFlowCouplingConstraint,
)
from zen_garden.elements.retrofitting_technology.retrofitting_technology import (
    RetrofittingTechnology,
)

RETROFITTING_TECHNOLOGY_CONSTRAINTS: list[type[GenericConstraint]] = [
    RetrofitFlowCouplingConstraint,
]

__all__ = [
    "RetrofittingTechnology",
    "RetrofitFlowCouplingConstraint",
]
