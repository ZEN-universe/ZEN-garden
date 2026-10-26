"""Retrofitting technology expressions."""

from zen_garden.model.component_types.expression import GenericExpression

from .retrofit_activity import RetrofitActivity
from .retrofit_base_activity import RetrofitBaseActivity

RETROFITTING_TECHNOLOGY_EXPRESSIONS: list[type[GenericExpression]] = [
    RetrofitActivity,
    RetrofitBaseActivity,
]

__all__ = [
    "RETROFITTING_TECHNOLOGY_EXPRESSIONS",
    "RetrofitActivity",
    "RetrofitBaseActivity",
]
