"""Shared helpers for technology structural masks."""

import numpy as np

from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.sets.set_years import SetYears


def year_offset(model_constructor, technology: str, parameter: str, *, ceil=False):
    """Return a duration parameter as a number of modeled year intervals."""
    value = getattr(model_constructor.optimization_model.parameters, parameter).sel(
        set_technologies=technology
    )
    interval = model_constructor.optimization_model.parameters[IntervalBetweenYears]
    ratio = float(value.item()) / interval
    return int(np.ceil(ratio) if ceil else np.floor(ratio))


def active_previous_years(
    model_constructor, technology: str, year, parameter: str
) -> list:
    """Return modeled earlier years still active under a duration parameter."""
    years = list(model_constructor.optimization_model.sets[SetYears])
    duration = year_offset(model_constructor, technology, parameter)
    first_year = year - (duration - 1)
    return [previous for previous in years if first_year <= previous <= year]
