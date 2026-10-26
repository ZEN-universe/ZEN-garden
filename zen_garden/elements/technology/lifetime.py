"""Lifetime and depreciation-period calculations for technology vintages."""

from typing import TYPE_CHECKING

import numpy as np

from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.technology.parameters.depreciation_time import DepreciationTime
from zen_garden.elements.technology.parameters.lifetime import Lifetime

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


def get_active_vintage_years(
    model_constructor: "ModelConstructor",
    technology: str,
    year: int,
    *,
    use_depreciation_time: bool = False,
) -> range:
    """Return modeled investment years still active in ``year``."""
    first_year = _get_first_active_vintage_year(
        model_constructor,
        technology,
        year,
        use_depreciation_time=use_depreciation_time,
    )
    first_modeled_year = model_constructor.model_schema.set_years[0]
    return range(max(first_year, first_modeled_year), year + 1)


def _get_first_active_vintage_year(
    model_constructor: "ModelConstructor",
    technology: str,
    year: int,
    *,
    use_depreciation_time: bool = False,
) -> int:
    """Return the earliest investment year still active in ``year``."""
    parameters = model_constructor.optimization_model.parameters.dict_parameters
    duration = (
        parameters[DepreciationTime][technology]
        if use_depreciation_time
        else parameters[Lifetime][technology]
    )
    interval = model_constructor.optimization_model.parameters[IntervalBetweenYears]
    active_intervals = int(np.floor(duration / interval)) - 1
    return year - active_intervals
