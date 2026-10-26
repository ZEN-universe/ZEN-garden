"""Construction-time mappings shared by capacity expressions."""

import itertools
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.parameters.construction_time import ConstructionTime

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor


def _get_investment_time_step(
    model_constructor: "ModelConstructor", technology: str, year: int
) -> int:
    """Return the investment year that commissions capacity in ``year``."""
    parameters = model_constructor.optimization_model.parameters.dict_parameters
    construction_time = parameters[ConstructionTime][technology]
    interval = model_constructor.optimization_model.parameters[IntervalBetweenYears]
    return year - int(np.ceil(construction_time / interval))


def get_investment_time_mapping(
    model_constructor: "ModelConstructor",
) -> pd.Series:
    """Return commissioning years mapped to their required investment years."""
    optimization_model = model_constructor.optimization_model
    mapping = pd.Series(
        {
            (
                technology,
                year,
                _get_investment_time_step(model_constructor, technology, year),
            ): 1
            for technology, year in itertools.product(
                optimization_model.sets[SetTechnologies],
                optimization_model.sets[SetYears],
            )
        }
    )
    mapping.index.names = [
        "set_technologies",
        "set_years",
        "set_time_steps_construction",
    ]
    return mapping
