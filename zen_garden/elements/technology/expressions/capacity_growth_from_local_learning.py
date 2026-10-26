import itertools

import pandas as pd
import xarray as xr

from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.parameters.knowledge_depreciation_rate import (
    KnowledgeDepreciationRate,
)
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.diffusion import diffusion_rate
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.model.component_types.expression import GenericExpression


class CapacityGrowthFromLocalLearning(GenericExpression):
    """Capacity growth supported by depreciated additions at the same location."""

    name = "capacity_growth_from_local_learning"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Capacity growth supported by depreciated additions at the same location.

        Formulation:

        .. math::

            d_{h,p,y}\\sum_{v<y}\\Delta K_{h,q,p,v}(1-\\delta_h)^{\\Delta y(y-1-v)}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        model = model_constructor.optimization_model
        capacity_addition = model.variables[CapacityAddition]
        years = pd.MultiIndex.from_tuples(
            [
                (year, previous)
                for year, previous in itertools.product(
                    model.sets[SetYears], model.sets[SetYears]
                )
                if previous < year
            ],
            names=["set_years", "set_years_prev"],
        )
        if not len(years):
            return capacity_addition.where(xr.DataArray(False))

        interval = model.parameters[IntervalBetweenYears]
        depreciation = model.parameters[KnowledgeDepreciationRate]
        factors_series = pd.Series(
            {
                (year, previous): (1 - depreciation)
                ** (interval * (year - 1 - previous))
                for year, previous in years
            }
        )
        factors_series.index.names = ["set_years", "set_years_prev"]
        previous_years = pd.Series(index=years, data=1).to_xarray().fillna(0)
        additions = capacity_addition.rename(
            {"set_years": "set_years_prev"}
        ).broadcast_like(previous_years)
        factors = factors_series.to_xarray().broadcast_like(additions.lower).fillna(0)
        return diffusion_rate(model_constructor) * (additions * factors).sum(
            "set_years_prev"
        )
