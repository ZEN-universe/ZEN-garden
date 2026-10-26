import itertools

import pandas as pd
import xarray as xr

from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.parameters.knowledge_depreciation_rate import (
    KnowledgeDepreciationRate,
)
from zen_garden.elements.energy_system.parameters.knowledge_spillover_rate import (
    KnowledgeSpilloverRate,
)
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.diffusion import (
    diffusion_rate,
    technology_type_mask,
)
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.model.component_types.expression import GenericExpression


class CapacityGrowthFromSpillovers(GenericExpression):
    """Capacity growth supported by local and cross-location installation knowledge."""

    name = "capacity_growth_from_spillovers"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        return model_constructor.optimization_model.parameters[
            KnowledgeSpilloverRate
        ] != float("inf")

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Capacity growth supported by local and cross-location installation
            knowledge.

        Formulation:

        .. math::

            d_{h,p,y} \\sum_{v<y}[ \\Delta K_{h,q,p,v} +s_h \\sum_{p\\prime\\ne p}
            \\Delta K_{h,q,p\\prime,v}](1-\\delta_h)^{ \\Delta y(y-1-v)}

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
        location_index = pd.Series(
            index=pd.MultiIndex.from_product(
                [
                    capacity_addition.coords["set_location"].values,
                    capacity_addition.coords["set_location"].values,
                ],
                names=["set_location", "set_location_temp"],
            )
        ).to_xarray()
        additions_at_nodes = (
            additions.rename({"set_location": "set_location_temp"})
            .broadcast_like(location_index)
            .sel(set_location_temp=model.sets[SetNodes])
            .sum("set_location_temp")
        )
        spillovers = additions_at_nodes - additions
        spillover_rate = xr.full_like(
            spillovers.const, model.parameters[KnowledgeSpilloverRate]
        )
        type_mask = technology_type_mask(model_constructor)
        node_mask = capacity_addition.coords["set_location"].isin(
            list(model.sets[SetNodes])
        )
        node_mask = node_mask.reindex(
            set_location=spillover_rate.coords["set_location"], fill_value=False
        )
        knowledge = (
            additions
            + spillover_rate.where(type_mask, 0).where(node_mask, 0) * spillovers
        )
        return diffusion_rate(model_constructor) * (knowledge * factors).sum(
            "set_years_prev"
        )
