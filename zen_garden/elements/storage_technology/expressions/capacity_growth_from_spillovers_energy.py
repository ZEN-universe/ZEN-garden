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
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.technology.diffusion import storage_energy_diffusion_rate
from zen_garden.model.component_types.expression import GenericExpression


class CapacityGrowthFromSpilloversEnergy(GenericExpression):
    """Storage energy growth supported by additions at all nodes."""

    name = "capacity_growth_from_spillovers_energy"
    unit_category = {"energy_quantity": 1}
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        return model_constructor.optimization_model.parameters[
            KnowledgeSpilloverRate
        ] != float("inf")

    @classmethod
    def get_expression(cls, model_constructor):
        r"""Construct learning spillovers for storage energy capacity.

        Summary:
            Include depreciated additions at the same and other nodes.

        Formulation:

        .. math::

            d_{h,n,y}\sum_{v<y}\left[\Delta E_{h,n,v}
            +s_h\sum_{n'\ne n}\Delta E_{h,n',v}\right]
            (1-\delta_h)^{\Delta y(y-1-v)}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Storage energy growth supported by node spillovers.
        """
        model = model_constructor.optimization_model
        addition = model.variables[CapacityAdditionEnergy]
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
            return addition.where(xr.DataArray(False))
        interval = model.parameters[IntervalBetweenYears]
        depreciation = model.parameters[KnowledgeDepreciationRate]
        factors = pd.Series(
            {
                (year, previous): (1 - depreciation)
                ** (interval * (year - 1 - previous))
                for year, previous in years
            }
        )
        factors.index.names = ["set_years", "set_years_prev"]
        previous_years = pd.Series(index=years, data=1).to_xarray().fillna(0)
        additions = addition.rename({"set_years": "set_years_prev"}).broadcast_like(
            previous_years
        )
        factor_array = factors.to_xarray().broadcast_like(additions.lower).fillna(0)
        node_index = pd.Series(
            index=pd.MultiIndex.from_product(
                [model.sets[SetNodes], model.sets[SetNodes]],
                names=["set_nodes", "set_nodes_other"],
            )
        ).to_xarray()
        additions_at_nodes = (
            additions.rename({"set_nodes": "set_nodes_other"})
            .broadcast_like(node_index)
            .sum("set_nodes_other")
        )
        spillovers = additions_at_nodes - additions
        knowledge = additions + model.parameters[KnowledgeSpilloverRate] * spillovers
        return storage_energy_diffusion_rate(model_constructor) * (
            knowledge * factor_array
        ).sum("set_years_prev")
