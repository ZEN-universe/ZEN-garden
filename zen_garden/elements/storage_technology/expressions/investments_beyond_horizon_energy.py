"""Storage energy investments completed after the modeled horizon."""

from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.storage_technology.variables.capacity_investment_energy import (  # noqa: E501
    CapacityInvestmentEnergy,
)
from zen_garden.elements.technology.construction_time import (
    get_investment_time_mapping,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.linear_expression import align_and_mask


class InvestmentsBeyondHorizonEnergy(GenericExpression):
    """Select storage energy investments that cannot finish in the horizon."""

    name = "investments_beyond_horizon_energy"
    unit_category = {"energy_quantity": 1}
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_time_steps_construction",
    )

    @classmethod
    def get_expression(cls, model_constructor):
        r"""Construct storage energy investments beyond the horizon.

        Summary:
            Select storage energy investments completed after the final year.

        Formulation:

        .. math::

            \{E^{investment}_{h,n,v}:v+\tau_h^{construction}\notin\mathcal Y\}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Storage energy investments completed beyond the horizon.
        """
        model = model_constructor.optimization_model
        investment = model.variables[CapacityInvestmentEnergy]
        mapping = get_investment_time_mapping(model_constructor).rename_axis(
            index={"set_technologies": "set_storage_technologies"}
        )
        mapping = mapping.loc[model.sets[SetStorageTechnologies]]
        current = mapping[
            mapping.index.get_level_values("set_time_steps_construction").isin(
                model.sets[SetYears]
            )
        ]
        current = current.to_xarray().broadcast_like(investment.mask).fillna(0)
        outside = (1 - current).min("set_years")
        renamed = investment.rename({"set_years": "set_time_steps_construction"})
        return align_and_mask(renamed, outside)
