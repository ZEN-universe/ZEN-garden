"""Storage energy capacity commissioned from modeled investments."""

from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
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
from zen_garden.model.linear_expression import transpose_linear_expression


class CapacityAdditionsFromNewInvestmentsEnergy(GenericExpression):
    """Map storage energy investments to their commissioning years."""

    name = "capacity_additions_from_new_investments_energy"
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        r"""Construct commissioned storage energy capacity.

        Summary:
            Map modeled storage energy investments to completion years.

        Formulation:

        .. math::

            \sum_{v\in\mathcal Y:v+\tau_h^{construction}=y}
            E^{investment}_{h,n,v}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Storage energy capacity commissioned from modeled investments.
        """
        model = model_constructor.optimization_model
        addition = model.variables[CapacityAdditionEnergy]
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
        current = current.transpose(*addition.dims, "set_time_steps_construction")
        investment = investment.rename(
            {"set_years": "set_time_steps_construction"}
        ).broadcast_like(current)
        investment = transpose_linear_expression(1 * investment, current.dims)
        return (current * investment).sum("set_time_steps_construction")
