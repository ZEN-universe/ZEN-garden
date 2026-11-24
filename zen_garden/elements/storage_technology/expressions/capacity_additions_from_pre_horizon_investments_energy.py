"""Storage energy capacity commissioned from pre-horizon investments."""

from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.energy_system.sets.set_years_entire_horizon import (
    SetYearsEntireHorizon,
)
from zen_garden.elements.storage_technology.parameters.capacity_investment_existing_energy import (  # noqa: E501
    CapacityInvestmentExistingEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.technology.construction_time import (
    get_investment_time_mapping,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class CapacityAdditionsFromPreHorizonInvestmentsEnergy(GenericExpression):
    """Map committed storage energy investments to completion years."""

    name = "capacity_additions_from_pre_horizon_investments_energy"
    unit_category = {"energy_quantity": 1}
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        r"""Construct pre-horizon storage energy additions.

        Summary:
            Map pre-horizon storage energy investments to completion years.

        Formulation:

        .. math::

            \sum_{v\notin\mathcal Y:v+\tau_h^{construction}=y}
            e^{investment,existing}_{h,n,v}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Storage energy capacity commissioned from pre-horizon investments.
        """
        model = model_constructor.optimization_model
        addition = model.variables[CapacityAdditionEnergy]
        mapping = get_investment_time_mapping(model_constructor).rename_axis(
            index={"set_technologies": "set_storage_technologies"}
        )
        mapping = mapping.loc[model.sets[SetStorageTechnologies]]
        investment_years = mapping.index.get_level_values("set_time_steps_construction")
        existing = mapping[
            investment_years.isin(model.sets[SetYearsEntireHorizon])
            & ~investment_years.isin(model.sets[SetYears])
        ]
        parameter = model.parameters[CapacityInvestmentExistingEnergy].rename(
            {"set_years_entire_horizon": "set_time_steps_construction"}
        )
        existing = existing.to_xarray().broadcast_like(parameter).fillna(0)
        existing = existing.transpose(*addition.dims, "set_time_steps_construction")
        result = (existing * parameter.broadcast_like(existing)).sum(
            "set_time_steps_construction"
        )
        return result.transpose(*addition.dims)
