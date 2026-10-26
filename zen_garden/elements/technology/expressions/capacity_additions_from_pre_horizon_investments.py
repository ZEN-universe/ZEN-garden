"""Capacity commissioned from investments made before the model horizon."""

from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.energy_system.sets.set_years_entire_horizon import (
    SetYearsEntireHorizon,
)
from zen_garden.elements.technology.construction_time import (
    get_investment_time_mapping,
)
from zen_garden.elements.technology.parameters.capacity_investment_existing import (
    CapacityInvestmentExisting,
)
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.model.component_types.expression import GenericExpression


class CapacityAdditionsFromPreHorizonInvestments(GenericExpression):
    """Map pre-horizon investments to their modeled commissioning years."""

    name = "capacity_additions_from_pre_horizon_investments"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct map pre-horizon investments to their modeled commissioning years.

        Summary:
            Map pre-horizon investments to their modeled commissioning years.

        Formulation:

        .. math::

             \\sum_{v\\notin \\mathcal Y:v +\\tau_h^{construction}
            =y}k^{investment,existing}_{h,q,p,v}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        optimization_model = model_constructor.optimization_model
        addition = optimization_model.variables[CapacityAddition]
        mapping = get_investment_time_mapping(model_constructor)
        investment_years = mapping.index.get_level_values("set_time_steps_construction")
        existing = mapping[
            investment_years.isin(optimization_model.sets[SetYearsEntireHorizon])
            & ~investment_years.isin(optimization_model.sets[SetYears])
        ]
        parameter = optimization_model.parameters[CapacityInvestmentExisting].rename(
            {"set_years_entire_horizon": "set_time_steps_construction"}
        )
        existing = existing.to_xarray().broadcast_like(parameter).fillna(0)
        existing = existing.transpose(*addition.dims, "set_time_steps_construction")
        result = (existing * parameter.broadcast_like(existing)).sum(
            "set_time_steps_construction"
        )
        return result.transpose(*addition.dims)
