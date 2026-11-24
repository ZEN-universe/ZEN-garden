"""Capacity commissioned from decisions inside the modeled horizon."""

from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.construction_time import (
    get_investment_time_mapping,
)
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.elements.technology.variables.capacity_investment import (
    CapacityInvestment,
)
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.linear_expression import transpose_linear_expression


class CapacityAdditionsFromNewInvestments(GenericExpression):
    """Map modeled investment decisions to their commissioning years."""

    name = "capacity_additions_from_new_investments"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct map modeled investment decisions to their commissioning years.

        Summary:
            Map modeled investment decisions to their commissioning years.

        Formulation:

        .. math::

            \\sum_{v\\in\\mathcal Y:v+\\tau_h^{construction}=y}K^{investment}_{h,q,p,v}

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
        investment = optimization_model.variables[CapacityInvestment]
        mapping = get_investment_time_mapping(model_constructor)
        current = mapping[
            mapping.index.get_level_values("set_time_steps_construction").isin(
                optimization_model.sets[SetYears]
            )
        ]
        current = current.to_xarray().broadcast_like(investment.mask).fillna(0)
        current = current.transpose(*addition.dims, "set_time_steps_construction")
        investment = investment.rename(
            {"set_years": "set_time_steps_construction"}
        ).broadcast_like(current)
        investment = transpose_linear_expression(1 * investment, current.dims)
        return (current * investment).sum("set_time_steps_construction")
