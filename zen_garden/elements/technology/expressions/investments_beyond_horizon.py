"""Investment decisions whose commissioning occurs after the horizon."""

from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.construction_time import (
    get_investment_time_mapping,
)
from zen_garden.elements.technology.variables.capacity_investment import (
    CapacityInvestment,
)
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.linear_expression import align_and_mask


class InvestmentsBeyondHorizon(GenericExpression):
    """Select investments that cannot commission inside the modeled horizon."""

    name = "investments_beyond_horizon"
    indices = (
        "set_technologies",
        "set_location",
        "set_time_steps_construction",
    )

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Select investments that cannot commission inside the modeled horizon.

        Formulation:

        .. math::

            \\{K^{investment}_{h,q,p,v}:v+\\tau_h^{construction}\\notin\\mathcal Y\\}

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
        investment = optimization_model.variables[CapacityInvestment]
        mapping = get_investment_time_mapping(model_constructor)
        current = mapping[
            mapping.index.get_level_values("set_time_steps_construction").isin(
                optimization_model.sets[SetYears]
            )
        ]
        current = current.to_xarray().broadcast_like(investment.mask).fillna(0)
        outside = (1 - current).min("set_years")
        renamed = investment.rename({"set_years": "set_time_steps_construction"})
        return align_and_mask(renamed, outside)
