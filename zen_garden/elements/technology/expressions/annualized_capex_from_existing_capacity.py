"""Annualized CAPEX attributable to existing capacity."""

from zen_garden.elements.energy_system.parameters.discount_rate import DiscountRate
from zen_garden.elements.technology.parameters.depreciation_time import DepreciationTime
from zen_garden.elements.technology.parameters.existing_capex import ExistingCapex
from zen_garden.model.component_types.expression import GenericExpression


class AnnualizedCapexFromExistingCapacity(GenericExpression):
    """Annualize remaining overnight CAPEX of existing capacity."""

    name = "annualized_capex_from_existing_capacity"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct annualize remaining overnight capex of existing capacity.

        Summary:
            Annualize remaining overnight CAPEX of existing capacity.

        Formulation:

        .. math::

            a(q,d_h)k^{existing,capex}_{h,p,y}

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
        discount_rate = optimization_model.parameters[DiscountRate]
        depreciation_time = optimization_model.parameters[DepreciationTime]
        annuity = (
            ((1 + discount_rate) ** depreciation_time * discount_rate)
            / ((1 + discount_rate) ** depreciation_time - 1)
            if discount_rate != 0
            else 1 / depreciation_time
        )
        return annuity * optimization_model.parameters[ExistingCapex]
