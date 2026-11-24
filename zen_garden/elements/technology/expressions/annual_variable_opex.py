"""Annual variable operating expenditure."""

from zen_garden.elements.energy_system.parameters.weight_representative_hours import (
    WeightRepresentativeHours,
)
from zen_garden.elements.technology.expressions.cost_opex_variable import (
    CostOpexVariable,
)
from zen_garden.model.component_types.expression import GenericExpression


class AnnualVariableOpex(GenericExpression):
    """Weight and sum variable OPEX over representative hours."""

    name = "annual_variable_opex"
    unit_category = {"money": 1}
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (CostOpexVariable,)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct weight and sum variable opex over representative hours.

        Summary:
            Weight and sum variable OPEX over representative hours.

        Formulation:

        .. math::

            \\sum_rw_rC^{opex,var}_{h,p,y,r}

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
        return (
            optimization_model.expressions[CostOpexVariable]
            * optimization_model.parameters[WeightRepresentativeHours]
        ).sum("set_representative_hours")
