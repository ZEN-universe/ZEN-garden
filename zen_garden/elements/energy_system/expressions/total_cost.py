from zen_garden.elements.energy_system.expressions.net_present_cost import (
    NetPresentCost,
)
from zen_garden.model.component_types.expression import GenericExpression


class TotalCost(GenericExpression):
    """Net present cost summed explicitly over all modeled years."""

    name = "total_cost"
    unit_category = {"money": 1}
    indices = ()
    doc = r"""Total net present cost: sum the complete expanded annual term below
        over every modeled year. No implicit reduction is applied.

        .. math::

            \sum_{y\in\mathcal Y}\left[\text{expanded per-year term below}\right]

        """ + NetPresentCost.doc
    dependencies = (NetPresentCost,)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct total net present cost objective expression.

        Summary:
            Total net present cost objective expression.

        The complete mathematical term is expanded in :attr:`doc`.

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        return model_constructor.optimization_model.expressions[NetPresentCost].sum(
            "set_years"
        )
