from inspect import cleandoc
from textwrap import indent

from zen_garden.elements.energy_system.expressions.net_present_cost import (
    NetPresentCost,
)
from zen_garden.model.component_types.expression import GenericExpression


class TotalCost(GenericExpression):
    """Net present cost summed explicitly over all modeled years."""

    name = "total_cost"
    unit_category = {"money": 1}
    indices = ()
    discounted_annual_formula = cleandoc(
        NetPresentCost.doc.split(".. math::", maxsplit=1)[1]
    )
    doc = (
        "Total net present cost over all modeled years.\n\n"
        ".. math::\n\n"
        + indent(
            r"\sum_{y\in\mathcal Y}" + "\n" + discounted_annual_formula,
            "    ",
        )
        + "\n\nThe outer sum runs over modeled planning periods :math:`y`. "
        "The inner sum over :math:`i` applies annual discount factors to the "
        "calendar years represented by each planning period."
    )
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
