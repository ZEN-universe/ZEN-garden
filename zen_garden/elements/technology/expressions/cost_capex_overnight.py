import linopy as lp
from linopy.expressions import LinearExpression

from zen_garden.elements.conversion_technology.expressions.conversion_overnight_capex import (  # noqa: E501
    ConversionOvernightCapex,
)
from zen_garden.elements.storage_technology.expressions.storage_overnight_capex import (
    StorageOvernightCapex,
)
from zen_garden.elements.technology.masks import CapacityAdditionEligibleMask
from zen_garden.elements.transport_technology.expressions.transport_overnight_capex import (  # noqa: E501
    TransportOvernightCapex,
)
from zen_garden.model.component_types.expression import GenericExpression


class CostCapexOvernight(GenericExpression):
    """Overnight capacity-addition cost across all technology types."""

    name = "cost_capex_overnight"
    doc = r"""Overnight power-capacity investment cost.

    .. math::

        \begin{cases}
        \kappa^{\mathrm{capex,conv}}_{h,n,y}\Delta K_{h,n,y},
        &h\in\mathcal H^{conv},\\
        \kappa^{\mathrm{capex,stor}}_{h,n,y}\Delta K_{h,n,y},
        &h\in\mathcal H^{stor},\\
        \kappa^{\mathrm{capex,trans}}_{h,e,y}\Delta K_{h,e,y}
        +\kappa^{\mathrm{distance}}_{h,e,y}G_{h,e,y},
        &h\in\mathcal H^{trans}.
        \end{cases}
    """
    unit_category = {"money": 1}
    indices = ("set_technologies", "set_location", "set_years")
    dependencies = (
        ConversionOvernightCapex,
        StorageOvernightCapex,
        TransportOvernightCapex,
    )
    masks = (CapacityAdditionEligibleMask,)
    mask_default_value = 0.0

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        pieces = [
            model.expressions[expression]
            for expression in (
                ConversionOvernightCapex,
                StorageOvernightCapex,
                TransportOvernightCapex,
            )
            if expression in model.expressions
        ]
        return lp.merge(
            pieces, compat="broadcast_equals", join="outer", cls=LinearExpression
        )
