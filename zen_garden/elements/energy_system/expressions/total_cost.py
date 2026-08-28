from zen_garden.model.component_types.expression import GenericExpression


class TotalCost(GenericExpression):
    """Total net present cost objective expression.

    .. math::
        J = \\sum_{y\\in\\mathcal{Y}} NPC_y
    """

    name = "total_cost"
    doc = "Total net present cost, summed over all modeled years"

    @classmethod
    def get_expression(cls, model_constructor):
        return model_constructor.optimization_model.variables["net_present_cost"].sum(
            "set_years"
        )
