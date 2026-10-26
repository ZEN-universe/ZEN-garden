from zen_garden.model.component_types.parameter import GenericParameter


class MarketShareUnbounded(GenericParameter):
    """Unbounded market share."""

    name = "market_share_unbounded"
    symbol = r"\chi"
    indices = ()
    doc = "Parameter which specifies the unbounded market share."
    unit_category = {}
