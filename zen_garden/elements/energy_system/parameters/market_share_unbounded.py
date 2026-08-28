from zen_garden.model.component_types.parameter import GenericParameter


class MarketShareUnbounded(GenericParameter):
    """Unbounded market share."""

    name = "market_share_unbounded"
    indices = ()
    doc = "Unbounded market share"
    unit_category = {}
