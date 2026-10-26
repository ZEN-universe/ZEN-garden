from zen_garden.model.component_types.parameter import GenericParameter


class KnowledgeDepreciationRate(GenericParameter):
    """Knowledge depreciation rate."""

    name = "knowledge_depreciation_rate"
    symbol = r"\delta^{\mathrm{knowledge}}_{h}"
    indices = ()
    doc = "Annual rate at which accumulated deployment knowledge depreciates."
    unit_category = {}
