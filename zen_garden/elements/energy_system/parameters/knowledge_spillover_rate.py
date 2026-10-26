from zen_garden.model.component_types.parameter import GenericParameter


class KnowledgeSpilloverRate(GenericParameter):
    """Knowledge spillover rate."""

    name = "knowledge_spillover_rate"
    symbol = r"\omega"
    indices = ()
    doc = "Parameter which specifies the knowledge spillover rate."
    unit_category = {}
