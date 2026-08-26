from zen_garden.topology.generic_parameter import GenericComputedParameters


class DepreciationTime(GenericComputedParameters):
    """Depreciation time of a newly built technology."""

    name = "depreciation_time"
    indices = ("set_technologies",)
    doc = "Depreciation time of a newly built technology"
    unit_category = {}
    input_loader = "depreciation_time"
    dependencies = ["lifetime"]

    @classmethod
    def store_input_data(cls, element, loader):
        loader.load_into(cls, element)
