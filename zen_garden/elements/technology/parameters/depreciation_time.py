from zen_garden.topology.generic_parameter import GenericParameter


class DepreciationTime(GenericParameter):
    """Depreciation time of a newly built technology."""

    name = "depreciation_time"
    indices = ("set_technologies",)
    doc = "Depreciation time of a newly built technology"
    unit_category = {}
    dependencies = ["lifetime"]

    @classmethod
    def store_input_data(cls, element):
        """Load depreciation time or default it to technology lifetime."""
        if cls.name in element.data_input.attribute_dict:
            value = element.data_input.extract_input_data(
                cls.name, index_sets=[], unit_category={}
            )
            value[0] = max(element.config.system.interval_between_years, value[0])
        else:
            value = element.lifetime.copy()
        cls._store_value(element, cls.name, value)
