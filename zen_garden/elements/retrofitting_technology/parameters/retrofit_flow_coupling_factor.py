from zen_garden.topology.generic_parameter import GenericComputedParameters


class RetrofitFlowCouplingFactor(GenericComputedParameters):
    """Flow coupling between a retrofitting technology and its base technology."""

    name = "retrofit_flow_coupling_factor"
    indices = ("set_retrofitting_technologies", "set_nodes", "set_hours")
    doc = "Flow coupling between a retrofitting technology and its base technology"
    unit_category = {}
    time_series = True
    dependencies = ["conversion_factor"]

    @classmethod
    def store_input_data(cls, element, loader):
        loader.load_into(cls, element)
