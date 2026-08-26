import numpy as np

from zen_garden.topology.generic_variable import GenericVariable


class FlowTransportLoss(GenericVariable):
    """Variable for carrier flow loss through transport technology."""

    name = "flow_transport_loss"
    indices = ["set_transport_technologies", "set_edges", "set_time_steps_operation"]
    doc = "Variable for carrier flow lost due to resistances etc. by transporting carrier through transport technology on edge i and time t"
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls):
        return 0, np.inf
