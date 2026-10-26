import numpy as np

from zen_garden.elements.storage_technology.masks import StorageOperationMask
from zen_garden.model.component_types.variable import GenericVariable


class FlowStorageSpillage(GenericVariable):
    """Variable for storage spillage."""

    name = "flow_storage_spillage"
    symbol = r"F^{\mathrm{spill}}_{h,n,t}"
    indices = [
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Energy spillage for storage technology."
    unit_category = {"energy_quantity": 1, "time": -1}
    masks = [StorageOperationMask]
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
