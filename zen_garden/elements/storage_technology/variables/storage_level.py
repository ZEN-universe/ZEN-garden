import numpy as np

from zen_garden.elements.storage_technology.masks import StorageStateMask
from zen_garden.model.component_types.variable import GenericVariable


class StorageLevel(GenericVariable):
    """Variable for storage level."""

    name = "storage_level"
    symbol = r"S^{\mathrm{level}}_{h,n,\tilde{t}}"
    indices = [
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours_storage",
    ]
    doc = "Storage level of storage technology on node in each storage time step."
    unit_category = {"energy_quantity": 1}
    masks = [StorageStateMask]
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return 0, np.inf
