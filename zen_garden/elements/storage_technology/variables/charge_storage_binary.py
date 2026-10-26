from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.model.component_types.variable import GenericVariable


class ChargeStorageBinary(GenericVariable):
    """Variable for charge storage binary."""

    name = "charge_storage_binary"
    symbol = r"z^{\mathrm{ch}}_{h,n,t}"
    indices = [
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    ]
    doc = (
        "Binary variable indicating whether the storage technology is in "
        "charging mode (1) or discharging mode (0)."
    )
    unit_category = {}
    variable_type = "binary"
    masks = [PowerOperationMask]
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None

    @classmethod
    def should_construct(cls, model_constructor):
        return model_constructor.config.system.storage_charge_discharge_binary
