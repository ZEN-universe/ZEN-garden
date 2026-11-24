from zen_garden.elements.storage_technology.masks import EnergyCapacityAvailableMask
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_energy import (
    CapacityEnergy,
)
from zen_garden.model.component_types.expression import GenericExpression


class CapacityPreviousEnergy(GenericExpression):
    """Storage energy capacity before current-year additions."""

    name = "capacity_previous_energy"
    doc = r"""Storage energy capacity before current-year additions.

    .. math::

        E_{h,n,y}-\Delta E_{h,n,y}
    """
    unit_category = {"energy_quantity": 1}
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapacityAvailableMask,)
    mask_default_value = 0.0

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        return model.variables[CapacityEnergy] - model.variables[CapacityAdditionEnergy]
