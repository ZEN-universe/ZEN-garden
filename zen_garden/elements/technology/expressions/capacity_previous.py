from zen_garden.elements.technology.masks import CapacityAvailableMask
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.model.component_types.expression import GenericExpression


class CapacityPrevious(GenericExpression):
    """Installed capacity before additions made in the current year."""

    name = "capacity_previous"
    doc = r"""Installed power capacity before current-year additions.

    .. math::

        K_{h,p,y}-\Delta K_{h,p,y}
    """
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapacityAvailableMask,)
    mask_default_value = 0.0

    @classmethod
    def get_expression(cls, model_constructor):
        model = model_constructor.optimization_model
        return model.variables[Capacity] - model.variables[CapacityAddition]
