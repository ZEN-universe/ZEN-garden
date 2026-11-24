from zen_garden.elements.storage_technology.expressions.storage_overnight_capex_energy import (  # noqa: E501
    StorageOvernightCapexEnergy,
)
from zen_garden.elements.storage_technology.masks import (
    EnergyCapacityAdditionEligibleMask,
)
from zen_garden.model.component_types.expression import GenericExpression


class CostCapexOvernightEnergy(GenericExpression):
    """Overnight cost of storage energy-capacity additions."""

    name = "cost_capex_overnight_energy"
    doc = r"""Overnight storage energy-capacity investment cost.

    .. math::

        \kappa^{\mathrm{capex,E}}_{h,n,y}\Delta E_{h,n,y}
    """
    unit_category = {"money": 1}
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    dependencies = (StorageOvernightCapexEnergy,)
    masks = (EnergyCapacityAdditionEligibleMask,)
    mask_default_value = 0.0

    @classmethod
    def get_expression(cls, model_constructor):
        return model_constructor.optimization_model.expressions[
            StorageOvernightCapexEnergy
        ]
