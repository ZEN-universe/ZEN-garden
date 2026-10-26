from zen_garden.elements.storage_technology.expressions.storage_overnight_capex_energy import (  # noqa: E501
    StorageOvernightCapexEnergy,
)
from zen_garden.elements.storage_technology.masks import (
    EnergyCapacityAdditionEligibleMask,
)
from zen_garden.elements.storage_technology.variables.cost_capex_overnight_energy import (  # noqa: E501
    CostCapexOvernightEnergy,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageTechnologyCapexEnergyConstraint(GenericConstraint):
    """Set storage overnight CAPEX from energy-capacity additions."""

    name = "constraint_storage_technology_capex_energy"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    masks = (EnergyCapacityAdditionEligibleMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_constraint(cls, model_constructor):
        r"""Construct storage energy-capacity investment costs.

        Summary:
            Set overnight CAPEX from storage energy-capacity additions.

        Formulation:

        .. math::

            C^{capex,overnight,E}_{h,n,y}
            = k^{capex,storage,E}_{h,n,y}\Delta E_{h,n,y}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint defining storage energy-capacity investment costs.
        """
        model = model_constructor.optimization_model
        return (
            model.variables[CostCapexOvernightEnergy]
            == model.expressions[StorageOvernightCapexEnergy]
        )
