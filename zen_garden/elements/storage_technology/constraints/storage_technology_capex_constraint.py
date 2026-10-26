from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.storage_technology.expressions.storage_overnight_capex import (
    StorageOvernightCapex,
)
from zen_garden.elements.technology.masks import CapacityAdditionEligibleMask
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.elements.technology.variables.cost_capex_overnight import (
    CostCapexOvernight,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageTechnologyCapexConstraint(GenericConstraint):
    """Set storage overnight CAPEX from power-capacity additions."""

    name = "constraint_storage_technology_capex"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (CapacityAdditionEligibleMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct the model component.

        Summary:
            Set storage overnight CAPEX from power- and energy-capacity additions.

        Formulation:

        .. math::

            C^{capex,overnight}=k^{capex,storage}\\Delta K

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        cost = model.variables[CostCapexOvernight].sel(
            set_technologies=model.sets[SetStorageTechnologies],
            set_location=model.sets[SetNodes],
        )
        return cost == model.expressions[StorageOvernightCapex]
