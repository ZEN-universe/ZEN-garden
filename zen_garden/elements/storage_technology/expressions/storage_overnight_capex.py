"""Overnight CAPEX of storage-capacity additions."""

from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.storage_technology.parameters.capex_specific_storage import (
    CapexSpecificStorage,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.model.component_types.expression import GenericExpression


class StorageOvernightCapex(GenericExpression):
    """Multiply storage power additions by their specific CAPEX."""

    name = "storage_overnight_capex"
    unit_category = {"money": 1}
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Multiply storage power and energy additions by their specific CAPEX.

        Formulation:

        .. math::

            k^{capex,storage}_{h,q,n,y}\\Delta K_{h,q,n,y}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        model = model_constructor.optimization_model
        technologies = model.sets[SetStorageTechnologies]
        nodes = model.sets[SetNodes]
        addition = model.variables[CapacityAddition].sel(
            set_technologies=technologies, set_location=nodes
        )
        specific = model.parameters[CapexSpecificStorage].rename(
            set_storage_technologies="set_technologies", set_nodes="set_location"
        )
        return specific * addition
