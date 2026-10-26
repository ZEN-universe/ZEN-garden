from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.storage_technology.expressions.storage_throughput import (
    StorageThroughput,
)
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.parameters.opex_specific_variable import (
    OpexSpecificVariable,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.elements.technology.variables.cost_opex_variable import CostOpexVariable
from zen_garden.model.component_types.constraint import GenericConstraint


class OpexTechnologyStorageConstraint(GenericConstraint):
    """Calculate variable OPEX from storage throughput."""

    name = "constraint_opex_technology_storage"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )
    masks = (PowerOperationMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct calculate variable opex from storage throughput.

        Summary:
            Calculate variable OPEX from storage throughput.

        Formulation:

        .. math::

            C^{opex,var}=k^{opex,var}(F^{ch}+F^{dis})

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
        technologies = model.sets[SetStorageTechnologies]
        nodes = model.sets[SetNodes]
        cost = model.variables[CostOpexVariable].sel(
            set_technologies=technologies, set_location=nodes
        )
        return (
            cost
            == model.parameters[OpexSpecificVariable]
            * model.expressions[StorageThroughput]
        )
