from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.storage_technology.masks import StorageStateMask
from zen_garden.elements.storage_technology.variables.capacity_energy import (
    CapacityEnergy,
)
from zen_garden.elements.storage_technology.variables.storage_level import StorageLevel
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class StorageLevelMaxConstraint(GenericConstraint):
    """Storage Level Max."""

    name = "constraint_storage_level_max"
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours_storage",
    )
    masks = (StorageStateMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Limit maximum storage level to capacity.

        Formulation:

        .. math::
            S^{\\mathrm{level}}_{h,n,\\tilde{t}} \\le K^{\\mathrm{energy}}_{h,n,y}

        Notation:

        :math:`S^{\\mathrm{level}}_{h,n,\\tilde{t}}`: storage level of storage
        technology :math:`h` at node :math:`n` in storage time step :math:`\\tilde{t}`
        of year :math:`y`
        :math:`K^{\\mathrm{energy}}_{h,n,y}`: energy capacity of storage technology
        :math:`h` on node :math:`n` in year :math:`y`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        techs = model_constructor.optimization_model.sets[SetStorageTechnologies]
        nodes = model_constructor.optimization_model.sets[SetNodes]
        capacity = model_constructor.optimization_model.variables[CapacityEnergy].sel(
            set_nodes=nodes, set_storage_technologies=techs
        )
        storage_level = model_constructor.optimization_model.variables[StorageLevel]
        lhs = storage_level - capacity
        rhs = 0
        constraints = lhs <= rhs

        return constraints
