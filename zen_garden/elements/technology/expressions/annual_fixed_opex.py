"""Annual fixed operating expenditure."""

from zen_garden.elements.storage_technology.parameters.opex_specific_fixed_energy import (  # noqa: E501
    OpexSpecificFixedEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_energy import (
    CapacityEnergy,
)
from zen_garden.elements.technology.parameters.opex_specific_fixed import (
    OpexSpecificFixed,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.model.component_types.expression import GenericExpression


class AnnualFixedOpex(GenericExpression):
    """Calculate fixed OPEX from installed power and energy capacity."""

    name = "annual_fixed_opex"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct calculate fixed opex from installed power and energy capacity.

        Summary:
            Calculate fixed OPEX from installed power and energy capacity.

        Formulation:

        .. math::

            \\sum_qk^{opex,fixed}_{h,q,p,y}K_{h,q,p,y}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        optimization_model = model_constructor.optimization_model
        power_cost = (
            optimization_model.parameters[OpexSpecificFixed]
            * optimization_model.variables[Capacity]
        )
        if not optimization_model.sets[SetStorageTechnologies]:
            return power_cost
        storage_energy_cost = (
            optimization_model.parameters[OpexSpecificFixedEnergy]
            * optimization_model.variables[CapacityEnergy]
        ).rename(
            {
                "set_storage_technologies": "set_technologies",
                "set_nodes": "set_location",
            }
        )
        return power_cost + storage_energy_cost
