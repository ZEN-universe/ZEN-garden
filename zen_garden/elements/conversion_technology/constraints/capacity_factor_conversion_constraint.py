from zen_garden.elements.conversion_technology.expressions.conversion_reference_flow import (  # noqa: E501
    ConversionReferenceFlow,
)
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.parameters.max_load import MaxLoad
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.model.component_types.constraint import GenericConstraint


class CapacityFactorConversionConstraint(GenericConstraint):
    """Capacity Factor Conversion."""

    name = "constraint_capacity_factor_conversion"
    indices = (
        "set_conversion_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    masks = (PowerOperationMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(
            model_constructor.optimization_model.sets[SetConversionTechnologies]
        )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Load is limited by the installed capacity and the maximum load factor.

        Formulation:

        .. math::
            F^{\\mathrm{ref}}_{h,n,t} \\leq
            \\ell^{\\mathrm{max}}_{h,n,t}K_{h,n,y}

        Notation:

        :math:`m_{h,n,t}^{\\mathrm{max}}`: maximum load factor of the
        technology :math:`h` at node :math:`n` in time step :math:`t` of year
        :math:`y`
        :math:`K_{h,n,y}`: installed capacity of the technology :math:`h` at
        node :math:`n` in year :math:`y`
        :math:`F^{\\mathrm{ref}}_{h,n,t}`: reference carrier flow of the
        technology :math:`h` at node :math:`n` in time step :math:`t` of year
        :math:`y`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        techs = model_constructor.optimization_model.sets[SetConversionTechnologies]
        nodes = model_constructor.optimization_model.sets[SetNodes]
        term_capacity = (
            model_constructor.optimization_model.parameters[MaxLoad].loc[
                techs, nodes, :, :
            ]
            * model_constructor.optimization_model.variables[Capacity].loc[
                techs, nodes, :
            ]
        ).rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        )
        term_reference_flow = model_constructor.optimization_model.expressions[
            ConversionReferenceFlow
        ]
        lhs = term_capacity - term_reference_flow
        rhs = 0
        constraints = lhs >= rhs

        return constraints
