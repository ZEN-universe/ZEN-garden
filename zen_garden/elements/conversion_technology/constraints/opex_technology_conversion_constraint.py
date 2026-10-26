from zen_garden.elements.conversion_technology.expressions.conversion_reference_flow import (  # noqa: E501
    ConversionReferenceFlow,
)
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.parameters.opex_specific_variable import (
    OpexSpecificVariable,
)
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.elements.technology.variables.cost_opex_variable import CostOpexVariable
from zen_garden.model.component_types.constraint import GenericConstraint


class OpexTechnologyConversionConstraint(GenericConstraint):
    """Calculate variable OPEX from conversion reference flow."""

    name = "constraint_opex_technology_conversion"
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
        """Construct calculate variable opex from conversion reference flow.

        Summary:
            Calculate variable OPEX from conversion reference flow.

        Formulation:

        .. math::

            C^{opex,var}=k^{opex,var}F^{ref}

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
        technologies = model.sets[SetConversionTechnologies]
        nodes = model.sets[SetNodes]
        cost = (
            model.variables[CostOpexVariable]
            .sel(set_technologies=technologies, set_location=nodes)
            .rename(
                set_technologies="set_conversion_technologies",
                set_location="set_nodes",
            )
        )
        factor = model.parameters[OpexSpecificVariable].rename(
            set_technologies="set_conversion_technologies",
            set_location="set_nodes",
        )
        return cost == factor * model.expressions[ConversionReferenceFlow]
