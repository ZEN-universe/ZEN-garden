import linopy as lp
from linopy.expressions import LinearExpression

from zen_garden.elements.conversion_technology.expressions.conversion_dependent_flow import (  # noqa: E501
    ConversionDependentFlow,
)
from zen_garden.elements.conversion_technology.expressions.conversion_reference_flow import (  # noqa: E501
    ConversionReferenceFlow,
)
from zen_garden.elements.conversion_technology.parameters.conversion_factor import (
    ConversionFactor,
)
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.utils import align_like


class CarrierConversionConstraint(GenericConstraint):
    """Couple dependent-carrier flows to conversion reference flow."""

    name = "constraint_carrier_conversion"
    indices = (
        "set_conversion_technologies",
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct couple dependent-carrier flows to conversion reference flow.

        Summary:
            Couple dependent-carrier flows to conversion reference flow.

        Formulation:

        .. math::

            F^{dep}_{h,c,n,y,r}=\\phi_{h,c,n,y,r}F^{ref}_{h,n,y,r}

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
        dependent_flow = model.expressions[ConversionDependentFlow]
        conversion_factor = align_like(
            model.parameters[ConversionFactor], dependent_flow
        )
        reference_flow = model.expressions[ConversionReferenceFlow].broadcast_like(
            conversion_factor
        )
        converted_reference_flow = -conversion_factor * reference_flow
        converted_reference_flow = LinearExpression(
            converted_reference_flow.data.assign(
                const=converted_reference_flow.const.sum("_term")
            ),
            model.lp_model,
        )
        lhs = lp.merge(
            [dependent_flow, converted_reference_flow],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        return lhs == 0
