"""Operational activity of technologies that can be retrofitted."""

from zen_garden.elements.conversion_technology.expressions.conversion_reference_flow import (  # noqa: E501
    ConversionReferenceFlow,
)
from zen_garden.elements.technology.sets.set_retrofitting_technologies import (
    SetRetrofittingTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class RetrofitBaseActivity(GenericExpression):
    """Select reference flow for technologies participating in retrofits."""

    name = "retrofit_base_activity"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = (
        "set_conversion_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )
    dependencies = (ConversionReferenceFlow,)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct select reference flow for technologies participating in retrofits.

        Summary:
            Select reference flow for technologies participating in retrofits.

        Formulation:

        .. math::

            \\{F^{ref}_{b,n,y,r}:b\\in\\mathcal H^{base}\\}

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
        return optimization_model.expressions[ConversionReferenceFlow].sel(
            {
                "set_conversion_technologies": optimization_model.sets[
                    SetRetrofittingTechnologies
                ]
            }
        )
