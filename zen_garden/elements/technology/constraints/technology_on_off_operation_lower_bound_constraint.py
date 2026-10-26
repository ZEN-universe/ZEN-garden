from zen_garden.elements.technology.expressions.technology_activity import (
    TechnologyActivity,
)
from zen_garden.elements.technology.parameters.min_load import MinLoad
from zen_garden.elements.technology.variables.capacity_on_off_helper_var import (
    CapacityOnOffHelperVar,
)
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.linear_expression import align_and_mask


class TechnologyOnOffOperationLowerBoundConstraint(GenericConstraint):
    """Enforce minimum load while a technology is switched on."""

    name = "constraint_technology_on_off_operation_lower_bound"
    indices = (
        "set_technologies",
        "set_location",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def should_construct(cls, model_constructor):
        technologies = model_constructor.optimization_model.sets.on_off_technologies()
        return bool(technologies)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct enforce minimum load while a technology is switched on.

        Summary:
            Enforce minimum load while a technology is switched on.

        Formulation:

        .. math::

            A\\geq\\underline aH

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
        helper = model.variables[CapacityOnOffHelperVar]
        activity = model.expressions[TechnologyActivity].reindex_like(helper.mask)
        return (
            align_and_mask(model.parameters[MinLoad] * helper - activity, helper.mask)
            <= 0
        )
