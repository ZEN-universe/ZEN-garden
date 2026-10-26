from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.elements.technology.variables.capacity_on_off_helper_var import (
    CapacityOnOffHelperVar,
)
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.linear_expression import align_and_mask


class TechnologyOnOffCapacityHelperUpperBoundConstraint(GenericConstraint):
    """Apply the upper capacity bound to the on/off capacity helper."""

    name = "constraint_technology_on_off_capacity_helper_upper_bound"
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
        """Construct apply the upper capacity bound to the on/off capacity helper.

        Summary:
            Apply the upper capacity bound to the on/off capacity helper.

        Formulation:

        .. math::

            H\\leq K

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
        capacity = model.variables[Capacity]
        return align_and_mask(helper - capacity, helper.mask) <= 0
