from zen_garden.elements.carrier.parameters.availability_import import (
    AvailabilityImport,
)
from zen_garden.elements.carrier.variables.flow_import import FlowImport
from zen_garden.model.component_types.constraint import GenericConstraint


class AvailabilityImportConstraint(GenericConstraint):
    """Limit hourly imports to their node- and time-specific availability."""

    name = "constraint_availability_import"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct limit hourly imports to their node- and time-specific availability.

        Summary:
            Limit hourly imports to their node- and time-specific availability.

        Formulation:

        .. math::

            F^{imp}_{c,n,y,r}\\leq a^{imp}_{c,n,y,r}

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
        return model.variables[FlowImport] <= model.parameters[AvailabilityImport]
