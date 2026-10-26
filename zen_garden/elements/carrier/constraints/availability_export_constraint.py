from zen_garden.elements.carrier.parameters.availability_export import (
    AvailabilityExport,
)
from zen_garden.elements.carrier.variables.flow_export import FlowExport
from zen_garden.model.component_types.constraint import GenericConstraint


class AvailabilityExportConstraint(GenericConstraint):
    """Limit hourly exports to their node- and time-specific availability."""

    name = "constraint_availability_export"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct limit hourly exports to their node- and time-specific availability.

        Summary:
            Limit hourly exports to their node- and time-specific availability.

        Formulation:

        .. math::

            F^{exp}_{c,n,y,r}\\leq a^{exp}_{c,n,y,r}

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
        return model.variables[FlowExport] <= model.parameters[AvailabilityExport]
