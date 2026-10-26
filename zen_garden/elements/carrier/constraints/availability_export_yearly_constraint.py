import numpy as np

from zen_garden.elements.carrier.expressions.annual_export_quantity import (
    AnnualExportQuantity,
)
from zen_garden.elements.carrier.parameters.availability_export_yearly import (
    AvailabilityExportYearly,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class AvailabilityExportYearlyConstraint(GenericConstraint):
    """Limit annual exports to their carrier- and node-specific availability."""

    name = "constraint_availability_export_yearly"
    indices = ("set_carriers", "set_nodes", "set_years")

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct the model component.

        Summary:
            Limit annual exports to their carrier- and node-specific availability.

        Formulation:

        .. math::

            \\sum_rw_rF^{exp}_{c,n,y,r}\\leq a^{exp,annual}_{c,n,y}

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
        available = model.parameters[AvailabilityExportYearly]
        mask = available != np.inf
        return model.expressions[AnnualExportQuantity].where(mask) <= available.where(
            mask
        )
