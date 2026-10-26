import numpy as np

from zen_garden.elements.carrier.expressions.annual_import_quantity import (
    AnnualImportQuantity,
)
from zen_garden.elements.carrier.parameters.availability_import_yearly import (
    AvailabilityImportYearly,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class AvailabilityImportYearlyConstraint(GenericConstraint):
    """Limit annual imports to their carrier- and node-specific availability."""

    name = "constraint_availability_import_yearly"
    indices = ("set_carriers", "set_nodes", "set_years")

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct the model component.

        Summary:
            Limit annual imports to their carrier- and node-specific availability.

        Formulation:

        .. math::

            \\sum_rw_rF^{imp}_{c,n,y,r}\\leq a^{imp,annual}_{c,n,y}

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
        available = model.parameters[AvailabilityImportYearly]
        mask = available != np.inf
        return model.expressions[AnnualImportQuantity].where(mask) <= available.where(
            mask
        )
