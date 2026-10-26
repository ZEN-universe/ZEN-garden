"""Annual imported carrier quantity."""

from zen_garden.elements.carrier.variables.flow_import import FlowImport
from zen_garden.elements.energy_system.parameters.weight_representative_hours import (
    WeightRepresentativeHours,
)
from zen_garden.model.component_types.expression import GenericExpression


class AnnualImportQuantity(GenericExpression):
    """Weight and sum imported flow over representative hours."""

    name = "annual_import_quantity"
    indices = ("set_carriers", "set_nodes", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct weight and sum imported flow over representative hours.

        Summary:
            Weight and sum imported flow over representative hours.

        Formulation:

        .. math::

            \\sum_r w_rF^{imp}_{c,n,y,r}

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
        return (
            optimization_model.variables[FlowImport]
            * optimization_model.parameters[WeightRepresentativeHours]
        ).sum("set_representative_hours")
