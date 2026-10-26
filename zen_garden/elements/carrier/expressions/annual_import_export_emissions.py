"""Annual emissions associated with carrier imports and exports."""

from zen_garden.elements.carrier.variables.carbon_emissions_carrier import (
    CarbonEmissionsCarrier,
)
from zen_garden.elements.energy_system.parameters.weight_representative_hours import (
    WeightRepresentativeHours,
)
from zen_garden.model.component_types.expression import GenericExpression


class AnnualImportExportEmissions(GenericExpression):
    """Aggregate carrier-trading emissions over carriers, nodes, and hours."""

    name = "annual_import_export_emissions"
    indices = ("set_years",)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Aggregate carrier-trading emissions over carriers, nodes, and hours.

        Formulation:

        .. math::

            \\sum_{c,n,r}w_rE^{carrier}_{c,n,y,r}

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
            optimization_model.variables[CarbonEmissionsCarrier]
            * optimization_model.parameters[WeightRepresentativeHours]
        ).sum(["set_carriers", "set_nodes", "set_representative_hours"])
