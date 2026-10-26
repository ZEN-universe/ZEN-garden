"""Annual operating emissions from technologies."""

from zen_garden.elements.energy_system.parameters.weight_representative_hours import (
    WeightRepresentativeHours,
)
from zen_garden.elements.technology.variables.carbon_emissions_technology import (
    CarbonEmissionsTechnology,
)
from zen_garden.model.component_types.expression import GenericExpression


class AnnualTechnologyEmissions(GenericExpression):
    """Aggregate technology emissions over technologies, locations, and hours."""

    name = "annual_technology_emissions"
    indices = ("set_years",)

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Aggregate technology emissions over technologies, locations, and hours.

        Formulation:

        .. math::

            \\sum_{h,p,r}w_rE^{technology}_{h,p,y,r}

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
            optimization_model.variables[CarbonEmissionsTechnology]
            * optimization_model.parameters[WeightRepresentativeHours]
        ).sum(["set_technologies", "set_location", "set_representative_hours"])
