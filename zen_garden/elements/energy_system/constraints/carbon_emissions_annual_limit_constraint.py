from zen_garden.elements.energy_system.parameters.carbon_emissions_annual_limit import (
    CarbonEmissionsAnnualLimit,
)
from zen_garden.elements.energy_system.variables.carbon_emissions_annual import (
    CarbonEmissionsAnnual,
)
from zen_garden.elements.energy_system.variables.carbon_emissions_annual_overshoot import (  # noqa: E501
    CarbonEmissionsAnnualOvershoot,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsAnnualLimitConstraint(GenericConstraint):
    """Carbon Emissions Annual Limit."""

    name = "constraint_carbon_emissions_annual_limit"
    indices = ("set_years",)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Time dependent carbon emissions limit from technologies and carriers.

        Formulation:

        .. math::
            M_y-M_y^{\\mathrm{ann,over}}\\leq \\overline{m}_y

        Notation:

        :math:`M_y^{\\mathrm{ann,over}}`: permitted overshoot of the annual emissions
        limit

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        lhs = (
            model_constructor.optimization_model.variables[CarbonEmissionsAnnual]
            - model_constructor.optimization_model.variables[
                CarbonEmissionsAnnualOvershoot
            ]
        )
        rhs = model_constructor.optimization_model.parameters[
            CarbonEmissionsAnnualLimit
        ]
        constraints = lhs <= rhs

        return constraints
