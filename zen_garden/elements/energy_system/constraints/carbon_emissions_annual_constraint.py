from zen_garden.elements.carrier.variables.carbon_emissions_carrier_total import (
    CarbonEmissionsCarrierTotal,
)
from zen_garden.elements.energy_system.variables.carbon_emissions_annual import (
    CarbonEmissionsAnnual,
)
from zen_garden.elements.technology.variables.carbon_emissions_technology_total import (
    CarbonEmissionsTechnologyTotal,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsAnnualConstraint(GenericConstraint):
    """Carbon Emissions Annual."""

    name = "constraint_carbon_emissions_annual"
    indices = ("set_years",)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Add up all carbon emissions from technologies and carriers.

        Formulation:

        .. math::
            M_y = M_y^\\mathrm{tech} + M_y^\\mathrm{carrier}

        Notation:

        :math:`M_y^\\mathrm{tech}`: carbon emissions from technologies in
        year :math:`y`
        :math:`M_y^\\mathrm{carrier}`: carbon emissions from carriers in year
        :math:`y`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        lhs = (
            model_constructor.optimization_model.variables[CarbonEmissionsAnnual]
            - model_constructor.optimization_model.variables[
                CarbonEmissionsTechnologyTotal
            ]
            - model_constructor.optimization_model.variables[
                CarbonEmissionsCarrierTotal
            ]
        )
        rhs = 0
        constraints = lhs == rhs

        return constraints
