from zen_garden.elements.technology.expressions.annual_technology_emissions import (
    AnnualTechnologyEmissions,
)
from zen_garden.elements.technology.variables.carbon_emissions_technology_total import (
    CarbonEmissionsTechnologyTotal,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsTechnologyTotalConstraint(GenericConstraint):
    """Carbon Emissions Technology Total."""

    name = "constraint_carbon_emissions_technology_total"
    indices = ("set_years",)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Calculate total carbon emissions of each technology.

        Formulation:

        .. math::
            M_y^{\\mathrm{tech}} = \\sum_{p\\in\\mathcal{P}}
            \\sum_{t\\in\\mathcal{T}_y}\\sum_{h\\in\\mathcal{H}}
            M^{\\mathrm{tech}}_{h,p,t} \\Delta t_t

        Notation:

        :math:`M_y^{\\mathrm{tech}}`: total carbon emissions of all technologies in
        year :math:`y`
        :math:`M^{\\mathrm{tech}}_{h,p,t}`: carbon emissions of technology
        :math:`h` at location :math:`p` in time step :math:`t` of year :math:`y`
        :math:`\\Delta t_t`: duration of time step :math:`t`

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        return (
            model.variables[CarbonEmissionsTechnologyTotal]
            == model.expressions[AnnualTechnologyEmissions]
        )
