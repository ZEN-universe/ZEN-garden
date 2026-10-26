from zen_garden.elements.energy_system.expressions.annual_carbon_cost import (
    AnnualCarbonCost,
)
from zen_garden.elements.energy_system.variables.cost_carbon_emissions_total import (
    CostCarbonEmissionsTotal,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CostCarbonEmissionsTotalConstraint(GenericConstraint):
    """Cost Carbon Emissions Total."""

    name = "constraint_cost_carbon_emissions_total"
    indices = ("set_years",)

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Carbon cost associated with the carbon emissions of the system in each year.

        Formulation:

        .. math::
            C^{\\mathrm{CO_2}}_y = M_y\\pi^{\\mathrm{CO_2}}_y
            + M_y^{\\mathrm{ann,over}}\\pi^{\\mathrm{CO_2,ann}}
            + \\mathbb{1}_{y=y_\\mathrm{last}}
            M_y^{\\mathrm{bud,over}}\\pi^{\\mathrm{CO_2,bud}}

        Overshoot terms are included only when their respective prices are finite.

        Notation:

        :math:`C^{\\mathrm{CO_2}}_y`: cost of carbon emissions in year :math:`y`
        :math:`M_y`: annual carbon emissions of energy system in year :math:`y`
        :math:`\\pi^{\\mathrm{CO_2}}_y`: carbon price in year :math:`y`
        :math:`M_y^{\\mathrm{ann,over}}`: annual carbon emissions overshoot in year
        :math:`y`
        :math:`\\pi^{\\mathrm{CO_2,ann}}`: carbon price for annual overshoot
        :math:`M_y^{\\mathrm{bud,over}}`: carbon-budget overshoot
        :math:`\\pi^{\\mathrm{CO_2,bud}}`: carbon price for budget overshoot. This cost
        is assigned only to the last modeled year.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        return (
            model.variables[CostCarbonEmissionsTotal]
            == model.expressions[AnnualCarbonCost]
        )
