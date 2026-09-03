import numpy as np

from zen_garden.model.component_types.constraint import GenericConstraint


class CostCarbonEmissionsTotalConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
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
        """
        mask_last_year = [
            year == model_constructor.model_schema.set_years[-1]
            for year in model_constructor.model_schema.set_years
        ]

        lhs = (
            model_constructor.optimization_model.variables[
                "cost_carbon_emissions_total"
            ]
            - model_constructor.optimization_model.variables["carbon_emissions_annual"]
            * model_constructor.optimization_model.parameters.price_carbon_emissions
        )
        # add cost for overshooting carbon emissions budget
        budget_overshoot = (
            model_constructor.optimization_model.parameters.price_carbon_emissions_budget_overshoot
        )
        if budget_overshoot != np.inf:
            lhs -= (
                model_constructor.optimization_model.variables[
                    "carbon_emissions_budget_overshoot"
                ].where(mask_last_year)
                * budget_overshoot.item()
            )
        # add cost for overshooting annual carbon emissions limit
        annual_overshoot = (
            model_constructor.optimization_model.parameters.price_carbon_emissions_annual_overshoot
        )
        if annual_overshoot != np.inf:
            lhs -= (
                model_constructor.optimization_model.variables[
                    "carbon_emissions_annual_overshoot"
                ]
                * annual_overshoot.item()
            )

        rhs = 0
        constraints = lhs == rhs

        model_constructor.optimization_model.add_constraint(
            "constraint_cost_carbon_emissions_total", constraints
        )
