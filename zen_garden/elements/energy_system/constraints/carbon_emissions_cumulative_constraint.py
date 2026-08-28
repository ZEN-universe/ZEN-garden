import xarray as xr

from zen_garden.model.component_types.constraint import GenericConstraint


class CarbonEmissionsCumulativeConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Cumulative carbon emissions over time.

        Formulation:

        .. math::
            \\text{First planning period } y = y_0, \\quad
            M_y^{\\mathrm{cum}} = m_0^{\\mathrm{cum}}+M_y
        .. math::
            \\text{Subsequent periods } y > y_0, \\quad M_y^{\\mathrm{cum}}
            = M_{y-1}^{\\mathrm{cum}} + (\\Delta y-1)M_{y-1}+M_y

        Notation:

        :math:`\\Delta y`: interval between planning periods
        :math:`M_y`: annual carbon emissions in year :math:`y`
        :math:`M_y^{\\mathrm{cum}}`: cumulative carbon emissions in year :math:`y`
        :math:`m_0^{\\mathrm{cum}}`: cumulative emissions before the modeled horizon
        """
        m = [
            True if year == model_constructor.model_schema.set_years[0] else False
            for year in model_constructor.model_schema.set_years
        ]

        lhs = (
            model_constructor.optimization_model.variables[
                "carbon_emissions_cumulative"
            ]
            - model_constructor.optimization_model.variables[
                "carbon_emissions_cumulative"
            ].shift(set_years=1)
            - model_constructor.optimization_model.variables[
                "carbon_emissions_annual"
            ].shift(set_years=1)
            * (model_constructor.config.system.interval_between_years - 1)
            - model_constructor.optimization_model.variables["carbon_emissions_annual"]
        )
        cumulative_existing = (
            model_constructor.optimization_model.parameters.carbon_emissions_cumulative_existing
        )
        rhs = (
            xr.ones_like(
                model_constructor.optimization_model.variables[
                    "carbon_emissions_cumulative"
                ].mask
            )
            * cumulative_existing
        ).where(m, 0)
        constraints = lhs == rhs

        model_constructor.optimization_model.add_constraint(
            "constraint_carbon_emissions_cumulative", constraints
        )
