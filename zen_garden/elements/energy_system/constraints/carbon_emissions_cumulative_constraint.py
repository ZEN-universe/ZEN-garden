import xarray as xr

from zen_garden.topology.generic_constraint import GenericConstraint


class CarbonEmissionsCumulativeConstraint(GenericConstraint):
    def build(self):
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
            True if year == self.energy_system.set_years[0] else False
            for year in self.energy_system.set_years
        ]

        lhs = (
            self.zen_model.variables["carbon_emissions_cumulative"]
            - self.zen_model.variables["carbon_emissions_cumulative"].shift(set_years=1)
            - self.zen_model.variables["carbon_emissions_annual"].shift(set_years=1)
            * (self.config.system.interval_between_years - 1)
            - self.zen_model.variables["carbon_emissions_annual"]
        )
        rhs = (
            xr.ones_like(self.zen_model.variables["carbon_emissions_cumulative"].mask)
            * self.zen_model.parameters.carbon_emissions_cumulative_existing
        ).where(m, 0)
        constraints = lhs == rhs

        self.zen_model.add_constraint(
            "constraint_carbon_emissions_cumulative", constraints
        )
