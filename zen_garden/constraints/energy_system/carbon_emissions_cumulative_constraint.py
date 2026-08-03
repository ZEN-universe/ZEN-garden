import xarray as xr

from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsCumulativeConstraint(GenericConstraint):
    def build(self):
        """Cumulative carbon emissions over time.

        .. math::
            \\text{First planning period } y = y_0, \\quad E_y^\\mathrm{cum} = E_y
        .. math::
            \\text{Subsequent periods } y > y_0, \\quad E_y^{cum}
            = E_{y-1}^{cum} + (dy-1)E_{y-1}+E_y

        :math:`dy`: interval between planning periods \n
        :math:`E_y`: annual carbon emissions in year :math:`y` \n
        :math:`E_y^{cum}`: cumulative carbon emissions in year :math:`y`

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
