from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsCarrierConstraint(GenericConstraint):
    def build(self):
        """Carbon emissions of importing and exporting carrier.

        .. math::
           \\theta_{c,n,t}^{\\mathrm{carrier}} = \\underline{\\epsilon_c}
           \\underline{U}_{c,n,t} - \\overline{\\epsilon_c} \\overline{U}_{c,n,t}

        :math:`\\theta_{c,n,t}^{\\mathrm{carrier}}`: carbon emissions of importing and
        exporting carrier :math:`c` at node :math:`n` and time step :math:`t`\n
        :math:`\\underline{\\epsilon_c}`: carbon intensity of carrier import :math:`c`\n
        :math:`\\overline{\\epsilon_c}`: carbon intensity of carrier export :math:`c`\n
        :math:`\\underline{U}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` and time step :math:`t`\n
        :math:`\\overline{U}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` and time step :math:`t`

        """
        # create times xarray with 1 where the operation time step is in the year
        times = self.get_year_time_step_array()
        # convert the carbon intensity carrier from yearly to operation time steps
        # TODO map and expand
        carbon_intensity_carrier_import = (
            self.zen_model.parameters.carbon_intensity_carrier_import.broadcast_like(
                times
            )
            * times
        ).sum("set_years")
        carbon_intensity_carrier_export = (
            self.zen_model.parameters.carbon_intensity_carrier_export.broadcast_like(
                times
            )
            * times
        ).sum("set_years")
        lhs = self.zen_model.variables["carbon_emissions_carrier"] - (
            self.zen_model.variables["flow_import"] * carbon_intensity_carrier_import
            - self.zen_model.variables["flow_export"] * carbon_intensity_carrier_export
        )

        rhs = 0

        constraints = lhs == rhs

        self.zen_model.add_constraint(
            "constraint_carbon_emissions_carrier", constraints
        )
