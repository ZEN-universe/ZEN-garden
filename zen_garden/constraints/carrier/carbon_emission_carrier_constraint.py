from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsCarrierConstraint(GenericConstraint):
    def build(self):
        r"""Summary:
        Carbon emissions of importing and exporting carrier.

        Formulation:

        .. math::
           M^{\\mathrm{carrier}}_{c,n,t} = \\varepsilon^{\\mathrm{imp}}_{c,y}
           F^{\\mathrm{imp}}_{c,n,t} - \\varepsilon^{\\mathrm{exp}}_{c,y}
           F^{\\mathrm{exp}}_{c,n,t}

        Notation:

        :math:`M^{\\mathrm{carrier}}_{c,n,t}`: carbon emissions of importing
        and exporting carrier :math:`c` at node :math:`n` in time step :math:`t`
        of year :math:`y`
        :math:`\\varepsilon^{\\mathrm{imp}}_{c,y}`: carbon intensity of carrier import
        :math:`c` in year :math:`y`
        :math:`\\varepsilon^{\\mathrm{exp}}_{c,y}`: carbon intensity of carrier export
        :math:`c` in year :math:`y`
        :math:`F^{\\mathrm{imp}}_{c,n,t}`: flow of carrier :math:`c` imported at
        node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`F^{\\mathrm{exp}}_{c,n,t}`: flow of carrier :math:`c` exported at
        node :math:`n` in time step :math:`t` of year :math:`y`
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
