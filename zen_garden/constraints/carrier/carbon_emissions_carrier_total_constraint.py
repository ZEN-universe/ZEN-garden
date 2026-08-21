from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsCarrierTotalConstraint(GenericConstraint):
    def build(self):
        r"""Summary:
        Total carbon emissions of importing and exporting carrier.

        Formulation:

        .. math::
            M_y^{\\mathrm{carrier}} = \\sum_{c\\in\\mathcal{C}}\\sum_{n\\in\\mathcal{N}}
            \\sum_{t\\in\\mathcal{T}_y} \\Delta t_t
            M^{\\mathrm{carrier}}_{c,n,t}

        Notation:

        :math:`M^{\\mathrm{carrier}}_{c,n,t}`: carbon emissions of importing
        and exporting carrier :math:`c` at node :math:`n` in time step :math:`t`
        of year :math:`y`
        :math:`\\Delta t_t`: duration of time step :math:`t`
        """
        term_summed_carbon_emissions_carrier = (
            self.zen_model.variables["carbon_emissions_carrier"]
            * self.get_year_time_step_duration_array()
        ).sum(["set_carriers", "set_nodes", "set_time_steps_operation"])
        lhs = (
            self.zen_model.variables["carbon_emissions_carrier_total"]
            - term_summed_carbon_emissions_carrier
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint(
            "constraint_carbon_emissions_carrier_total", constraints
        )
