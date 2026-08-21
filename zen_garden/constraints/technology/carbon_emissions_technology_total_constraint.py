from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsTechnologyTotalConstraint(GenericConstraint):
    def build(self):
        r"""Summary:
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
        """
        term_summed_carbon_emissions_technology = (
            self.zen_model.variables["carbon_emissions_technology"]
            * self.get_year_time_step_duration_array()
        ).sum(["set_technologies", "set_location", "set_time_steps_operation"])
        lhs = (
            self.zen_model.variables["carbon_emissions_technology_total"]
            - term_summed_carbon_emissions_technology
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint(
            "constraint_carbon_emissions_technology_total", constraints
        )
