from zen_garden.constraints.generic_constraint import GenericConstraint


class CarbonEmissionsTechnologyTotalConstraint(GenericConstraint):
    def build(self):
        """Calculate total carbon emissions of each technology.

        .. math::
            E_y^{\\mathcal{H}} = \\sum_{p\\in\\mathcal{P}}
            \\sum_{t\\in\\mathcal{T}}\\sum_{h\\in\\mathcal{H}} \\theta_{h,p,t} \\tau_{t}

        :math:`E_y^{\\mathcal{H}}`: total carbon emissions of each technology in
        year :math:`y` \n
        :math:`\\theta_{h,p,t}`: carbon emissions of technology :math:`h` at
        location :math:`p` in time step :math:`t` \n
        :math:`\\tau_{t}`: duration of time step :math:`t`

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
