from zen_garden.topology.generic_constraint import GenericConstraint


class CarbonEmissionsTechnologyTotalConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
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
            model_constructor.zen_model.variables["carbon_emissions_technology"]
            * cls.get_year_time_step_duration_array(model_constructor)
        ).sum(["set_technologies", "set_location", "set_time_steps_operation"])
        lhs = (
            model_constructor.zen_model.variables["carbon_emissions_technology_total"]
            - term_summed_carbon_emissions_technology
        )
        rhs = 0
        constraints = lhs == rhs

        model_constructor.zen_model.add_constraint(
            "constraint_carbon_emissions_technology_total", constraints
        )
