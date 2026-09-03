from zen_garden.model.component_types.constraint import GenericConstraint


class OpexEmissionsTechnologyConversionConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Calculate opex and carbon emissions of each technology.

        Formulation:

        .. math::
            \\begin{aligned}
            C^{\\mathrm{op,var}}_{h,n,t}
            &= \\kappa^{\\mathrm{op,var}}_{h,n,t}F^{\\mathrm{ref}}_{h,n,t},\\\\
            M^{\\mathrm{tech}}_{h,n,t}
            &= \\varepsilon^{\\mathrm{op}}_{h,n}F^{\\mathrm{ref}}_{h,n,t}.
            \\end{aligned}

        Notation:

        :math:`C^{\\mathrm{op,var}}_{h,n,t}`: variable OPEX of conversion technology
        :math:`h` at node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`\\kappa^{\\mathrm{op,var}}_{h,n,t}`: specific variable OPEX
        :math:`F^{\\mathrm{ref}}_{h,n,t}`: reference carrier flow of the
        technology :math:`h` at node :math:`n` in time step :math:`t` of year
        :math:`y`
        :math:`M^{\\mathrm{tech}}_{h,n,t}`: operating carbon emissions
        :math:`\\varepsilon^{\\mathrm{op}}_{h,n}`: carbon intensity of the
        conversion technology
        """
        techs = model_constructor.optimization_model.sets["set_conversion_technologies"]
        if len(techs) == 0:
            return
        nodes = model_constructor.optimization_model.sets["set_nodes"]
        term_reference_flow_opex = cls.get_flow_expression_conversion(
            model_constructor,
            techs,
            nodes,
            factor=model_constructor.optimization_model.parameters.opex_specific_variable.rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            ),
        )
        term_reference_flow_emissions = cls.get_flow_expression_conversion(
            model_constructor,
            techs,
            nodes,
            factor=model_constructor.optimization_model.parameters.carbon_intensity_technology.rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            ),
        )
        lhs_opex = (
            1
            * model_constructor.optimization_model.variables["cost_opex_variable"].loc[
                techs, nodes, :
            ]
        ).rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        ) - term_reference_flow_opex
        lhs_emissions = (
            1
            * model_constructor.optimization_model.variables[
                "carbon_emissions_technology"
            ].loc[techs, nodes, :]
        ).rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        ) - term_reference_flow_emissions
        rhs = 0
        constraints_opex = lhs_opex == rhs
        constraints_emissions = lhs_emissions == rhs

        model_constructor.optimization_model.add_constraint(
            "constraint_opex_technology_conversion", constraints_opex
        )
        model_constructor.optimization_model.add_constraint(
            "constraint_carbon_emissions_technology_conversion", constraints_emissions
        )
