from zen_garden.constraints.generic_constraint import GenericConstraint


class OpexEmissionsTechnologyConversionConstraint(GenericConstraint):
    def build(self):
        """Calculate opex and carbon emissions of each technology.

        .. math::
            O_{h,p,t}^\\mathrm{t} = \\beta_{h,p,t} G_{i,n,t}^\\mathrm{r} \n
            \\theta_{h,p,t} = \\epsilon_h G_{i,n,t}^\\mathrm{r}

        :math:`O_{h,p,t}^\\mathrm{t}`: variable opex of the technology :math:`h` at
        node :math:`p` in time step :math:`t` \n
        :math:`\\beta_{h,p,t}`: specific variable opex of the technology :math:`h` at
        node :math:`p` in time step :math:`t` \n
        :math:`G_{i,n,t}^\\mathrm{r}`: reference carrier flow of the
        technology :math:`i` at node :math:`n` in time step :math:`t` \n
        :math:`\\theta^{\\mathrm{tech}}_{h,p,t}`: carbon emissions of operating the
        technology :math:`h` at node :math:`p` in time step :math:`t` \n
        :math:`\\epsilon_h`: carbon intensity of the reference carrier of
        technology :math:`h`


        """
        techs = self.zen_model.sets["set_conversion_technologies"]
        if len(techs) == 0:
            return
        nodes = self.zen_model.sets["set_nodes"]
        term_reference_flow_opex = self.get_flow_expression_conversion(
            techs,
            nodes,
            factor=self.zen_model.parameters.opex_specific_variable.rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            ),
        )
        term_reference_flow_emissions = self.get_flow_expression_conversion(
            techs,
            nodes,
            factor=self.zen_model.parameters.carbon_intensity_technology.rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            ),
        )
        lhs_opex = (
            1 * self.zen_model.variables["cost_opex_variable"].loc[techs, nodes, :]
        ).rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        ) - term_reference_flow_opex
        lhs_emissions = (
            1
            * self.zen_model.variables["carbon_emissions_technology"].loc[
                techs, nodes, :
            ]
        ).rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        ) - term_reference_flow_emissions
        rhs = 0
        constraints_opex = lhs_opex == rhs
        constraints_emissions = lhs_emissions == rhs

        self.zen_model.add_constraint(
            "constraint_opex_technology_conversion", constraints_opex
        )
        self.zen_model.add_constraint(
            "constraint_carbon_emissions_technology_conversion", constraints_emissions
        )
