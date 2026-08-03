from zen_garden.constraints.generic_constraint import GenericConstraint


class OpexEmissionsTechnologyStorageConstraint(GenericConstraint):
    def build(self):
        """Calculate opex of each technology.

        .. math::
            O_{h,p,t}^\\mathrm{t} = \\beta_{h,p,t} (\\underline{H}_{k,n,t} +
            \\overline{H}_{k,n,t}) \n
            \\theta_{h,p,t}^{\\mathrm{tech}} = \\epsilon_h (\\underline{H}_{k,n,t} +
            \\overline{H}_{k,n,t})

        :math:`O_{h,p,t}^\\mathrm{t}`: variable operational expenditures for storage
        technology :math:`h` on node :math:`n` and time :math:`t` \n
        :math:`\\beta_{h,p,t}`: specific variable operational expenditures for storage
        technology :math:`h` on node :math:`n` and time :math:`t` \n
        :math:`\\underline{H}_{k,n,t}`: carrier flow into storage technology :math:`k`
        on node :math:`n` and time :math:`t` \n
        :math:`\\overline{H}_{k,n,t}`: carrier flow out of storage technology :math:`k`
        on node :math:`n` and time :math:`t` \n
        :math:`\\theta_{h,p,t}^{\\mathrm{tech}}`: carbon emissions for storage
        technology :math:`h` on node :math:`n` and time :math:`t` \n
        :math:`\\epsilon_h`: carbon intensity for operating storage technology :math:`h`
        on node :math:`n`

        """
        techs = self.zen_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        nodes = self.zen_model.sets["set_nodes"]
        lhs_opex = self.zen_model.variables["cost_opex_variable"].sel(
            {"set_technologies": techs, "set_location": nodes}
        ) - (
            self.zen_model.parameters.opex_specific_variable
            * self.get_flow_expression_storage()
        )
        lhs_emissions = self.zen_model.variables["carbon_emissions_technology"].sel(
            {"set_technologies": techs, "set_location": nodes}
        ) - (
            self.zen_model.parameters.carbon_intensity_technology
            * self.get_flow_expression_storage()
        )
        lhs_opex = lhs_opex.rename(
            {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }
        )
        lhs_emissions = lhs_emissions.rename(
            {
                "set_technologies": "set_storage_technologies",
                "set_location": "set_nodes",
            }
        )
        rhs = 0
        constraints_opex = lhs_opex == rhs
        constraints_emissions = lhs_emissions == rhs

        self.zen_model.add_constraint(
            "constraint_opex_technology_storage", constraints_opex
        )
        self.zen_model.add_constraint(
            "constraint_carbon_emissions_technology_storage", constraints_emissions
        )
