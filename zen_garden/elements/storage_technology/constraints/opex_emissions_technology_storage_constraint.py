from zen_garden.topology.generic_constraint import GenericConstraint


class OpexEmissionsTechnologyStorageConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Calculate variable OPEX and carbon emissions of each storage technology.

        Formulation:

        .. math::
            \\begin{aligned}
            C^{\\mathrm{op,var}}_{h,n,t}
            &= \\kappa^{\\mathrm{op,var}}_{h,n,t}
            (F^{\\mathrm{ch}}_{h,n,t} + F^{\\mathrm{dis}}_{h,n,t}),\\\\
            M^{\\mathrm{tech}}_{h,n,t}
            &= \\varepsilon^{\\mathrm{op}}_{h,n}
            (F^{\\mathrm{ch}}_{h,n,t} + F^{\\mathrm{dis}}_{h,n,t}).
            \\end{aligned}

        Notation:

        :math:`C^{\\mathrm{op,var}}_{h,n,t}`: variable OPEX of storage technology
        :math:`h` at node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`\\kappa^{\\mathrm{op,var}}_{h,n,t}`: specific variable OPEX
        :math:`F^{\\mathrm{ch}}_{h,n,t}`: carrier flow into storage technology
        :math:`h`
        :math:`F^{\\mathrm{dis}}_{h,n,t}`: carrier flow out of storage technology
        :math:`h`
        :math:`M^{\\mathrm{tech}}_{h,n,t}`: operating carbon emissions
        :math:`\\varepsilon^{\\mathrm{op}}_{h,n}`: carbon intensity of the storage
        technology
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
