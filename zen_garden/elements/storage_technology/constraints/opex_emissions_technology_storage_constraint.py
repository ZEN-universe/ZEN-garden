from zen_garden.model.component_types.constraint import GenericConstraint


class OpexEmissionsTechnologyStorageConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
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
        techs = model_constructor.optimization_model.sets["set_storage_technologies"]
        if len(techs) == 0:
            return
        nodes = model_constructor.optimization_model.sets["set_nodes"]
        lhs_opex = model_constructor.optimization_model.variables[
            "cost_opex_variable"
        ].sel({"set_technologies": techs, "set_location": nodes}) - (
            model_constructor.optimization_model.parameters.opex_specific_variable
            * cls.get_flow_expression_storage(model_constructor)
        )
        lhs_emissions = model_constructor.optimization_model.variables[
            "carbon_emissions_technology"
        ].sel({"set_technologies": techs, "set_location": nodes}) - (
            model_constructor.optimization_model.parameters.carbon_intensity_technology
            * cls.get_flow_expression_storage(model_constructor)
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

        model_constructor.optimization_model.add_constraint(
            "constraint_opex_technology_storage", constraints_opex
        )
        model_constructor.optimization_model.add_constraint(
            "constraint_carbon_emissions_technology_storage", constraints_emissions
        )
