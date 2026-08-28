from zen_garden.topology.generic_constraint import GenericConstraint


class OpexEmissionsTechnologyTransportConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Calculate opex of each technology.

        Formulation:

        .. math::
            \\begin{aligned}
            C^{\\mathrm{op,var}}_{h,e,t}
            &= \\kappa^{\\mathrm{op,var}}_{h,e,t}F^{\\\\mathrm{trans}}_{h,e,t},\\\\
            M^{\\mathrm{tech}}_{h,e,t}
            &= \\varepsilon^{\\mathrm{op}}_{h,e}F^{\\\\mathrm{trans}}_{h,e,t}.
            \\end{aligned}

        Notation:

        :math:`C^{\\mathrm{op,var}}_{h,e,t}`: variable OPEX of transport
        technology :math:`h` on edge :math:`e` at time :math:`t` in year :math:`y`
        :math:`\\kappa^{\\mathrm{op,var}}_{h,e,t}`: specific variable OPEX of
        transport technology
        :math:`h` in year :math:`y`
        :math:`F^{\\mathrm{trans}}_{h,e,t}`: carrier flow through transport
        technology :math:`h` on edge :math:`e` at time :math:`t` in year :math:`y`
        :math:`M^{\\mathrm{tech}}_{h,e,t}`: carbon emissions from transport
        technology :math:`h` on edge :math:`e` at time :math:`t` in year :math:`y`
        :math:`\\varepsilon^{\\mathrm{op}}_{h,e}`: carbon intensity of transport
        technology
        """
        techs = model_constructor.zen_model.sets["set_transport_technologies"]
        if len(techs) == 0:
            return
        edges = model_constructor.zen_model.sets["set_edges"]
        lhs_opex = model_constructor.zen_model.variables["cost_opex_variable"].loc[
            techs, edges, :
        ] - (
            model_constructor.zen_model.parameters.opex_specific_variable
            * model_constructor.zen_model.variables["flow_transport"].rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
        ).sel(
            {"set_technologies": techs, "set_location": edges}
        )
        lhs_emissions = model_constructor.zen_model.variables[
            "carbon_emissions_technology"
        ].loc[techs, edges, :] - (
            model_constructor.zen_model.parameters.carbon_intensity_technology
            * model_constructor.zen_model.variables["flow_transport"].rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
        ).sel(
            {"set_technologies": techs, "set_location": edges}
        )
        lhs_opex = lhs_opex.rename(
            {
                "set_technologies": "set_transport_technologies",
                "set_location": "set_edges",
            }
        )
        lhs_emissions = lhs_emissions.rename(
            {
                "set_technologies": "set_transport_technologies",
                "set_location": "set_edges",
            }
        )
        rhs = 0
        constraints_opex = lhs_opex == rhs
        constraints_emissions = lhs_emissions == rhs
        ### return
        model_constructor.zen_model.add_constraint(
            "constraint_opex_technology_transport", constraints_opex
        )
        model_constructor.zen_model.add_constraint(
            "constraint_carbon_emissions_technology_transport", constraints_emissions
        )
