from zen_garden.constraints.generic_constraint import GenericConstraint


class OpexEmissionsTechnologyTransportConstraint(GenericConstraint):
    def build(self):
        """Calculate opex of each technology.

        .. math::
            O_{j,t,y}^\\mathrm{t} = \\beta_{j,y} F_{j,e,t,y}

        :math:`O_{h,p,t}^\\mathrm{t}`: Variable operating expenditures of transport
        technology :math:`j` on edge :math:`e` at time :math:`t` in year :math:`y` \n
        :math:`\\beta_{j,y}`: Specific variable operating expenditures of transport
        technology :math:`j` in year :math:`y` \n
        :math:`F_{j,e,t,y}`: Reference flow of carrier through transport
        technology :math:`j` on edge :math:`e` at time :math:`t` in year :math:`y`

        """
        techs = self.zen_model.sets["set_transport_technologies"]
        if len(techs) == 0:
            return
        edges = self.zen_model.sets["set_edges"]
        lhs_opex = self.zen_model.variables["cost_opex_variable"].loc[
            techs, edges, :
        ] - (
            self.zen_model.parameters.opex_specific_variable
            * self.zen_model.variables["flow_transport"].rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
        ).sel(
            {"set_technologies": techs, "set_location": edges}
        )
        lhs_emissions = self.zen_model.variables["carbon_emissions_technology"].loc[
            techs, edges, :
        ] - (
            self.zen_model.parameters.carbon_intensity_technology
            * self.zen_model.variables["flow_transport"].rename(
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
        self.zen_model.add_constraint(
            "constraint_opex_technology_transport", constraints_opex
        )
        self.zen_model.add_constraint(
            "constraint_carbon_emissions_technology_transport", constraints_emissions
        )
