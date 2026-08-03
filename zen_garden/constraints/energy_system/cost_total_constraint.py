from zen_garden.constraints.generic_constraint import GenericConstraint


class CostTotalConstraint(GenericConstraint):
    def build(self):
        """Add up all costs from technologies and carriers.

        .. math::
            C_y = CAPEX_y + OPEX_y^\\mathrm{t} + OPEX_y^\\mathrm{c} + OPEX_y^\\mathrm{e}

        :math:`C_y`: total cost of energy system in year :math:`y` \n
        :math:`CAPEX_y`: annual capital expenditures in year :math:`y` \n
        :math:`OPEX_y^\\mathrm{t}`: annual operational expenditures for operating
        technologies in year :math:`y` \n
        :math:`OPEX_y^\\mathrm{c}`: annual operational expenditures for for importing
        and exporting carriers in year :math:`y` \n
        :math:`OPEX_y^\\mathrm{e}`: annual operational expenditures for carbon
        emissions in year :math:`y`

        """
        lhs = (
            self.zen_model.variables["cost_total"]
            - self.zen_model.variables["cost_capex_yearly_total"]
            - self.zen_model.variables["cost_opex_yearly_total"]
            - self.zen_model.variables["cost_carrier_total"]
            - self.zen_model.variables["cost_carbon_emissions_total"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_cost_total", constraints)
