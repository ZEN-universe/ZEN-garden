from zen_garden.constraints.generic_constraint import GenericConstraint


class CapacityCapexCouplingConstraint(GenericConstraint):
    def build(self):
        """Couples capacity variables based on modeling technique.

        .. math::
            \\Delta S_{h,p,y} = \\Delta S_{h,p,y}^\\mathrm{approx}

        :math:`\\Delta S_{h,p,y}`: capacity addition of the technology :math:`h` at
        node :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}^\\mathrm{approx}`: approximated capacity addition of
        the technology :math:`h` at node :math:`p` in year :math:`y`

        """
        techs = self.zen_model.sets["set_conversion_technologies"]
        nodes = self.zen_model.sets["set_nodes"]
        capacity_addition = (
            self.zen_model.variables["capacity_addition"]
            .loc[techs, "power", nodes]
            .rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            )
        )
        cost_capex_overnight = (
            self.zen_model.variables["cost_capex_overnight"]
            .loc[techs, "power", nodes]
            .rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            )
        )

        ### formulate constraint
        lhs_capacity = (
            capacity_addition - self.zen_model.variables["capacity_approximation"]
        )
        lhs_capex = (
            cost_capex_overnight - self.zen_model.variables["capex_approximation"]
        )
        rhs = 0
        constraints_capacity = lhs_capacity == rhs
        constraints_capex = lhs_capex == rhs
        ### return
        self.zen_model.add_constraint(
            "constraint_capacity_coupling", constraints_capacity
        )
        self.zen_model.add_constraint("constraint_capex_coupling", constraints_capex)
