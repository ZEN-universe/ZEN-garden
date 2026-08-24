from zen_garden.topology.generic_constraint import GenericConstraint


class TechnologyCapacityLowerLimitConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Constrain installed capacity by each positive lower limit.

        Formulation:

        .. math::
            K_{h,p,y} \\geq \\underline{k}_{h,p,y}

        No constraint is created where the configured lower limit is zero.
        For storage technologies, the equation is applied independently to power
        and energy capacity.

        Notation:

        :math:`K_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y`
        :math:`\\underline{k}_{h,p,y}`: lower capacity limit
        """

        # In TechnologyRules, we access variables and parameters directly via self
        capacity = self.zen_model.variables["capacity"]
        capacity_lower_limit = self.zen_model.parameters.capacity_lower_limit

        # Create a mask so we only build constraints
        # where the user actually provided a number
        mask = capacity_lower_limit > 0.0

        # Apply the mask using xarray's .where() so we don't build empty/NaN constraints
        lhs = capacity.where(mask)
        rhs = capacity_lower_limit.where(mask, 0.0)

        # Total Capacity >= Lower Bound
        constraint = lhs >= rhs

        # Add the constraint to the model
        self.zen_model.add_constraint(
            "constraint_technology_capacity_lower_limit", constraint
        )
