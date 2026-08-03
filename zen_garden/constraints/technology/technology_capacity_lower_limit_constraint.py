from zen_garden.constraints.generic_constraint import GenericConstraint


class TechnologyCapacityLowerLimitConstraint(GenericConstraint):
    def build(self):
        """Constraint that installed capacity must be >= the defined lower limit."""

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
