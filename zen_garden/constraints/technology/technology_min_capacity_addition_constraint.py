from zen_garden.constraints.generic_constraint import GenericConstraint


class TechnologyMinCapacityAdditionConstraint(GenericConstraint):
    def build(self):
        """Min capacity addition of technology.

        .. math::
            \\Delta s^\\mathrm{min}_{h} g_{i,p,y} \\le \\Delta S_{h,p,y}

        :math:`\\Delta s^\\mathrm{min}_{h}`: minimum capacity addition of
        technology :math:`h` \n
        :math:`g_{i,p,y}`: binary variable which equals 1 if technology is installed
        at location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested
        capacity after construction) at location :math:`p` in year :math:`y`

        """
        capacity_addition_min = self.zen_model.parameters.capacity_addition_min
        mask = (capacity_addition_min != 0) & (capacity_addition_min.notnull())

        # if mask is empty, return None
        if not mask.any():
            return None

        lhs = mask * (
            capacity_addition_min * self.zen_model.variables["technology_installation"]
            - self.zen_model.variables["capacity_addition"]
        )
        rhs = 0
        constraints = lhs <= rhs

        ### return
        self.zen_model.add_constraint(
            "constraint_technology_min_capacity_addition", constraints
        )
