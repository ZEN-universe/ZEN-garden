import numpy as np

from zen_garden.constraints.generic_constraint import GenericConstraint


class TechnologyCapacityLimitConstraint(GenericConstraint):
    def build(self):
        """Limited capacity_limit of technology.

        .. math::
            \\text{if existing capacities < capacity limit: }
            s^\\mathrm{max}_{h,p,y} \\geq S_{h,p,y}
        .. math::
            \\text{else: } \\Delta S_{h,p,y} = 0

        :math:`S_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`s^\\mathrm{max}_{h,p,y}`: capacity limit of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested
        capacity after construction) at location :math:`p` in year :math:`y`

        """
        # if the capacity limit is not reached by the existing capacities,
        # the capacity is constrained by the capacity limit.
        # if the capacity limit is reached, the capacity addition is 0.
        capacity_limit_not_reached = (
            self.zen_model.parameters.existing_capacities
            < self.zen_model.parameters.capacity_limit
        )
        # create mask so that skipped if capacity_limit is inf
        m = self.zen_model.parameters.capacity_limit != np.inf

        lhs_not_reached = (
            self.zen_model.variables["capacity"]
            .where(m)
            .where(capacity_limit_not_reached)
        )
        rhs_not_reached = self.zen_model.parameters.capacity_limit.where(m, 0.0).where(
            capacity_limit_not_reached, 0.0
        )
        constraints_not_reached = lhs_not_reached <= rhs_not_reached
        lhs_reached = (
            self.zen_model.variables["capacity_addition"]
            .where(m)
            .where(~capacity_limit_not_reached)
        )
        rhs_reached = 0
        if not self.config.system.allow_investment:
            lhs_reached = self.zen_model.variables["capacity_addition"]
        constraints_reached = lhs_reached == rhs_reached

        self.zen_model.add_constraint(
            "constraint_technology_capacity_limit_not_reached", constraints_not_reached
        )
        self.zen_model.add_constraint(
            "constraint_technology_capacity_limit_reached", constraints_reached
        )
