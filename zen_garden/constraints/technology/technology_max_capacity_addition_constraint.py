import numpy as np

from zen_garden.constraints.generic_constraint import GenericConstraint


class TechnologyMaxCapacityAdditionConstraint(GenericConstraint):
    def build(self):
        """Max capacity addition of technology.

        .. math::
            s^\\mathrm{max}_{h} g_{i,p,y} \\ge \\Delta S_{h,p,y}

        :math:`s^\\mathrm{add, max}_{h}`: maximum capacity addition of
        technology :math:`h`  \n
        :math:`g_{i,p,y}`: binary variable which equals 1 if technology is installed
        at location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested
        capacity after construction) at location :math:`p` in year :math:`y`

        """
        capacity_addition_max = self.zen_model.parameters.capacity_addition_max
        mask = (
            (capacity_addition_max != np.inf)
            & (capacity_addition_max != 0)
            & (capacity_addition_max.notnull())
        )

        # if mask is empty, return None
        if not mask.any():
            return None
        lhs = mask * (
            capacity_addition_max * self.zen_model.variables["technology_installation"]
            - self.zen_model.variables["capacity_addition"]
        )
        rhs = 0
        constraints = lhs >= rhs

        self.zen_model.add_constraint(
            "constraint_technology_max_capacity_addition", constraints
        )
