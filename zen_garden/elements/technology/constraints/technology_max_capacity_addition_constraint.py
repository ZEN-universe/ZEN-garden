import numpy as np

from zen_garden.topology.generic_constraint import GenericConstraint


class TechnologyMaxCapacityAdditionConstraint(GenericConstraint):
    def build(self):
        """Summary:
        Max capacity addition of technology.

        Formulation:

        .. math::
            \\overline{\\Delta k}_{h} g_{h,p,y}
            \\geq \\Delta K_{h,p,y}

        The constraint is omitted for zero, missing, or infinite maximum additions.

        Notation:

        :math:`\\overline{\\Delta k}_{h}`: maximum capacity addition for technology
        :math:`h` (with capacity type implicit)
        :math:`g_{h,p,y}`: binary variable which equals 1 if technology is installed
        at location :math:`p` in year :math:`y`
        :math:`\\Delta K_{h,p,y}`: capacity addition
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
