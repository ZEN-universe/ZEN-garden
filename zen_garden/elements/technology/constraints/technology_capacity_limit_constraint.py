import numpy as np

from zen_garden.topology.generic_constraint import GenericConstraint


class TechnologyCapacityLimitConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Limited capacity_limit of technology.

        Formulation:

        .. math::
            \\text{if existing capacities < capacity limit: }
            \\overline{k}_{h,p,y} \\geq K_{h,p,y}
        .. math::
            \\text{else: } \\Delta K_{h,p,y} = 0

        Infinite capacity limits are skipped. If investment is disabled globally,
        all capacity additions are fixed to zero irrespective of the limit.

        For storage technologies, the equation is applied independently to power
        capacity and energy capacity :math:`K^{\\mathrm{energy}}_{h,n,y}`.

        Notation:

        :math:`K_{h,p,y}`: installed capacity
        :math:`\\overline{k}_{h,p,y}`: capacity limit
        :math:`\\Delta K_{h,p,y}`: capacity addition
        """
        # if the capacity limit is not reached by the existing capacities,
        # the capacity is constrained by the capacity limit.
        # if the capacity limit is reached, the capacity addition is 0.
        capacity_limit_not_reached = (
            model_constructor.zen_model.parameters.existing_capacities
            < model_constructor.zen_model.parameters.capacity_limit
        )
        # create mask so that skipped if capacity_limit is inf
        m = model_constructor.zen_model.parameters.capacity_limit != np.inf

        lhs_not_reached = (
            model_constructor.zen_model.variables["capacity"]
            .where(m)
            .where(capacity_limit_not_reached)
        )
        rhs_not_reached = model_constructor.zen_model.parameters.capacity_limit.where(
            m, 0.0
        ).where(capacity_limit_not_reached, 0.0)
        constraints_not_reached = lhs_not_reached <= rhs_not_reached
        lhs_reached = (
            model_constructor.zen_model.variables["capacity_addition"]
            .where(m)
            .where(~capacity_limit_not_reached)
        )
        rhs_reached = 0
        if not model_constructor.config.system.allow_investment:
            lhs_reached = model_constructor.zen_model.variables["capacity_addition"]
        constraints_reached = lhs_reached == rhs_reached

        model_constructor.zen_model.add_constraint(
            "constraint_technology_capacity_limit_not_reached", constraints_not_reached
        )
        model_constructor.zen_model.add_constraint(
            "constraint_technology_capacity_limit_reached", constraints_reached
        )
