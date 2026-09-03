import numpy as np

from zen_garden.model.component_types.constraint import GenericConstraint


class CostLimitShedDemandConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Cost and limit of shedding demand of carrier.

        Formulation:

        .. math::
           \\begin{aligned}
           C^{\\mathrm{shed}}_{c,n,t}
               &= \\pi^{\\mathrm{shed}}_c F^{\\mathrm{shed}}_{c,n,t},
               && \\pi^{\\mathrm{shed}}_c < \\infty, \\\\
           0 \\leq F^{\\mathrm{shed}}_{c,n,t}
               &\\leq d_{c,n,t},
               && \\pi^{\\mathrm{shed}}_c < \\infty, \\\\
           F^{\\mathrm{shed}}_{c,n,t}
               &= 0,
               && \\pi^{\\mathrm{shed}}_c = \\infty.
           \\end{aligned}

        Notation:

        :math:`C^{\\mathrm{shed}}_{c,n,t}`: total cost of
        shedding demand of carrier :math:`c` at node :math:`n` in time step :math:`t`
        of year :math:`y`
        :math:`\\pi^{\\mathrm{shed}}_c`: price to shed demand of carrier :math:`c`
        :math:`F^{\\mathrm{shed}}_{c,n,t}`: shed demand of carrier :math:`c` at
        node :math:`n` in time step :math:`t` of year :math:`y`
        :math:`d_{c,n,t}`: demand of carrier :math:`c` at node :math:`n` in
        time step :math:`t` of year :math:`y`
        """
        ### mask for finite price, otherwise the shed demand is zero
        mask = (
            model_constructor.optimization_model.parameters.price_shed_demand != np.inf
        )

        # cost of shedding demand
        lhs_cost = (
            model_constructor.optimization_model.variables["cost_shed_demand"]
            - model_constructor.optimization_model.parameters.price_shed_demand
            * model_constructor.optimization_model.variables["shed_demand"]
        ).where(mask)
        rhs_cost = 0
        constraints_cost = lhs_cost == rhs_cost

        # limit of shedding demand:
        #   either the demand (price != inf) or zero (price == inf)
        lhs_shed_demand = model_constructor.optimization_model.variables["shed_demand"]
        rhs_shed_demand = model_constructor.optimization_model.parameters.demand.where(
            mask, 0.0
        )
        constraints_shed_demand = lhs_shed_demand <= rhs_shed_demand

        model_constructor.optimization_model.add_constraint(
            "constraint_cost_shed_demand", constraints_cost
        )
        model_constructor.optimization_model.add_constraint(
            "constraint_limit_shed_demand", constraints_shed_demand
        )
