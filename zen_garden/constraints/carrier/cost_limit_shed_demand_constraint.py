import numpy as np

from zen_garden.constraints.generic_constraint import GenericConstraint


class CostLimitShedDemandConstraint(GenericConstraint):
    def build(self):
        """Cost and limit of shedding demand of carrier.

        .. math::
           O_{c,n,t}^{\\mathrm{shed\\ demand}} = D_{c,n,t} \\nu_c \n
           D_{c,n,t} \\leq d_{c,n,t}

        :math:`O_{c,n,t}^{\\mathrm{shed\\ demand}}`: total cost of shedding
        demand of carrier :math:`c` at node :math:`n` and time step :math:`t`\n
        :math:`\\nu_c`: price to shed demand of carrier :math:`c`\n
        :math:`D_{c,n,t}`: shed demand of carrier :math:`c` at node :math:`n` and
        time step :math:`t`\n
        :math:`d_{c,n,t}`: demand of carrier :math:`c` at node :math:`n` and
        time step :math:`t`


        """
        ### mask for finite price, otherwise the shed demand is zero
        mask = self.zen_model.parameters.price_shed_demand != np.inf

        # cost of shedding demand
        lhs_cost = (
            self.zen_model.variables["cost_shed_demand"]
            - self.zen_model.parameters.price_shed_demand
            * self.zen_model.variables["shed_demand"]
        ).where(mask)
        rhs_cost = 0
        constraints_cost = lhs_cost == rhs_cost

        # limit of shedding demand:
        #   either the demand (price != inf) or zero (price == inf)
        lhs_shed_demand = self.zen_model.variables["shed_demand"]
        rhs_shed_demand = self.zen_model.parameters.demand.where(mask, 0.0)
        constraints_shed_demand = lhs_shed_demand <= rhs_shed_demand

        self.zen_model.add_constraint("constraint_cost_shed_demand", constraints_cost)
        self.zen_model.add_constraint(
            "constraint_limit_shed_demand", constraints_shed_demand
        )
