import pandas as pd

from zen_garden.constraints.generic_constraint import GenericConstraint


class NetPresentCostConstraint(GenericConstraint):
    def build(self):
        """Discounts the annual capital flows to calculate the net_present_cost.

        .. math::
            NPC_y = \\sum_{i \\in [0,dy(y))-1]}
            \\left( \\dfrac{1}{1+r} \\right)^{\\left(dy (y-y_0) + i \\right)} C_y

        :math:`NPC_y`: net present cost of energy system in year :math:`y` \n
        :math:`C_y`: total cost of energy system in year :math:`y` \n
        :math:`r`: discount rate \n
        :math:`dy`: interval between planning periods \n

        """
        factor = pd.Series(index=self.energy_system.set_years)
        for year in self.energy_system.set_years:
            ### auxiliary calculations
            if year == self.energy_system.set_years_entire_horizon[-1]:
                interval_between_years = 1
            else:
                interval_between_years = self.config.system.interval_between_years
            # economic discount
            factor[year] = sum(
                (
                    (1 / (1 + self.zen_model.parameters.discount_rate))
                    ** (
                        self.config.system.interval_between_years
                        * (year - self.energy_system.set_years[0])
                        + _intermediate_time_step
                    )
                )
                for _intermediate_time_step in range(0, interval_between_years)
            )
        term_discounted_cost_total = self.zen_model.variables["cost_total"] * factor

        lhs = self.zen_model.variables["net_present_cost"] - term_discounted_cost_total
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_net_present_cost", constraints)
