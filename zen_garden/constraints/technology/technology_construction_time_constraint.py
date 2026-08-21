import itertools

import linopy as lp
import numpy as np
import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.constraints.generic_constraint import GenericConstraint


class TechnologyConstructionTimeConstraint(GenericConstraint):
    def build(self):
        r"""Summary:
        Construction time of technology: time between investment and availability.

        Formulation:

        .. math::
            \\text{if start time step is modeled: } \\Delta K_{h,p,y} =
            \\Delta K_{h,p,(y-\\Delta y^{\\mathrm{construction}})}^\\mathrm{invest}
        .. math::
            \\text{elif start time step is before the modeled horizon:}
            \\Delta K_{h,p,y} =
            k^{\\mathrm{ex,inv}}_{h,p,
            (y-\\Delta y^{\\mathrm{construction}})}
        .. math::
            \\text{else: } \\Delta K_{h,p,y} = 0

        Investments whose completion would occur after the modeled horizon are
        fixed to zero:

        .. math::
            \\Delta K^{\\mathrm{inv}}_{h,p,y}=0
            \\quad\\text{if }y+\\Delta y^\\mathrm{construction}
            \\notin\\mathcal{Y}.

        For storage technologies, each equation is applied independently to power
        and energy capacity.

        Notation:

        :math:`\\Delta K_{h,p,y}`: size of built technology :math:`h` (invested
        capacity after construction) at location :math:`p` in year :math:`y`
        :math:`\\Delta K_{h,p,y}^\\mathrm{invest}`: size of invested technology at
        location :math:`p` in year :math:`y`
        :math:`k^{\\mathrm{ex,inv}}_{h,p,y}`: size of the previously invested
        capacities at location :math:`p` in year :math:`y`
        :math:`\\Delta y^\\mathrm{construction}`: construction time rounded up to an
        integer number of planning intervals
        """
        # get investment time step
        investment_time = pd.Series(
            {
                (
                    t,
                    y,
                    self._get_investment_time_step(t, y),
                ): 1
                for t, y in itertools.product(
                    self.zen_model.sets["set_technologies"],
                    self.zen_model.sets["set_years"],
                )
            }
        )
        investment_time.index.names = [
            "set_technologies",
            "set_years",
            "set_time_steps_construction",
        ]

        # select masks
        mask_current_time_steps = investment_time.index.get_level_values(
            "set_time_steps_construction"
        ).isin(self.zen_model.sets["set_years"])
        mask_existing_time_steps = (
            investment_time.isin(self.zen_model.sets["set_years_entire_horizon"])
            & ~mask_current_time_steps
        )
        # broadcast capacity investment and capacity investment existing
        capacity_investment = self.zen_model.variables["capacity_investment"]
        investment_time_current = (
            investment_time[mask_current_time_steps]
            .dropna()
            .to_xarray()
            .broadcast_like(capacity_investment.mask)
            .fillna(0)
        )
        investment_time_existing = (
            investment_time[mask_existing_time_steps]
            .dropna()
            .to_xarray()
            .broadcast_like(capacity_investment.mask)
            .fillna(0)
        )
        # gets the time steps where no investment can be made without the
        #   addition exceeding the horizon
        investment_time_outside = (1 - investment_time_current).min("set_years")

        capacity_investment = capacity_investment.rename(
            {"set_years": "set_time_steps_construction"}
        )
        capacity_investment_addition = capacity_investment.broadcast_like(
            investment_time_current
        )
        capacity_investment_existing = (
            self.zen_model.parameters.capacity_investment_existing
        )
        capacity_investment_existing = capacity_investment_existing.rename(
            {"set_years_entire_horizon": "set_time_steps_construction"}
        ).broadcast_like(investment_time_existing)

        ### formulate constraint
        lhs = lp.merge(
            [
                1 * self.zen_model.variables["capacity_addition"],
                -(investment_time_current * capacity_investment_addition).sum(
                    "set_time_steps_construction"
                ),
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = (investment_time_existing * capacity_investment_existing).sum(
            "set_time_steps_construction"
        )
        rhs = xr.align(lhs.const, rhs, join="left")[1]
        constraints = lhs == rhs
        # constrain capacity_investment where no investment can be made
        #   without the addition exceeding the horizon
        lhs_outside = self.align_and_mask(capacity_investment, investment_time_outside)
        rhs_outside = 0
        constraints_outside = lhs_outside == rhs_outside

        self.zen_model.add_constraint(
            "constraint_technology_construction_time", constraints
        )
        self.zen_model.add_constraint(
            "constraint_technology_construction_time_outside", constraints_outside
        )

    def _get_investment_time_step(self, tech, year):
        """Returns investment time step of technology, considering construction time.

        returns investment time step of technology, i.e., the time step in which the
        technology is invested considering the construction time.

        :param params: parameters of the model
        :param tech: name of technology
        :param year: yearly time step
        :return: investment time step
        """
        # get params and system
        construction_time = self.zen_model.parameters.dict_parameters.construction_time[
            tech
        ]
        # conservative estimate of construction time (ceil)
        del_construction_time = int(
            np.ceil(construction_time / self.config.system.interval_between_years)
        )
        return year - del_construction_time
