"""Rules for the Technology class."""

import itertools
import logging
from typing import TYPE_CHECKING, cast

import linopy as lp
import numpy as np
import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.element import Element
from zen_garden.elements.generic_rule import GenericRule
from zen_garden.elements.technology import Technology
from zen_garden.model.components.zen_index import ZenIndex
from zen_garden.model.components.zen_set import ZenSet

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.model.config import Config
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.model.zen_model import ZenModel
    from zen_garden.services.element_registry import ElementRegistry

logger = logging.getLogger(__name__)


class TechnologyRules(GenericRule):
    """Rules for the Technology class."""

    def __init__(
        self,
        config: "Config",
        zen_model: "ZenModel",
        energy_system: "EnergySystem",
        time_steps: "TimeStepsDicts",
        element_registry: "ElementRegistry",
    ):
        """Inits the rules.

        :param config: Config object
        :param zen_model: ZenModel object
        :param energy_system: EnergySystem object
        :param element_registry: ElementRegistry object
        """
        self.element_registry = element_registry
        super().__init__(config, zen_model, energy_system, time_steps)

    def constraint_cost_capex_yearly_total(self):
        """Sums over all technologies to calculate total capex.

        .. math::
            CAPEX_y = \\sum_{h\\in\\mathcal{H}}\\sum_{p\\in\\mathcal{P}}A_{h,p,y} +
            \\sum_{k\\in\\mathcal{K}}\\sum_{n\\in\\mathcal{N}}A^\\mathrm{e}_{k,n,y}

        :math:`A_{h,p,y}`: annual capex of technology :math:`h` at location :math:`p`
        in year :math:`y`

        """
        lhs = self.zen_model.lp_model.variables[
            "cost_capex_yearly_total"
        ] - self.zen_model.lp_model.variables["cost_capex_yearly"].sum(
            ["set_technologies", "set_capacity_types", "set_location"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_cost_capex_yearly_total", constraints)

    def constraint_cost_opex_yearly_total(self):
        """Sums over all technologies to calculate total opex.

        .. math::
            OPEX_y = \\sum_{h\\in\\mathcal{H}}\\sum_{p\\in\\mathcal{P}} OPEX_{h,p,y}

        :math:`OPEX_{h,p,y}`: opex of operating technology :math:`h` at
        location :math:`p` in year :math:`y`

        """
        lhs = self.zen_model.lp_model.variables[
            "cost_opex_yearly_total"
        ] - self.zen_model.lp_model.variables["cost_opex_yearly"].sum(
            ["set_technologies", "set_location"]
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint("constraint_cost_opex_yearly_total", constraints)

    def constraint_technology_capacity_limit(self):
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
            self.zen_model.lp_model.variables["capacity"]
            .where(m)
            .where(capacity_limit_not_reached)
        )
        rhs_not_reached = self.zen_model.parameters.capacity_limit.where(m, 0.0).where(
            capacity_limit_not_reached, 0.0
        )
        constraints_not_reached = lhs_not_reached <= rhs_not_reached
        lhs_reached = (
            self.zen_model.lp_model.variables["capacity_addition"]
            .where(m)
            .where(~capacity_limit_not_reached)
        )
        rhs_reached = 0
        if not self.config.system.allow_investment:
            lhs_reached = self.zen_model.lp_model.variables["capacity_addition"]
        constraints_reached = lhs_reached == rhs_reached

        self.zen_model.add_constraint(
            "constraint_technology_capacity_limit_not_reached", constraints_not_reached
        )
        self.zen_model.add_constraint(
            "constraint_technology_capacity_limit_reached", constraints_reached
        )

    def constraint_technology_capacity_lower_limit(self):
        """Constraint that installed capacity must be >= the defined lower limit."""

        # In TechnologyRules, we access variables and parameters directly via self
        capacity = self.zen_model.lp_model.variables["capacity"]
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

    def constraint_technology_min_capacity_addition(self):
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
            capacity_addition_min
            * self.zen_model.lp_model.variables["technology_installation"]
            - self.zen_model.lp_model.variables["capacity_addition"]
        )
        rhs = 0
        constraints = lhs <= rhs

        ### return
        self.zen_model.add_constraint(
            "constraint_technology_min_capacity_addition", constraints
        )

    def constraint_technology_max_capacity_addition(self):
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
            capacity_addition_max
            * self.zen_model.lp_model.variables["technology_installation"]
            - self.zen_model.lp_model.variables["capacity_addition"]
        )
        rhs = 0
        constraints = lhs >= rhs

        self.zen_model.add_constraint(
            "constraint_technology_max_capacity_addition", constraints
        )

    def constraint_technology_construction_time(self):
        """Construction time of technology: time between investment and availability.

        .. math::
            \\text{if start time step in set time steps yearly: } \\Delta S_{h,p,y} =
            \\Delta S_{h,p,(y-dy^{\\mathrm{construction}})}^\\mathrm{invest}
        .. math::
            \\text{elif start time step in set time steps yearly entire horizon:}
            \\Delta S_{h,p,y} =
            \\Delta s^\\mathrm{ex,invest}_{h,p,(y-dy^{\\mathrm{construction}})}
        .. math::
            \\text{else: } \\Delta S_{h,p,y} = 0

        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested
        capacity after construction) at location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}^\\mathrm{invest}`: size of invested technology at
        location :math:`p` in year :math:`y` \n
        :math:`\\Delta s^\\mathrm{ex,invest}_{h,p,y}`: size of the previously invested
        capacities at location :math:`p` in year :math:`y` \n

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
        capacity_investment = self.zen_model.lp_model.variables["capacity_investment"]
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
                1 * self.zen_model.lp_model.variables["capacity_addition"],
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

    def constraint_technology_lifetime(self):
        """Calculates remaining capacity of technologies based on the lifetime.

        limited lifetime of the technologies. calculates 'capacity', i.e., the
        capacity at the end of the year and 'capacity_previous', i.e., the capacity at
        the beginning of the year.

        .. math::
            S_{h,p,y} = \\sum_{\\tilde{y}=\\max(y_0,y-\\lceil\\frac{l_h}
            {\\Delta^\\mathrm{y}}\\rceil+1)}^y \\Delta S_{h,p,\\tilde{y}}
            + \\sum_{\\hat{y}=\\psi(\\min(y_0-1,y-\\lceil\\frac{l_h}
            {\\Delta^\\mathrm{y}}\\rceil+1))}^{\\psi(y_0)}
            \\Delta s^\\mathrm{ex}_{h,p,\\hat{y}}

        :math:`S_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested capacity
        after construction) at location :math:`p` in year :math:`y` \n
        :math:`\\Delta s^\\mathrm{ex}_{h,p,y}`: size of the previously invested
        capacities at location :math:`p` in year :math:`y`
        """
        lt_range = pd.MultiIndex.from_tuples(
            [
                (t, y, py)
                for t, y in itertools.product(
                    self.zen_model.sets["set_technologies"],
                    self.zen_model.sets["set_years"],
                )
                for py in list(self.get_lifetime_range(t, y))
            ],
            names=[
                "set_technologies",
                "set_years",
                "set_years_prev",
            ],
        )
        lt_range = pd.Series(index=lt_range, data=-1)
        lt_range = (
            lt_range.to_xarray()
            .broadcast_like(self.zen_model.lp_model.variables["capacity"].lower)
            .fillna(0)
        )
        capacity_addition = self.zen_model.lp_model.variables[
            "capacity_addition"
        ].rename({"set_years": "set_years_prev"})
        capacity_addition = capacity_addition.broadcast_like(lt_range)
        expr = (lt_range * capacity_addition).sum("set_years_prev")
        lhs = lp.merge(
            [1 * self.zen_model.lp_model.variables["capacity"], expr],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        lhs_previous = lp.merge(
            [
                1 * self.zen_model.lp_model.variables["capacity_previous"],
                expr,
                1 * self.zen_model.lp_model.variables["capacity_addition"],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = xr.align(
            lhs.const, self.zen_model.parameters.existing_capacities, join="left"
        )[1]
        constraints = lhs == rhs
        constraints_previous = lhs_previous == rhs

        ### return
        self.zen_model.add_constraint("constraint_technology_lifetime", constraints)
        self.zen_model.add_constraint(
            "constraint_technology_lifetime_previous", constraints_previous
        )

    def constraint_technology_diffusion_limit(self):
        """Limits technology diffusion based on existing capacity in the previous year.

        For storage and conversion technologies: \n
        .. math::
            \\Delta S_{k,n,y}\\leq ((1+\\vartheta_k)^{\\mathrm{dy}}-1)(K_{k,n,y} +
            \\omega \\sum_{\\tilde{n}\\in\\tilde{\\mathcal{N}}}K_{k,\\tilde{n},y})
            +\\mathrm{dy}(\\xi\\sum_{\\tilde{k}
            \\in\\tilde{\\mathcal{K}}}S_{\\tilde{k},n,y} + \\zeta_k)

        For transport technologies: \n
        .. math::
            \\Delta S_{j,e,y}\\leq ((1+\\vartheta_j)^{\\mathrm{dy}}-1)K_{j,e,y}
            + \\mathrm{dy}(\\xi\\sum_{\\tilde{j}\\in\\tilde{\\mathcal{J}}}
            S_{\\tilde{j},e,y} + \\zeta_j)

        :math:`\\Delta S_{j,e,y}`: size of built technology :math:`j` (invested capacity
        after construction) at location :math:`e` in year :math:`y` \n
        :math:`\\vartheta_j`: maximum diffusion rate of technology :math:`j` which is
        the maximum increase in capacity between investment steps \n
        :math:`K_{j,e,y}`: existing knowledge of how to install the technology :math:`j`
        at location :math:`e` in year :math:`y` \n
        :math:`\\xi`: parameter which specifies the unbounded market share \n
        :math:`\\zeta_j`: parameter which specifies the unbounded capacity addition that
        can be added each year (only for delayed technology deployment) \n
        :math:`dy`: interval between planning periods\n
        :math:`\\omega`: parameter which specifies the knowledge spillover rate

        """
        # load variables and parameters
        capacity_addition = self.zen_model.lp_model.variables["capacity_addition"]
        capacity_existing = self.zen_model.parameters.capacity_existing
        knowledge_depreciation_rate = (
            self.zen_model.parameters.knowledge_depreciation_rate
        )
        interval_between_years = self.config.system.interval_between_years
        spillover_rate = self.zen_model.parameters.knowledge_spillover_rate
        # technology diffusion rate per investment period
        tdr = (
            1 + self.zen_model.parameters.max_diffusion_rate
        ) ** interval_between_years - 1
        tdr = tdr.broadcast_like(capacity_addition.lower)
        tdr_sum = tdr.sum("set_location")
        mask_inf_tdr = ~(tdr == np.inf)
        mask_inf_tdr_sum = ~(tdr_sum == np.inf)
        # if all tdr are inf, we can skip the constraint
        if (~mask_inf_tdr).all():
            return
        # mask for knowledge spillover rate (sr) to exclude transport technologies
        mask_technology_type = pd.Series(
            index=pd.Index(self.zen_model.sets["set_technologies"]), data=1
        )
        mask_technology_type.index.name = "set_technologies"
        mask_technology_type[
            mask_technology_type.index.isin(
                self.zen_model.sets["set_transport_technologies"]
            )
        ] = 0
        mask_technology_type = mask_technology_type.to_xarray()
        # create mask for knowledge spillover rate (sr) to exclude edges
        mask_location = pd.Series(
            index=pd.Index(capacity_addition.coords["set_location"]), data=1
        )
        mask_location.index.name = "set_location"
        mask_location[mask_location.index.isin(self.zen_model.sets["set_edges"])] = 0
        mask_location = mask_location.to_xarray()
        # mask match technology type and location
        mask_transport_edge = (1 - mask_technology_type) & (1 - mask_location)
        mask_not_transport_not_edge = mask_technology_type & mask_location
        mask_technology_location = mask_transport_edge | mask_not_transport_not_edge
        # create xarray for previous years
        years = pd.MultiIndex.from_tuples(
            [
                (y, py)
                for y, py in itertools.product(
                    self.zen_model.sets["set_years"],
                    self.zen_model.sets["set_years"],
                )
                if py < y
            ],
            names=["set_years", "set_years_prev"],
        )
        # only formulate term_knowledge if there are previous years
        term_knowledge_no_spillover = capacity_addition.where(
            xr.DataArray(False)
        )  # dummy term
        term_knowledge = capacity_addition.where(xr.DataArray(False))  # dummy term
        if len(years) != 0:
            # kdr for capacity additions
            kdr = {
                (y, py): (1 - knowledge_depreciation_rate)
                ** (interval_between_years * (y - 1 - py))
                for y, py in years
            }
            kdr = pd.Series(kdr)
            kdr.index.names = ["set_years", "set_years_prev"]
            kdr = kdr.to_xarray().fillna(0)

            years = pd.Series(index=years, data=1)
            years = years.to_xarray().fillna(0)
            # expand and sum capacity addition over all nodes for spillover
            capacity_addition_years = capacity_addition.rename(
                {"set_years": "set_years_prev"}
            ).broadcast_like(years)
            kdr = kdr.broadcast_like(capacity_addition_years.lower)
            term_knowledge_no_spillover = tdr * (capacity_addition_years * kdr).sum(
                "set_years_prev"
            )
            # if spillover rate is not inf, calculate term knowledge with spillover
            if spillover_rate != np.inf:
                location_index = pd.Series(
                    index=pd.MultiIndex.from_product(
                        [
                            capacity_addition.coords["set_location"].values,
                            capacity_addition.coords["set_location"].values,
                        ],
                        names=["set_location", "set_location_temp"],
                    )
                ).to_xarray()
                capacity_addition_location = (
                    capacity_addition_years.rename(
                        {"set_location": "set_location_temp"}
                    )
                    .broadcast_like(location_index)
                    .sel({"set_location_temp": self.zen_model.sets["set_nodes"]})
                    .sum("set_location_temp")
                )
                # calculate term spillover
                term_spillover = capacity_addition_location - capacity_addition_years
                sr = xr.full_like(term_spillover.const, spillover_rate)
                sr = sr.where(mask_technology_type, 0).where(mask_location, 0)
                # annual knowledge addition
                term_knowledge = capacity_addition_years + sr * term_spillover
                term_knowledge = tdr * (term_knowledge * kdr).sum("set_years_prev")
        # unbounded market share --> only for same technology class
        capacity_previous = self.zen_model.lp_model.variables["capacity_previous"]
        market_share_unbounded = {
            (t, ot): (
                self.zen_model.parameters.market_share_unbounded
                if self.zen_model.sets["set_reference_carriers"][t][0]
                == self.zen_model.sets["set_reference_carriers"][ot][0]
                else 0
            )
            for t in self.zen_model.sets["set_technologies"]
            for ot in self._get_class_set_of_element(t, Technology)
        }
        market_share_unbounded = pd.Series(market_share_unbounded)
        market_share_unbounded.index.names = [
            "set_technologies",
            "set_other_technologies",
        ]
        market_share_unbounded = (
            market_share_unbounded.to_xarray()
            .broadcast_like(capacity_previous.lower)
            .fillna(0)
        )
        mask_market_share_unbounded = market_share_unbounded != 0
        term_unbounded_addition = (
            (
                market_share_unbounded
                * capacity_previous.rename(
                    {"set_technologies": "set_other_technologies"}
                )
            )
            .where(mask_market_share_unbounded)
            .sum("set_other_technologies")
        )
        # existing capacities
        delta_years = interval_between_years * (
            capacity_addition.coords["set_years"] - 1 - self.energy_system.set_years[0]
        )
        lifetime_existing = self.zen_model.parameters.lifetime_existing
        lifetime = self.zen_model.parameters.lifetime
        kdr_existing = (1 - knowledge_depreciation_rate) ** (
            delta_years + lifetime - lifetime_existing
        )
        capacity_existing_total_nosr = capacity_existing
        # capacity addition unbounded
        capacity_addition_unbounded = (
            self.zen_model.parameters.capacity_addition_unbounded
        )
        capacity_addition_unbounded = capacity_addition_unbounded.broadcast_like(tdr)
        capacity_addition_unbounded = capacity_addition_unbounded.where(
            mask_technology_location, 0
        )
        # build constraints for all nodes summed ("sn")
        lhs_sn = lp.merge(
            [
                1 * capacity_addition,
                -1 * term_knowledge_no_spillover,
                -1 * term_unbounded_addition,
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        ).sum("set_location")
        rhs_sn = (
            tdr
            * (capacity_existing_total_nosr * kdr_existing).sum(
                "set_technologies_existing"
            )
            + capacity_addition_unbounded
        ).sum("set_location")
        rhs_sn = rhs_sn.broadcast_like(lhs_sn.const)
        # mask for tdr == inf
        lhs_sn = self.align_and_mask(lhs_sn, mask_inf_tdr_sum)
        rhs_sn = self.align_and_mask(rhs_sn, mask_inf_tdr_sum)
        # combine constraint
        constraints_sn = lhs_sn <= rhs_sn
        self.zen_model.add_constraint(
            "constraint_technology_diffusion_limit_total", constraints_sn
        )
        # build constraints for all nodes ("an") if spillover rate is not inf
        if spillover_rate != np.inf:
            # existing capacities with spillover
            capacity_existing_total = capacity_existing + spillover_rate * (
                capacity_existing.sum("set_location") - capacity_existing
            ).where(mask_technology_type, 0)
            lhs_an = lp.merge(
                [
                    1 * capacity_addition,
                    -1 * term_knowledge,
                    -1 * term_unbounded_addition,
                ],
                compat="broadcast_equals",
                join="outer",
                cls=LinearExpression,
            )
            rhs_an = (
                tdr
                * (capacity_existing_total * kdr_existing).sum(
                    "set_technologies_existing"
                )
                + capacity_addition_unbounded
            )
            rhs_an = rhs_an.broadcast_like(lhs_an.const)
            # mask for tdr == inf
            lhs_an = self.align_and_mask(lhs_an, mask_inf_tdr)
            rhs_an = self.align_and_mask(rhs_an, mask_inf_tdr)
            # combine constraint
            constraints_an = lhs_an <= rhs_an
            self.zen_model.add_constraint(
                "constraint_technology_diffusion_limit", constraints_an
            )

    def _get_class_set_of_element(
        self, element_name: str, class_name: type[Element]
    ) -> ZenSet:
        """Returns the set of all elements in the class of the element.

        :param element_name: name of element
        :param klass: class of the elements to return
        :return: class_set: set of all elements in the class of the element
        """
        element = self.element_registry.get_element(class_name, element_name)
        if element is None:
            raise ValueError(f"Element {element_name} not found in class {class_name}")
        return self.zen_model.sets[element.label]

    def constraint_cost_capex_yearly(self, index: ZenIndex):
        """Aggregates the capex of built capacity and of existing capacity.

        .. math::
            A_{h,p,y} = f_h (\\sum_{\\tilde{y} = \\max(y_0,y-\\lceil\\frac{l_h}
            {\\mathrm{dy}}\\rceil+1)}^y \\alpha_{h,y}\\Delta S_{h,p,\\tilde{y}}
            + \\sum_{\\hat{y}=\\psi(\\min(y_0-1,y-\\lceil\\frac{l_h}
            {\\mathrm{dy}}\\rceil+1))}^{\\psi(y_0)} \\alpha_{h,y_0}
            \\Delta s^\\mathrm{ex}_{h,p,\\hat{y}})

        :math:`A_{h,p,y}`: annual capex of technology :math:`h` at location :math:`p`
        in year :math:`y` \n
        :math:`f_h`: annuity factor of technology :math:`h` \n
        :math:`\\alpha_{h,y}`: unit cost of capital investment of technology :math:`h`
        in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}`: size of built technology :math:`h` (invested capacity
        after construction) at location :math:`p` in year :math:`y` \n
        :math:`\\Delta s^\\mathrm{ex}_{h,p,y}`: size of the previously added capacities
        at location :math:`p` in year :math:`y` \n
        :math:`l_h`: depreciation time of technology :math:`h`   \n
        :math:`\\mathrm{dy}`: interval between planning periods


        """
        ### masks
        # not needed

        # Annuity factor
        dr = self.zen_model.parameters.discount_rate
        lt = self.zen_model.parameters.depreciation_time

        if dr != 0:
            a = ((1 + dr) ** lt * dr) / ((1 + dr) ** lt - 1)
        else:
            a = 1 / lt

        lt_range = pd.MultiIndex.from_tuples(
            [
                (t, y, py)
                for t, y in index.get_unique(["set_technologies", "set_years"])
                for py in list(
                    self.get_lifetime_range(t, y, use_depreciation_time=True)
                )
            ]
        )

        lt_range = pd.Series(index=lt_range, data=-1)
        lt_range.index.names = [
            "set_technologies",
            "set_years",
            "set_years_prev",
        ]
        lt_range = (
            lt_range.to_xarray()
            .broadcast_like(self.zen_model.lp_model.variables["capacity"].lower)
            .fillna(0)
        )

        cost_capex_overnight = self.zen_model.lp_model.variables[
            "cost_capex_overnight"
        ].rename({"set_years": "set_years_prev"})
        cost_capex_overnight = cost_capex_overnight.broadcast_like(lt_range)
        expr = (lt_range * a * cost_capex_overnight).sum("set_years_prev")
        lhs = lp.merge(
            [1 * self.zen_model.lp_model.variables["cost_capex_yearly"], expr],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        rhs = (a * self.zen_model.parameters.existing_capex).broadcast_like(lhs.const)
        constraints = lhs == rhs

        ### return
        self.zen_model.add_constraint("constraint_cost_capex_yearly", constraints)

    def constraint_cost_opex_yearly(self):
        """Yearly opex for a technology at a location in each year.

        .. math::
            OPEX_{h,p,y} = \\sum_{t\\in\\mathcal{T}}\\tau_t O_{h,p,t}^t
            + \\gamma_{h,y} S_{h,p,y} + \\gamma_{k,y}^\\mathrm{e} S_{k,n,y}^\\mathrm{e}

        :math:`OPEX_{h,p,y}`: opex of operating technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\tau_t`: duration of time step :math:`t` \n
        :math:`O_{h,p,t}^t`: variable opex of operating technology :math:`h` at
        location :math:`p` in time step :math:`t` \n
        :math:`\\gamma_{h,y}`: specific fixed opex of technology :math:`h` in
        year :math:`y` \n
        :math:`S_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`\\gamma_{k,y}^\\mathrm{e}`: specific fixed opex of storage
        technology :math:`k` in year :math:`y` \n
        :math:`S_{k,n,y}^\\mathrm{e}`: installed capacity of storage
        technology :math:`k` at node :math:`n` in year :math:`y`

        """
        times_dict: dict[str, pd.Series] = {
            y: self.zen_model.parameters.time_steps_operation_duration.loc[
                self.time_steps.get_time_steps_year2operation(y)
            ].to_series()
            for y in self.zen_model.sets["set_years"]
        }
        times = pd.concat(times_dict, keys=times_dict.keys())
        times.index.names = ["set_years", "set_time_steps_operation"]
        times = times.to_xarray().broadcast_like(
            self.zen_model.lp_model.variables["cost_opex_variable"].mask
        )
        term_opex_variable = (
            self.zen_model.lp_model.variables["cost_opex_variable"] * times
        ).sum("set_time_steps_operation")
        term_opex_fixed = (
            self.zen_model.parameters.opex_specific_fixed
            * self.zen_model.lp_model.variables["capacity"]
        ).sum("set_capacity_types")
        lhs = (
            self.zen_model.lp_model.variables["cost_opex_yearly"]
            - term_opex_variable
            - term_opex_fixed
        )
        rhs = 0
        constraints = lhs == rhs

        ### return
        self.zen_model.add_constraint("constraint_cost_opex_yearly", constraints)

    def constraint_carbon_emissions_technology_total(self):
        """Calculate total carbon emissions of each technology.

        .. math::
            E_y^{\\mathcal{H}} = \\sum_{p\\in\\mathcal{P}}
            \\sum_{t\\in\\mathcal{T}}\\sum_{h\\in\\mathcal{H}} \\theta_{h,p,t} \\tau_{t}

        :math:`E_y^{\\mathcal{H}}`: total carbon emissions of each technology in
        year :math:`y` \n
        :math:`\\theta_{h,p,t}`: carbon emissions of technology :math:`h` at
        location :math:`p` in time step :math:`t` \n
        :math:`\\tau_{t}`: duration of time step :math:`t`

        """
        term_summed_carbon_emissions_technology = (
            self.zen_model.lp_model.variables["carbon_emissions_technology"]
            * self.get_year_time_step_duration_array()
        ).sum(["set_technologies", "set_location", "set_time_steps_operation"])
        lhs = (
            self.zen_model.lp_model.variables["carbon_emissions_technology_total"]
            - term_summed_carbon_emissions_technology
        )
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint(
            "constraint_carbon_emissions_technology_total", constraints
        )

    def constraint_technology_on_off(self, techs_on_off):
        """If technology is on, the binary variable is 1, else 0.

        The min load constraint is expressed as six constraints
        (here for conversion technologies):

        .. math::
             m^\\mathrm{min}_{i,n,t}S^\\mathrm{approx}_{i,n,t}\\leq
             G^\\mathrm{r}_{i,n,t} \\leq S^\\mathrm{approx}_{i,n,t} \n
             0 \\leq S^\\mathrm{approx}_{i,n,t}
             \\leq s^\\mathrm{max}_{i,n,y} B_{i,n,t} \n
             S_{i,n,y} - s^\\mathrm{max}_{i,n,y}(1-B_{i,n,t})
             \\leq S^\\mathrm{approx}_{i,n,t} \\leq S_{i,n,y}

        :math:`m^\\mathrm{min}_{i,n,t}`: minimum load parameter for
        technology :math:`i`, node :math:`n`, time step :math:`t` \n
        :math:`G_{i,n,t}^\\mathrm{r}`: reference carrier flow of the
        technology :math:`i` at node :math:`n` in time step :math:`t` \n
        :math:`S_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y` \n
        :math:`B_{i,n,t}`: binary variable indicating whether the technology is on or
        off for technology :math:`i`, node :math:`n`, time step :math:`t` \n
        :math:`S^\\mathrm{approx}_{i,n,t}`: helper variable that represents the product
        of :math:`S_{i,n,y}` and :math:`B_{i,n,t}` \n
        :math:`s^\\mathrm{max}_{i,n,y}`: Big-M limit on :math:`S_{h,p,y}`
        """
        # sets
        conversion_techs = self.zen_model.sets["set_conversion_technologies"]
        storage_techs = self.zen_model.sets["set_storage_technologies"]
        transport_techs = self.zen_model.sets["set_transport_technologies"]
        nodes = self.zen_model.sets["set_nodes"]
        times = self.zen_model.sets["set_time_steps_operation"]
        time_step_year = xr.DataArray(
            [self.time_steps.convert_time_step_operation2year(t) for t in times.data],
            coords=[times],
            dims=["set_time_steps_operation"],
        )
        if len(techs_on_off) == 0:
            return None
        # params and variables
        min_load = self.zen_model.parameters.min_load
        capacity = self.zen_model.lp_model.variables["capacity"].sel(
            {"set_capacity_types": "power", "set_years": time_step_year}
        )
        big_M = capacity.upper
        binary = self.zen_model.lp_model.variables["tech_on_var"]
        capacity_on_off_helper = self.zen_model.lp_model.variables[
            "capacity_on_off_helper_var"
        ]
        # mask for on_off variables
        mask_on_off = binary.mask
        # assert that no big-M is inf
        sel_big_M = (big_M.where(mask_on_off) == np.inf).to_series()
        big_M_elements = sel_big_M[sel_big_M].index.droplevel(2).unique().to_list()
        assert ~sel_big_M.any(), (
            f"Big-M is inf for {big_M_elements}. "
            f"Please set finite capacity limits of the technologies."
        )
        # flows
        list_flow_reference = []
        if len(conversion_techs) > 0:
            list_flow_reference.append(
                self.get_flow_expression_conversion(conversion_techs, nodes).rename(
                    {
                        "set_conversion_technologies": "set_technologies",
                        "set_nodes": "set_location",
                    }
                )
            )
        if len(storage_techs) > 0:
            list_flow_reference.append(self.get_flow_expression_storage(rename=True))
        if len(transport_techs) > 0:
            list_flow_reference.append(
                self.zen_model.lp_model.variables["flow_transport"]
                .rename(
                    {
                        "set_transport_technologies": "set_technologies",
                        "set_edges": "set_location",
                    }
                )
                .to_linexpr()
            )
        flow_reference = lp.merge(
            list_flow_reference,
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        flow_reference = flow_reference.reindex_like(mask_on_off)
        # constraints
        # constraint 1, operational limit
        # 1a, lower bound
        lhs_1a = self.align_and_mask(
            min_load * capacity_on_off_helper - flow_reference, mask_on_off
        )
        rhs_1a = 0
        constraints_1a = lhs_1a <= rhs_1a
        self.zen_model.add_constraint(
            "constraint_technology_on_off_operation_lower_bound", constraints_1a
        )
        # 1a, upper bound
        lhs_1b = self.align_and_mask(
            -capacity_on_off_helper + flow_reference, mask_on_off
        )
        rhs_1b = 0
        constraints_1b = lhs_1b <= rhs_1b
        self.zen_model.add_constraint(
            "constraint_technology_on_off_operation_upper_bound", constraints_1b
        )
        # constraint 2, limit capacity helper
        # (lower bound already given by variable definition)
        lhs_2 = self.align_and_mask(
            capacity_on_off_helper - big_M * binary, mask_on_off
        )
        rhs_2 = 0
        constraints_2 = lhs_2 <= rhs_2
        self.zen_model.add_constraint(
            "constraint_technology_on_off_capacity_helper", constraints_2
        )
        # constraint 3, capacity helper bounds
        # 3a, lower bound
        lhs_3a = self.align_and_mask(
            capacity + big_M * binary - capacity_on_off_helper, mask_on_off
        )
        rhs_3a = big_M
        constraints_3a = lhs_3a <= rhs_3a
        self.zen_model.add_constraint(
            "constraint_technology_on_off_capacity_helper_lower_bound", constraints_3a
        )
        # 3b, upper bound
        lhs_3b = self.align_and_mask(capacity_on_off_helper - capacity, mask_on_off)
        rhs_3b = 0
        constraints_3b = lhs_3b <= rhs_3b
        self.zen_model.add_constraint(
            "constraint_technology_on_off_capacity_helper_upper_bound", constraints_3b
        )

    def get_lifetime_range(self, tech, year, use_depreciation_time=False):
        """Get active year range of technology: either lifetime or depreciation time.

        :param tech: name of the technology
        :param year: yearly time step
        :param use_depreciation_time: boolean indicating whether to use depreciation
            time instead of lifetime, namely for CAPEX calculation
        :return: lifetime or depreciation time range of technology
        """
        first_lifetime_year = self.get_first_lifetime_time_step(
            tech,
            year,
            use_depreciation_time,
        )
        first_lifetime_year = max(
            first_lifetime_year,
            cast(int, self.zen_model.sets["set_years"][0]),
        )
        return range(first_lifetime_year, year + 1)

    def get_first_lifetime_time_step(self, tech, year, use_depreciation_time=False):
        """Get first time step of active capacity of technology.

        Returns the first time step within the lifetime or depreciation time of the
        technology, i.e., the earliest past time step whose installed capacity is
        still active at the given time step.

        :param tech: name of the technology
        :param year: current yearly time step
        :param use_depreciation_time: boolean indicating whether to use depreciation
            time instead of standard lifetime for capacity calculation
        :return: first time step where capacity or investment is still valid
        """
        # get params and system
        params = self.zen_model.parameters.dict_parameters
        lifetime = (
            params.depreciation_time[tech]
            if use_depreciation_time
            else params.lifetime[tech]
        )
        # conservative estimate of lifetime (floor)
        del_lifetime = (
            int(np.floor(lifetime / self.config.system.interval_between_years)) - 1
        )
        return year - del_lifetime
