import itertools
from typing import TYPE_CHECKING

import linopy as lp
import numpy as np
import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.element import Element
from zen_garden.elements.technology import Technology
from zen_garden.model.components.zen_set import BaseSet
from zen_garden.topology.generic_constraint import GenericConstraint

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.model.config import Config
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.model.zen_model import ZenModel
    from zen_garden.services.element_registry import ElementRegistry


class TechnologyDiffusionLimitConstraint(GenericConstraint):
    def __init__(
        self,
        config: "Config",
        zen_model: "ZenModel",
        energy_system: "EnergySystem",
        time_steps: "TimeStepsDicts",
        element_registry: "ElementRegistry",
    ):
        super().__init__(config, zen_model, energy_system, time_steps)
        self.element_registry = element_registry

    def build(self):
        """Summary:
        Limit additions using depreciated installation knowledge.

        When knowledge spillover is finite, a location-wise constraint is added
        for each applicable technology location. Conversion and storage
        technologies can receive node-to-node spillover; transport technologies
        are constrained on edges and receive no node-to-node spillover.

        Formulation:

        .. math::
            \\Delta K_{h,p,y}\\leq
            d_{h,y}K^\\omega_{h,p,y}
            +\\chi\\sum_{\\tilde h\\in\\tilde{\\mathcal H}}
            K^{\\mathrm{prev}}_{\\tilde h,p,y}
            +k^{\\mathrm{add,free}}_h.

        where :math:`d_{h,y}=(1+r^{\\mathrm{diff}}_{h,y})^{\\Delta y}-1`.
        The unbounded market share term :math:`\\chi` is separate from
        knowledge spillover. It uses :math:`K^{\\mathrm{prev}}` for other
        technologies in the target technology's class (excluding the target
        itself) whose reference carrier matches the target. The term is
        calculated at the target location; ``market_share_unbounded`` supplies
        the coefficient :math:`\\chi`.

        Define the no-spillover knowledge stock as

        .. math::
            K^0_{h,p,y}=K^{\\mathrm{add},0}_{h,p,y}+K^{\\mathrm{ex},0}_{h,p,y},
            \\
            K^{\\mathrm{add},0}_{h,p,y}=\\sum_{\\tilde y<y}
            (1-\\delta_h)^{\\Delta y\\,(y-1-\\tilde y)}
            \\Delta K_{h,p,\\tilde y}.

        Here :math:`K^{\\mathrm{ex},0}` is existing capacity at :math:`p`
        weighted by its remaining-lifetime depreciation factor. For finite
        :math:`\\omega`, conversion and storage technologies use

        .. math::
            K^\\omega_{h,p,y}=K^0_{h,p,y}+\\omega
            \\sum_{p'\\ne p}K^0_{h,p',y},

        where the sum is over other nodes. For transport technologies,
        :math:`K^\\omega=K^0` because node-to-node spillover is zero.

        For finite maximum diffusion rates, a global constraint is added:

        .. math::
            \\sum_p\\Delta K_{h,p,y}\\leq
            \\sum_p\\left[
            d_{h,y}K^0_{h,p,y}
            +\\chi\\sum_{\\tilde h\\in\\tilde{\\mathcal H}}
            K^{\\mathrm{prev}}_{\\tilde h,p,y}
            +k^{\\mathrm{add,free}}_h\\right].

        If :math:`\\omega=\\infty`, only the global constraint is created.
        In the global constraint, :math:`K^0` is the depreciated installation-knowledge
        stock without node-to-node spillover. It is distinct from K^\\omega,
        which is used by the location-wise constraints when spillover applies.

        For storage technologies, each equation is applied independently to power
        and energy capacity.

        Notation:

        :math:`\\Delta K_{h,p,y}`: size of built technology :math:`h` (invested capacity
        after construction) at location :math:`p` in year :math:`y`
        :math:`r^{\\mathrm{diff}}_{h,y}`: maximum diffusion rate of technology
        :math:`h` in year :math:`y`, which is the maximum increase in
        capacity between investment steps
        :math:`K^\\omega_{h,p,y}`: depreciated installation-knowledge stock
        at location :math:`p`, including applicable knowledge spillover
        :math:`K^0_{h,p,y}`: depreciated installation-knowledge stock at
        location :math:`p` without node-to-node knowledge spillover
        :math:`\\chi`: parameter which specifies the unbounded market share
        :math:`K^{\\mathrm{prev}}_{h,p,y}`: capacity available before the
        current year's addition
        :math:`\\tilde{\\mathcal H}`: other technologies in the target
        technology's class, excluding the target itself, with the same
        reference carrier
        :math:`k^{\\mathrm{add,free}}_h`: parameter which specifies the unbounded
        capacity addition that can be added each investment step (only for
        delayed technology deployment)
        :math:`\\Delta y`: interval between planning periods
        :math:`\\omega`: parameter which specifies the knowledge spillover rate
        """
        # load variables and parameters
        capacity_addition = self.zen_model.variables["capacity_addition"]
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
        capacity_previous = self.zen_model.variables["capacity_previous"]
        market_share_unbounded = {
            (t, ot): (
                self.zen_model.parameters.market_share_unbounded
                if ot != t
                and self.zen_model.sets["set_reference_carriers"][t][0]
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
    ) -> BaseSet:
        """Returns the set of all elements in the class of the element.

        :param element_name: name of element
        :param klass: class of the elements to return
        :return: class_set: set of all elements in the class of the element
        """
        element = self.element_registry.get_element(class_name, element_name)
        if element is None:
            raise ValueError(f"Element {element_name} not found in class {class_name}")
        return self.zen_model.sets[element.label]
