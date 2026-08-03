from typing import cast

import linopy as lp
import numpy as np
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.constraints.generic_constraint import GenericConstraint


class TechnologyOnOffConstraint(GenericConstraint):
    def build(self, techs_on_off=None):
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
        assert techs_on_off is not None, "techs_on_off must be provided"

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
        capacity = self.zen_model.variables["capacity"].sel(
            {"set_capacity_types": "power", "set_years": time_step_year}
        )
        big_M = capacity.upper
        binary = self.zen_model.variables["tech_on_var"]
        capacity_on_off_helper = self.zen_model.variables["capacity_on_off_helper_var"]
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
                self.zen_model.variables["flow_transport"]
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
