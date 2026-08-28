import linopy as lp
import numpy as np
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.model.component_types.constraint import GenericConstraint


class TechnologyOnOffConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        If technology is on, the binary variable is 1, else 0.

        The minimum-load relation is expressed through a linearized product of
        installed capacity and the on/off binary:

        Formulation:

        .. math::
            \\begin{aligned}
            \\ell^{\\mathrm{min}}_{h,p,t}\\widehat{K}_{h,p,t}
            &\\leq F^{\\mathrm{act}}_{h,p,t}
            \\leq \\widehat{K}_{h,p,t},\\\\
            0 &\\leq \\widehat{K}_{h,p,t}
            \\leq \\overline{k}_{h,p,y}z^{\\mathrm{on}}_{h,p,t},\\\\
            K_{h,p,y}-\\overline{k}_{h,p,y}(1-z^{\\mathrm{on}}_{h,p,t})
            &\\leq \\widehat{K}_{h,p,t}\\leq K_{h,p,y}.
            \\end{aligned}

        Notation:

        :math:`\\ell^{\\mathrm{min}}_{h,p,t}`: minimum load parameter for
        technology :math:`h` at location :math:`p` in time step :math:`t` of
        year :math:`y`
        :math:`F^{\\mathrm{act}}_{h,p,t}`: constrained activity flow:
        :math:`F^{\\mathrm{ref}}_{h,n,t}`
        for conversion technologies, :math:`F^{\\mathrm{ch}}_{h,n,t}
        +F^{\\mathrm{dis}}_{h,n,t}` for storage technologies, and
        :math:`F^{\\mathrm{trans}}_{h,e,t}` for transport technologies
        :math:`K_{h,p,y}`: installed capacity of technology :math:`h` at
        location :math:`p` in year :math:`y`
        :math:`z^{\\mathrm{on}}_{h,p,t}`: binary variable indicating whether
        technology :math:`h` is on at location :math:`p` in time step :math:`t` of
        year :math:`y`
        :math:`\\widehat{K}_{h,p,t}`: helper variable representing the
        product of :math:`K_{h,p,y}` and :math:`z^{\\mathrm{on}}_{h,p,t}`
        :math:`\\overline{k}_{h,p,y}`: Big-M limit on :math:`K_{h,p,y}`
        """
        techs_on_off = model_constructor.zen_model.create_custom_set(
            ["set_technologies", "set_on_off"]
        )[0]

        # sets
        conversion_techs = model_constructor.zen_model.sets[
            "set_conversion_technologies"
        ]
        storage_techs = model_constructor.zen_model.sets["set_storage_technologies"]
        transport_techs = model_constructor.zen_model.sets["set_transport_technologies"]
        nodes = model_constructor.zen_model.sets["set_nodes"]
        times = model_constructor.zen_model.sets["set_time_steps_operation"]
        time_step_year = xr.DataArray(
            [
                model_constructor.time_steps.convert_time_step_operation2year(t)
                for t in times.data
            ],
            coords=[times],
            dims=["set_time_steps_operation"],
        )
        if len(techs_on_off) == 0:
            # No technology needs on/off modelling: drop the helper variables
            # that were only added for this constraint.
            for helper_variable in ("tech_on_var", "capacity_on_off_helper_var"):
                if helper_variable in model_constructor.zen_model.lp_model.variables:
                    model_constructor.zen_model.lp_model.variables.remove(
                        helper_variable
                    )
            return None
        # params and variables
        min_load = model_constructor.zen_model.parameters.min_load
        capacity = model_constructor.zen_model.variables["capacity"].sel(
            {"set_capacity_types": "power", "set_years": time_step_year}
        )
        big_M = capacity.upper
        binary = model_constructor.zen_model.variables["tech_on_var"]
        capacity_on_off_helper = model_constructor.zen_model.variables[
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
                cls.get_flow_expression_conversion(
                    model_constructor, conversion_techs, nodes
                ).rename(
                    {
                        "set_conversion_technologies": "set_technologies",
                        "set_nodes": "set_location",
                    }
                )
            )
        if len(storage_techs) > 0:
            list_flow_reference.append(
                cls.get_flow_expression_storage(model_constructor, rename=True)
            )
        if len(transport_techs) > 0:
            list_flow_reference.append(
                model_constructor.zen_model.variables["flow_transport"]
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
        lhs_1a = cls.align_and_mask(
            min_load * capacity_on_off_helper - flow_reference, mask_on_off
        )
        rhs_1a = 0
        constraints_1a = lhs_1a <= rhs_1a
        model_constructor.zen_model.add_constraint(
            "constraint_technology_on_off_operation_lower_bound", constraints_1a
        )
        # 1a, upper bound
        lhs_1b = cls.align_and_mask(
            -capacity_on_off_helper + flow_reference, mask_on_off
        )
        rhs_1b = 0
        constraints_1b = lhs_1b <= rhs_1b
        model_constructor.zen_model.add_constraint(
            "constraint_technology_on_off_operation_upper_bound", constraints_1b
        )
        # constraint 2, limit capacity helper
        # (lower bound already given by variable definition)
        lhs_2 = cls.align_and_mask(capacity_on_off_helper - big_M * binary, mask_on_off)
        rhs_2 = 0
        constraints_2 = lhs_2 <= rhs_2
        model_constructor.zen_model.add_constraint(
            "constraint_technology_on_off_capacity_helper", constraints_2
        )
        # constraint 3, capacity helper bounds
        # 3a, lower bound
        lhs_3a = cls.align_and_mask(
            capacity + big_M * binary - capacity_on_off_helper, mask_on_off
        )
        rhs_3a = big_M
        constraints_3a = lhs_3a <= rhs_3a
        model_constructor.zen_model.add_constraint(
            "constraint_technology_on_off_capacity_helper_lower_bound", constraints_3a
        )
        # 3b, upper bound
        lhs_3b = cls.align_and_mask(capacity_on_off_helper - capacity, mask_on_off)
        rhs_3b = 0
        constraints_3b = lhs_3b <= rhs_3b
        model_constructor.zen_model.add_constraint(
            "constraint_technology_on_off_capacity_helper_upper_bound", constraints_3b
        )
