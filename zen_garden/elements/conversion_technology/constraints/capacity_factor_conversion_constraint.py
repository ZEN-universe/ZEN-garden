import xarray as xr

from zen_garden.topology.generic_constraint import GenericConstraint


class CapacityFactorConversionConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Load is limited by the installed capacity and the maximum load factor.

        Formulation:

        .. math::
            F^{\\mathrm{ref}}_{h,n,t} \\leq
            \\ell^{\\mathrm{max}}_{h,n,t}K_{h,n,y}

        Notation:

        :math:`m_{h,n,t}^{\\mathrm{max}}`: maximum load factor of the
        technology :math:`h` at node :math:`n` in time step :math:`t` of year
        :math:`y`
        :math:`K_{h,n,y}`: installed capacity of the technology :math:`h` at
        node :math:`n` in year :math:`y`
        :math:`F^{\\mathrm{ref}}_{h,n,t}`: reference carrier flow of the
        technology :math:`h` at node :math:`n` in time step :math:`t` of year
        :math:`y`
        """
        techs = model_constructor.zen_model.sets["set_conversion_technologies"]
        if len(techs) == 0:
            return
        nodes = model_constructor.zen_model.sets["set_nodes"]
        times = model_constructor.zen_model.parameters.max_load.coords[
            "set_time_steps_operation"
        ]
        time_step_year = xr.DataArray(
            [
                model_constructor.time_steps.convert_time_step_operation2year(t)
                for t in times.data
            ],
            coords=[times],
        )
        term_capacity = (
            model_constructor.zen_model.parameters.max_load.loc[techs, nodes, :]
            * model_constructor.zen_model.variables["capacity"].loc[
                techs, "power", nodes, time_step_year
            ]
        ).rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        )
        term_reference_flow = cls.get_flow_expression_conversion(
            model_constructor, techs, nodes
        )
        lhs = term_capacity - term_reference_flow
        rhs = 0
        constraints = lhs >= rhs

        model_constructor.zen_model.add_constraint(
            "constraint_capacity_factor_conversion", constraints
        )
