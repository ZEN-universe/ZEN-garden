import numpy as np
import xarray as xr

from zen_garden.constraints.generic_constraint import GenericConstraint


class MinimumFullLoadHoursConstraint(GenericConstraint):
    def build(self):
        """Sets minimum full load hours for each unit.

        This constraint requires that a minimum number of full_load_hours be met
        over the course of year. Full load hours are the amount of hours that
        a conversion technology would need to run at full capacity in order
        to produce an output equivalent to its yearly total. The constraint can
        be used to require a conversion technology to always operate at
        baseload capacity. This can be helpful for technologies where ramping
        is not possible or economical for reasons not captured by the model.

        **Mathematical formulation:**

        .. math::
            \\sum_t G_{i,n,t,y}^\\mathrm{r} \\geq
            \\bigg( \\sum_{t \\in\\mathcal{T}} \\tau_t \\bigg)
            \\underline{\\pi}_{i,n,y} S_{i,n,y}
            \\qquad \\forall i,n,y

        The sum simply yields the unaggregated time steps per year, set in the
        systems.json file.

        **Constraint parameters:**

        - :math:`\\underline{\\pi}_{i,n,y}`: minimum number of full load hours,
          expressed as a fraction of the unaggregated time steps per year. Takes
          separate values for each technology :math:`i` at node :math:`n` and
          planning period :math:`y`\n

        **Constraint variables:**

        - :math:`S_{i,n,y}`: installed capacity of the technology :math:`i` at
          node :math:`n` in planning period :math:`y` \n

        - :math:`G_{i,n,t}^\\mathrm{r}`: reference carrier flow of the technology
          :math:`i` at node :math:`n` in time step :math:`t` in planning
          period :math:`y`


        """
        # get dimensions
        techs = self.zen_model.sets["set_conversion_technologies"]
        if len(techs) == 0:
            return
        nodes = self.zen_model.sets["set_nodes"]
        # define mask
        min_full_load_hours_fraction = (
            self.zen_model.parameters.min_full_load_hours_fraction
        )
        mask = xr.DataArray(
            ~np.isclose(min_full_load_hours_fraction, 0),
            dims=min_full_load_hours_fraction.dims,
            coords=min_full_load_hours_fraction.coords,
        )
        # create constraint
        term_capacity = (
            min_full_load_hours_fraction
            * self.config.system.unaggregated_time_steps_per_year
            * self.zen_model.variables["capacity"]
            .sel(
                {
                    "set_technologies": techs,
                    "set_capacity_types": ["power"],
                    "set_location": nodes,
                }
            )
            .rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            )
        )
        term_annual_production = (
            self.get_flow_expression_conversion(techs, nodes)
            * self.get_year_time_step_duration_array()
        ).sum("set_time_steps_operation")

        lhs = term_annual_production.where(mask) - term_capacity.where(mask)
        rhs = 0
        constraints = lhs >= rhs

        self.zen_model.add_constraint("constraint_minimum_full_load_hours", constraints)
