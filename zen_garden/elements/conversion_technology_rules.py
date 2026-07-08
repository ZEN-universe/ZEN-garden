import itertools
import logging

import linopy as lp
import numpy as np
import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.elements.generic_rule import GenericRule
from zen_garden.utils import align_like

logger = logging.getLogger(__name__)


class ConversionTechnologyRules(GenericRule):
    """Rules for the ConversionTechnology class."""

    def constraint_capacity_factor_conversion(self):
        """Load is limited by the installed capacity and the maximum load factor.

        .. math::
            G_{i,n,t}^\\mathrm{r} \\leq m^{\\mathrm{max}}_{i,n,t}S_{i,n,y}

        :math:`m_{i,n,t}^{\\mathrm{max}}`: maximum load factor of the
        technology :math:`i` at node :math:`n` in time step :math:`t` \n
        :math:`S_{i,n,y}`: installed capacity of the technology :math:`i` at
        node :math:`n` in year :math:`y` \n
        :math:`G_{i,n,t}^\\mathrm{r}`: reference carrier flow of the
        technology :math:`i` at node :math:`n` in time step :math:`t`


        """
        techs = self.zen_model.sets["set_conversion_technologies"]
        if len(techs) == 0:
            return
        nodes = self.zen_model.sets["set_nodes"]
        times = self.zen_model.parameters.max_load.coords["set_time_steps_operation"]
        time_step_year = xr.DataArray(
            [
                self.energy_system.time_steps.convert_time_step_operation2year(t)
                for t in times.data
            ],
            coords=[times],
        )
        term_capacity = (
            self.zen_model.parameters.max_load.loc[techs, nodes, :]
            * self.zen_model.lp_model.variables["capacity"].loc[
                techs, "power", nodes, time_step_year
            ]
        ).rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        )
        term_reference_flow = self.get_flow_expression_conversion(techs, nodes)
        lhs = term_capacity - term_reference_flow
        rhs = 0
        constraints = lhs >= rhs

        self.zen_model.constraints.add_constraint(
            "constraint_capacity_factor_conversion", constraints
        )

    def constraint_minimum_full_load_hours(self):
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
            * self.zen_model.lp_model.variables["capacity"]
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

        self.zen_model.constraints.add_constraint(
            "constraint_minimum_full_load_hours", constraints
        )

    def constraint_opex_emissions_technology_conversion(self):
        """Calculate opex and carbon emissions of each technology.

        .. math::
            O_{h,p,t}^\\mathrm{t} = \\beta_{h,p,t} G_{i,n,t}^\\mathrm{r} \n
            \\theta_{h,p,t} = \\epsilon_h G_{i,n,t}^\\mathrm{r}

        :math:`O_{h,p,t}^\\mathrm{t}`: variable opex of the technology :math:`h` at
        node :math:`p` in time step :math:`t` \n
        :math:`\\beta_{h,p,t}`: specific variable opex of the technology :math:`h` at
        node :math:`p` in time step :math:`t` \n
        :math:`G_{i,n,t}^\\mathrm{r}`: reference carrier flow of the
        technology :math:`i` at node :math:`n` in time step :math:`t` \n
        :math:`\\theta^{\\mathrm{tech}}_{h,p,t}`: carbon emissions of operating the
        technology :math:`h` at node :math:`p` in time step :math:`t` \n
        :math:`\\epsilon_h`: carbon intensity of the reference carrier of
        technology :math:`h`


        """
        techs = self.zen_model.sets["set_conversion_technologies"]
        if len(techs) == 0:
            return
        nodes = self.zen_model.sets["set_nodes"]
        term_reference_flow_opex = self.get_flow_expression_conversion(
            techs,
            nodes,
            factor=self.zen_model.parameters.opex_specific_variable.rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            ),
        )
        term_reference_flow_emissions = self.get_flow_expression_conversion(
            techs,
            nodes,
            factor=self.zen_model.parameters.carbon_intensity_technology.rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            ),
        )
        lhs_opex = (
            1
            * self.zen_model.lp_model.variables["cost_opex_variable"].loc[
                techs, nodes, :
            ]
        ).rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        ) - term_reference_flow_opex
        lhs_emissions = (
            1
            * self.zen_model.lp_model.variables["carbon_emissions_technology"].loc[
                techs, nodes, :
            ]
        ).rename(
            {
                "set_technologies": "set_conversion_technologies",
                "set_location": "set_nodes",
            }
        ) - term_reference_flow_emissions
        rhs = 0
        constraints_opex = lhs_opex == rhs
        constraints_emissions = lhs_emissions == rhs

        self.zen_model.constraints.add_constraint(
            "constraint_opex_technology_conversion", constraints_opex
        )
        self.zen_model.constraints.add_constraint(
            "constraint_carbon_emissions_technology_conversion", constraints_emissions
        )

    def constraint_linear_capex(self):
        """If capacity and capex have a linear relationship.

        .. math::
            A_{h,p,y}^{approximation} = \\alpha_{h,n,y} \\Delta S_{h,p,y}^{approx}

        :math:`A_{h,p,y}^{approx}`: approximated capex of the technology :math:`h`
        at node :math:`p` in year :math:`y` \n
        :math:`\\alpha_{h,n,y}`: specific capex of the technology :math:`h` at
        node :math:`n` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}^{approx}`: approximated capacity of the
        technology :math:`h` at node :math:`p` in year :math:`y`

        """
        capex_specific_conversion = self.zen_model.parameters.capex_specific_conversion
        capex_specific_conversion = capex_specific_conversion.rename(
            {
                old: new
                for old, new in zip(
                    list(capex_specific_conversion.dims),
                    [
                        "set_conversion_technologies",
                        "set_nodes",
                        "set_time_steps_yearly",
                    ],
                    strict=False,
                )
            }
        )
        capex_specific_conversion = capex_specific_conversion.broadcast_like(
            self.zen_model.lp_model.variables["capacity_approximation"].lower
        )
        mask = ~np.isnan(capex_specific_conversion)
        lhs = lp.merge(
            [
                1 * self.zen_model.lp_model.variables["capex_approximation"],
                -capex_specific_conversion
                * self.zen_model.lp_model.variables["capacity_approximation"],
            ],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        )
        lhs = self.align_and_mask(lhs, mask)
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.constraints.add_constraint(
            "constraint_linear_capex", constraints
        )

    def constraint_capacity_capex_coupling(self):
        """Couples capacity variables based on modeling technique.

        .. math::
            \\Delta S_{h,p,y} = \\Delta S_{h,p,y}^\\mathrm{approx}

        :math:`\\Delta S_{h,p,y}`: capacity addition of the technology :math:`h` at
        node :math:`p` in year :math:`y` \n
        :math:`\\Delta S_{h,p,y}^\\mathrm{approx}`: approximated capacity addition of
        the technology :math:`h` at node :math:`p` in year :math:`y`

        """
        techs = self.zen_model.sets["set_conversion_technologies"]
        nodes = self.zen_model.sets["set_nodes"]
        capacity_addition = (
            self.zen_model.lp_model.variables["capacity_addition"]
            .loc[techs, "power", nodes]
            .rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            )
        )
        cost_capex_overnight = (
            self.zen_model.lp_model.variables["cost_capex_overnight"]
            .loc[techs, "power", nodes]
            .rename(
                {
                    "set_technologies": "set_conversion_technologies",
                    "set_location": "set_nodes",
                }
            )
        )

        ### formulate constraint
        lhs_capacity = (
            capacity_addition
            - self.zen_model.lp_model.variables["capacity_approximation"]
        )
        lhs_capex = (
            cost_capex_overnight
            - self.zen_model.lp_model.variables["capex_approximation"]
        )
        rhs = 0
        constraints_capacity = lhs_capacity == rhs
        constraints_capex = lhs_capex == rhs
        ### return
        self.zen_model.constraints.add_constraint(
            "constraint_capacity_coupling", constraints_capacity
        )
        self.zen_model.constraints.add_constraint(
            "constraint_capex_coupling", constraints_capex
        )

    def constraint_carrier_conversion(self):
        """Conversion factor between reference carrier and dependent carrier.

        .. math::
            G^\\mathrm{d}_{i,n,t} = \\eta_{i,c,n,y}G^\\mathrm{r}_{i,n,t}

        :math:`G^\\mathrm{d}_{i,n,t}`: dependent carrier flow of the
        technology :math:`i` at node :math:`n` in time step :math:`t` \n
        :math:`\\eta_{i,c,n,y}`: conversion factor of the technology :math:`i` from
        reference carrier to dependent carrier :math:`c` at node :math:`n`
        in year :math:`y` \n
        :math:`G^\\mathrm{r}_{i,n,t}`: reference carrier flow of the
        technology :math:`i` at node :math:`n` in time step :math:`t`

        """
        # dependent carriers
        flow_conversion_input_dep = self.zen_model.lp_model.variables[
            "flow_conversion_input"
        ].rename({"set_input_carriers": "set_dependent_carriers"})
        flow_conversion_output_dep = self.zen_model.lp_model.variables[
            "flow_conversion_output"
        ].rename({"set_output_carriers": "set_dependent_carriers"})
        dc_in = pd.Series(
            {
                (t, c): (
                    True
                    if c in self.zen_model.sets["set_dependent_carriers"][t]
                    else False
                )
                for t, c in itertools.product(
                    self.zen_model.sets["set_conversion_technologies"],
                    self.zen_model.sets["set_input_carriers"].superset,
                )
            }
        )
        dc_out = pd.Series(
            {
                (t, c): (
                    True
                    if c in self.zen_model.sets["set_dependent_carriers"][t]
                    else False
                )
                for t, c in itertools.product(
                    self.zen_model.sets["set_conversion_technologies"],
                    self.zen_model.sets["set_output_carriers"].superset,
                )
            }
        )
        dc_in.index.names = ["set_conversion_technologies", "set_dependent_carriers"]
        dc_out.index.names = ["set_conversion_technologies", "set_dependent_carriers"]
        combined_dependent_index = xr.align(
            flow_conversion_input_dep.lower,
            flow_conversion_output_dep.lower,
            join="outer",
        )[0]
        dc_in = align_like(dc_in.to_xarray(), combined_dependent_index, astype=bool)
        dc_out = align_like(dc_out.to_xarray(), combined_dependent_index, astype=bool)
        dc = dc_in | dc_out
        term_flow_dependent = lp.merge(
            [1 * flow_conversion_input_dep, 1 * flow_conversion_output_dep],
            compat="broadcast_equals",
            join="outer",
            cls=LinearExpression,
        ).where(dc)
        conversion_factor = align_like(
            self.zen_model.parameters.conversion_factor, term_flow_dependent
        )
        # reference carriers
        flow_conversion_input = self.zen_model.lp_model.variables[
            "flow_conversion_input"
        ].broadcast_like(conversion_factor)
        flow_conversion_output = self.zen_model.lp_model.variables[
            "flow_conversion_output"
        ].broadcast_like(conversion_factor)
        rc_in = pd.Series(
            {
                (t, c): (
                    True
                    if c in self.zen_model.sets["set_reference_carriers"][t]
                    else False
                )
                for t, c in itertools.product(
                    self.zen_model.sets["set_conversion_technologies"],
                    self.zen_model.sets["set_input_carriers"].superset,
                )
            }
        )
        rc_out = pd.Series(
            {
                (t, c): (
                    True
                    if c in self.zen_model.sets["set_reference_carriers"][t]
                    else False
                )
                for t, c in itertools.product(
                    self.zen_model.sets["set_conversion_technologies"],
                    self.zen_model.sets["set_output_carriers"].superset,
                )
            }
        )
        rc_in.index.names = ["set_conversion_technologies", "set_input_carriers"]
        rc_out.index.names = ["set_conversion_technologies", "set_output_carriers"]
        rc_in = align_like(rc_in.to_xarray(), flow_conversion_input)
        rc_out = align_like(rc_out.to_xarray(), flow_conversion_output)
        term_flow_reference = flow_conversion_input.where(rc_in).sum(
            "set_input_carriers"
        ) + flow_conversion_output.where(rc_out).sum("set_output_carriers")
        # formulate constraint
        lhs = term_flow_dependent - conversion_factor * term_flow_reference
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.constraints.add_constraint(
            "constraint_carrier_conversion", constraints
        )
