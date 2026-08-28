import itertools

import linopy as lp
import pandas as pd
import xarray as xr
from linopy.expressions import LinearExpression

from zen_garden.topology.generic_constraint import GenericConstraint
from zen_garden.utils import align_like


class CarrierConversionConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Conversion factor between reference carrier and dependent carrier.

        Formulation:

        .. math::
            G^\\mathrm{d}_{c,h,n,t} =
            \\eta^{\\mathrm{conv}}_{h,c,n,t}F^{\\mathrm{ref}}_{h,n,t}

        For each technology, the implementation selects reference and dependent
        flows from their configured input or output carrier dimensions.

        Notation:

        :math:`G^\\mathrm{d}_{c,h,n,t}`: dependent carrier flow of the
        technology :math:`h` at node :math:`n` in time step :math:`t` of year
        :math:`y`
        :math:`\\eta^{\\mathrm{conv}}_{h,c,n,t}`: conversion factor of
        technology :math:`h` from
        its reference carrier to dependent carrier :math:`c`
        :math:`F^{\\mathrm{ref}}_{h,n,t}`: reference carrier flow of the
        technology :math:`h` at node :math:`n` in time step :math:`t` of year
        :math:`y`
        """
        # dependent carriers
        flow_conversion_input_dep = model_constructor.zen_model.variables[
            "flow_conversion_input"
        ].rename({"set_input_carriers": "set_dependent_carriers"})
        flow_conversion_output_dep = model_constructor.zen_model.variables[
            "flow_conversion_output"
        ].rename({"set_output_carriers": "set_dependent_carriers"})
        dc_in = pd.Series(
            {
                (t, c): (
                    True
                    if c
                    in model_constructor.zen_model.sets["set_dependent_carriers"][t]
                    else False
                )
                for t, c in itertools.product(
                    model_constructor.zen_model.sets["set_conversion_technologies"],
                    model_constructor.zen_model.sets[
                        "set_input_carriers"
                    ].coordinate_values,
                )
            }
        )
        dc_out = pd.Series(
            {
                (t, c): (
                    True
                    if c
                    in model_constructor.zen_model.sets["set_dependent_carriers"][t]
                    else False
                )
                for t, c in itertools.product(
                    model_constructor.zen_model.sets["set_conversion_technologies"],
                    model_constructor.zen_model.sets[
                        "set_output_carriers"
                    ].coordinate_values,
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
            model_constructor.zen_model.parameters.conversion_factor,
            term_flow_dependent,
        )
        # reference carriers
        flow_conversion_input = model_constructor.zen_model.variables[
            "flow_conversion_input"
        ].broadcast_like(conversion_factor)
        flow_conversion_output = model_constructor.zen_model.variables[
            "flow_conversion_output"
        ].broadcast_like(conversion_factor)
        rc_in = pd.Series(
            {
                (t, c): (
                    True
                    if c
                    in model_constructor.zen_model.sets["set_reference_carriers"][t]
                    else False
                )
                for t, c in itertools.product(
                    model_constructor.zen_model.sets["set_conversion_technologies"],
                    model_constructor.zen_model.sets[
                        "set_input_carriers"
                    ].coordinate_values,
                )
            }
        )
        rc_out = pd.Series(
            {
                (t, c): (
                    True
                    if c
                    in model_constructor.zen_model.sets["set_reference_carriers"][t]
                    else False
                )
                for t, c in itertools.product(
                    model_constructor.zen_model.sets["set_conversion_technologies"],
                    model_constructor.zen_model.sets[
                        "set_output_carriers"
                    ].coordinate_values,
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

        model_constructor.zen_model.add_constraint(
            "constraint_carrier_conversion", constraints
        )
