"""Rules for the RetrofittingTechnology class."""

import itertools
import logging

import pandas as pd

from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.utils import align_like

logger = logging.getLogger(__name__)


class RetrofitFlowCouplingConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Limit retrofit reference flow by the associated base-technology flow.

        Formulation:

        .. math::
            F^{\\mathrm{ref}}_{h^{\\mathrm{retro}},n,t} \\leq
            \\eta^{\\mathrm{retro}}_{h^{\\mathrm{retro}},n,t}
            F^{\\mathrm{ref}}_{h^{\\mathrm{base}},n,t}

        Notation:

        :math:`F^{\\mathrm{ref}}_{h^{\\mathrm{retro}},n,t}`: reference flow of retrofit
        technology :math:`h^{\\mathrm{retro}}`
        :math:`F^{\\mathrm{ref}}_{h^{\\mathrm{base}},n,t}`: reference flow of its
        associated base
        technology :math:`h^{\\mathrm{base}}`
        :math:`\\eta^{\\mathrm{retro}}_{h^{\\mathrm{retro}},n,t}`: retrofit
        flow-coupling factor.
        Reference flow is selected from the input or output flow according to each
        technology's configured reference carrier.
        """
        flow_conversion_input = model_constructor.zen_model.variables[
            "flow_conversion_input"
        ]
        flow_conversion_output = model_constructor.zen_model.variables[
            "flow_conversion_output"
        ]
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
        retrofit_base_technologies = pd.Series(
            {
                t: rt
                for t in model_constructor.zen_model.sets["set_conversion_technologies"]
                if t
                in model_constructor.zen_model.sets[
                    "set_retrofitting_base_technologies"
                ]
                for rt in model_constructor.zen_model.sets[
                    "set_retrofitting_base_technologies"
                ][t]
            },
            name="set_conversion_technologies",
        )
        retrofit_base_technologies.index.name = "set_conversion_technologies"
        retrofit_flow_coupling = (
            model_constructor.zen_model.parameters.retrofit_flow_coupling_factor.rename(
                {"set_retrofitting_technologies": "set_conversion_technologies"}
            )
        )
        term_flow_retrofit = cls.map_and_expand(
            term_flow_reference, retrofit_base_technologies
        )
        term_flow_base = term_flow_reference.sel(
            {
                "set_conversion_technologies": model_constructor.zen_model.sets[
                    "set_retrofitting_technologies"
                ]
            }
        )
        lhs = term_flow_base - retrofit_flow_coupling * term_flow_retrofit
        rhs = 0
        constraints = lhs <= rhs

        model_constructor.zen_model.add_constraint(
            "constraint_retrofit_flow_coupling", constraints
        )
