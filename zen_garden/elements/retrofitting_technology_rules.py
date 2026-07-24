"""Rules for the RetrofittingTechnology class."""

import itertools
import logging

import pandas as pd

from zen_garden.elements.generic_rule import GenericRule
from zen_garden.utils import align_like

logger = logging.getLogger(__name__)


class RetrofittingTechnologyRules(GenericRule):
    """Rules for the RetrofittingTechnology class."""

    def constraint_retrofit_flow_coupling(self):
        """Couples reference flow variables based on modeling technique.

        .. math::
            \\text{if reference carrier in input carriers}
            \\underline{G}_{i,n,t}^\\mathrm{r} = G^\\mathrm{d,approximation}_{i,n,t}
        .. math::
            \\text{if reference carrier in output carriers}
            \\overline{G}_{i,n,t}^\\mathrm{r} = G^\\mathrm{d,approximation}_{i,n,t}

        """
        flow_conversion_input = self.zen_model.variables["flow_conversion_input"]
        flow_conversion_output = self.zen_model.variables["flow_conversion_output"]
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
        retrofit_base_technologies = pd.Series(
            {
                t: rt
                for t in self.zen_model.sets["set_conversion_technologies"]
                if t in self.zen_model.sets["set_retrofitting_base_technologies"]
                for rt in self.zen_model.sets["set_retrofitting_base_technologies"][t]
            },
            name="set_conversion_technologies",
        )
        retrofit_base_technologies.index.name = "set_conversion_technologies"
        retrofit_flow_coupling = (
            self.zen_model.parameters.retrofit_flow_coupling_factor.rename(
                {"set_retrofitting_technologies": "set_conversion_technologies"}
            )
        )
        term_flow_retrofit = self.map_and_expand(
            term_flow_reference, retrofit_base_technologies
        )
        term_flow_base = term_flow_reference.sel(
            {
                "set_conversion_technologies": self.zen_model.sets[
                    "set_retrofitting_technologies"
                ]
            }
        )
        lhs = term_flow_base - retrofit_flow_coupling * term_flow_retrofit
        rhs = 0
        constraints = lhs <= rhs

        self.zen_model.add_constraint("constraint_retrofit_flow_coupling", constraints)
