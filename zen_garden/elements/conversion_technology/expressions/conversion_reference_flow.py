"""Reference-carrier flow of conversion technologies."""

import itertools

import pandas as pd

from zen_garden.elements.conversion_technology.sets.set_input_carriers import (
    SetInputCarriers,
)
from zen_garden.elements.conversion_technology.sets.set_output_carriers import (
    SetOutputCarriers,
)
from zen_garden.elements.conversion_technology.variables.flow_conversion_input import (
    FlowConversionInput,
)
from zen_garden.elements.conversion_technology.variables.flow_conversion_output import (
    FlowConversionOutput,
)
from zen_garden.elements.technology.sets.set_conversion_technologies import (
    SetConversionTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.utils import align_like


class ConversionReferenceFlow(GenericExpression):
    """Select each conversion technology's configured reference-carrier flow."""

    name = "conversion_reference_flow"
    indices = (
        "set_conversion_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Select each conversion technology's configured reference-carrier flow.

        Formulation:

        .. math::

             \\sum_{c \\in \\mathcal C_h^{ref} \\cap \\mathcal
            C_h^{in}}F^{in}_{h,c,n,y,r} + \\sum_{c \\in \\mathcal C_h^{ref} \\cap
            \\mathcal C_h^{out}}F^{out}_{h,c,n,y,r}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        optimization_model = model_constructor.optimization_model
        input_flow = optimization_model.variables[FlowConversionInput]
        output_flow = optimization_model.variables[FlowConversionOutput]
        technologies = optimization_model.sets[SetConversionTechnologies]
        input_mask = pd.Series(
            {
                (technology, carrier): carrier
                in optimization_model.sets.get_related(
                    "set_technologies", technology, "set_reference_carriers"
                )
                for technology, carrier in itertools.product(
                    technologies,
                    optimization_model.sets[SetInputCarriers].coordinate_values,
                )
            }
        )
        output_mask = pd.Series(
            {
                (technology, carrier): carrier
                in optimization_model.sets.get_related(
                    "set_technologies", technology, "set_reference_carriers"
                )
                for technology, carrier in itertools.product(
                    technologies,
                    optimization_model.sets[SetOutputCarriers].coordinate_values,
                )
            }
        )
        input_mask.index.names = [
            "set_conversion_technologies",
            "set_input_carriers",
        ]
        output_mask.index.names = [
            "set_conversion_technologies",
            "set_output_carriers",
        ]
        input_mask = align_like(input_mask.to_xarray(), input_flow, astype=bool)
        output_mask = align_like(output_mask.to_xarray(), output_flow, astype=bool)
        return input_flow.where(input_mask).sum(
            "set_input_carriers"
        ) + output_flow.where(output_mask).sum("set_output_carriers")
