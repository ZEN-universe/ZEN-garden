from zen_garden.elements.energy_system.sets.set_edges import SetEdges
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.parameters.opex_specific_variable import (
    OpexSpecificVariable,
)
from zen_garden.elements.technology.sets.set_transport_technologies import (
    SetTransportTechnologies,
)
from zen_garden.elements.technology.variables.cost_opex_variable import CostOpexVariable
from zen_garden.elements.transport_technology.masks import FiniteTransportRouteMask
from zen_garden.elements.transport_technology.variables.flow_transport import (
    FlowTransport,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class OpexTechnologyTransportConstraint(GenericConstraint):
    """Calculate variable OPEX from transported flow."""

    name = "constraint_opex_technology_transport"
    indices = (
        "set_transport_technologies",
        "set_edges",
        "set_years",
        "set_representative_hours",
    )
    masks = (PowerOperationMask, FiniteTransportRouteMask)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetTransportTechnologies])

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct calculate variable opex from transported flow.

        Summary:
            Calculate variable OPEX from transported flow.

        Formulation:

        .. math::

            C^{opex,var}=k^{opex,var}F^{trans}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
        """
        model = model_constructor.optimization_model
        technologies = model.sets[SetTransportTechnologies]
        edges = model.sets[SetEdges]
        cost = (
            model.variables[CostOpexVariable]
            .sel(set_technologies=technologies, set_location=edges)
            .rename(
                set_technologies="set_transport_technologies", set_location="set_edges"
            )
        )
        factor = (
            model.parameters[OpexSpecificVariable]
            .sel(set_technologies=technologies, set_location=edges)
            .rename(
                set_technologies="set_transport_technologies", set_location="set_edges"
            )
        )
        return cost == factor * model.variables[FlowTransport]
