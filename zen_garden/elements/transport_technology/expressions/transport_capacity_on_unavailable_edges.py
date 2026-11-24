"""Transport capacity proposed on unavailable edges."""

import numpy as np

from zen_garden.elements.energy_system.sets.set_edges import SetEdges
from zen_garden.elements.technology.sets.set_transport_technologies import (
    SetTransportTechnologies,
)
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.elements.transport_technology.parameters.distance import Distance
from zen_garden.model.component_types.expression import GenericExpression


class TransportCapacityOnUnavailableEdges(GenericExpression):
    """Select transport additions whose configured distance is infinite."""

    name = "transport_capacity_on_unavailable_edges"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        """Return whether transport technologies exist in the model."""
        return bool(model_constructor.optimization_model.sets[SetTransportTechnologies])

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct select transport additions whose configured distance is infinite.

        Summary:
            Select transport additions whose configured distance is infinite.

        Formulation:

        .. math::

            \\{\\Delta K_{h,power,e,y}:d_{h,e}=\\infty\\}

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        model = model_constructor.optimization_model
        technologies = model.sets[SetTransportTechnologies]
        edges = model.sets[SetEdges]
        addition = model.variables[CapacityAddition].sel(
            set_technologies=technologies,
            set_location=edges,
        )
        unavailable = np.isinf(
            model.parameters[Distance].rename(
                set_transport_technologies="set_technologies",
                set_edges="set_location",
            )
        )
        return addition.where(unavailable)
