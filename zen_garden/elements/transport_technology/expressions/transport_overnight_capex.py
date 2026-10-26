"""Overnight CAPEX of transport-capacity additions."""

import numpy as np

from zen_garden.elements.energy_system.sets.set_edges import SetEdges
from zen_garden.elements.technology.sets.set_transport_technologies import (
    SetTransportTechnologies,
)
from zen_garden.elements.technology.variables.capacity_addition import CapacityAddition
from zen_garden.elements.technology.variables.technology_installation import (
    TechnologyInstallation,
)
from zen_garden.elements.transport_technology.parameters.capex_specific_transport import (  # noqa: E501
    CapexSpecificTransport,
)
from zen_garden.elements.transport_technology.parameters.distance import Distance
from zen_garden.elements.transport_technology.parameters.transport_capex_distance import (  # noqa: E501
    TransportCapexDistance,
)
from zen_garden.model.component_types.expression import GenericExpression


class TransportOvernightCapex(GenericExpression):
    """Combine fixed and distance-dependent transport investment costs."""

    name = "transport_overnight_capex"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        """Return whether transport technologies exist in the model."""
        return bool(model_constructor.optimization_model.sets[SetTransportTechnologies])

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct combine fixed and distance-dependent transport investment costs.

        Summary:
            Combine fixed and distance-dependent transport investment costs.

        Formulation:

        .. math::

            k^{capex,trans}_{h,e,y} \\Delta K_{h,power,e,y}
            +k^{distance}_{h,e,y}G_{h,power,e,y}

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
        installation = model.variables[TechnologyInstallation].sel(
            set_technologies=technologies,
            set_location=edges,
        )
        fixed = model.parameters[CapexSpecificTransport].rename(
            set_transport_technologies="set_technologies", set_edges="set_location"
        )
        distance_cost = model.parameters[TransportCapexDistance].rename(
            set_transport_technologies="set_technologies", set_edges="set_location"
        )
        available = ~np.isinf(
            model.parameters[Distance].rename(
                set_transport_technologies="set_technologies",
                set_edges="set_location",
            )
        )
        return (fixed * addition + distance_cost * installation).where(available)
