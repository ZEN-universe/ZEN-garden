import numpy as np

from zen_garden.elements.energy_system.sets.set_edges import SetEdges
from zen_garden.elements.technology.masks import PowerCapacityAdditionMask
from zen_garden.elements.technology.sets.set_transport_technologies import (
    SetTransportTechnologies,
)
from zen_garden.elements.technology.variables.cost_capex_overnight import (
    CostCapexOvernight,
)
from zen_garden.elements.transport_technology.expressions.transport_capacity_on_unavailable_edges import (  # noqa: E501
    TransportCapacityOnUnavailableEdges,
)
from zen_garden.elements.transport_technology.expressions.transport_overnight_capex import (  # noqa: E501
    TransportOvernightCapex,
)
from zen_garden.elements.transport_technology.parameters.distance import Distance
from zen_garden.model.component_types.constraint import GenericConstraint


class TransportTechnologyCapexConstraint(GenericConstraint):
    """Set transport CAPEX and prohibit additions on unavailable edges."""

    name = "constraint_transport_technology_capex"
    indices = ("set_technologies", "set_location", "set_years")
    masks = (PowerCapacityAdditionMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetTransportTechnologies])

    @classmethod
    def get_constraint(cls, model_constructor):
        """Construct set transport capex and prohibit additions on unavailable edges.

        Summary:
            Set transport CAPEX and prohibit additions on unavailable edges.

        Formulation:

        .. math::

            C^{capex,overnight}=k^{capex,trans}\\Delta K+k^{distance}G

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
        cost = model.variables[CostCapexOvernight].sel(
            set_technologies=technologies,
            set_location=edges,
        )
        available = ~np.isinf(
            model.parameters[Distance].rename(
                set_transport_technologies="set_technologies",
                set_edges="set_location",
            )
        )
        return (
            cost.where(available)
            - model.expressions[TransportOvernightCapex]
            + model.expressions[TransportCapacityOnUnavailableEdges]
            == 0
        )
