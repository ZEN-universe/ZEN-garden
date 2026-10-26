"""Transport outflow by carrier and node."""

import linopy as lp
import numpy as np
import xarray as xr

from zen_garden.elements.carrier.parameters.demand import Demand
from zen_garden.elements.carrier.variables.flow_import import FlowImport
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.technology.sets.set_transport_technologies import (
    SetTransportTechnologies,
)
from zen_garden.elements.transport_technology.variables.flow_transport import (
    FlowTransport,
)
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.registries.multi_index_helper import MultiIndexHelper


class NodalTransportOutflow(GenericExpression):
    """Aggregate outgoing transport flow."""

    name = "nodal_transport_outflow"
    indices = (
        "set_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    )

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct aggregate outgoing transport flow.

        Summary:
            Aggregate outgoing transport flow.

        Formulation:

        .. math::

            \\sum_{h,e\\in\\delta^+(n)}F^{trans}_{h,e,y,r}

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
        if optimization_model.variables[FlowTransport].size == 0:
            return optimization_model.variables[FlowImport].where(False).to_linexpr()

        index_values, index_names = optimization_model.create_custom_set(
            ["set_carriers", "set_nodes", "set_years", "set_representative_hours"]
        )
        index = MultiIndexHelper(index_values, index_names)
        connected_edges = {
            node: model_constructor.network_topology.calculate_connected_edges(
                node, "out"
            )
            for node in optimization_model.sets[SetNodes]
        }
        max_edges = max(len(edges) for edges in connected_edges.values())
        demand = optimization_model.parameters[Demand]
        coords = [
            demand.coords["set_carriers"],
            demand.coords["set_nodes"],
            demand.coords["set_years"],
            demand.coords["set_representative_hours"],
            xr.DataArray(
                np.arange(
                    len(optimization_model.sets[SetTransportTechnologies])
                    * (2 * max_edges + 1)
                ),
                dims=["_term"],
            ),
        ]
        shape = tuple(len(coord) for coord in coords)
        variables = np.full(shape, -1, dtype=int)
        coefficients = np.full(shape, np.nan, dtype=float)
        carrier_positions = {value: pos for pos, value in enumerate(coords[0].values)}
        node_positions = {value: pos for pos, value in enumerate(coords[1].values)}

        for carrier, node in index.get_unique([0, 1]):
            technologies = [
                technology
                for technology in optimization_model.sets[SetTransportTechnologies]
                if carrier
                in optimization_model.sets.get_related(
                    "set_technologies", technology, "set_reference_carriers"
                )
            ]
            edges = connected_edges[node]
            flow_labels = (
                optimization_model.variables[FlowTransport]
                .labels.loc[technologies, edges, ...]
                .data
            )
            flow_labels = flow_labels.reshape((-1, *flow_labels.shape[-2:])).transpose(
                1, 2, 0
            )
            carrier_pos = carrier_positions[carrier]
            node_pos = node_positions[node]
            count = flow_labels.shape[-1]
            variables[carrier_pos, node_pos, ..., :count] = flow_labels
            coefficients[carrier_pos, node_pos, ..., :count] = 1

        return lp.LinearExpression(
            xr.Dataset(
                {
                    "coeffs": xr.DataArray(coefficients, coords=coords),
                    "vars": xr.DataArray(variables, coords=coords),
                }
            ),
            optimization_model.lp_model,
        )
