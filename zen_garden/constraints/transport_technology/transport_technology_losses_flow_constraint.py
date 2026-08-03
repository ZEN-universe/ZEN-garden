from typing import cast

import numpy as np
import xarray as xr

from zen_garden.constraints.generic_constraint import GenericConstraint


class TransportTechnologyLossesFlowConstraint(GenericConstraint):
    def build(self):
        """Compute the flow losses for a carrier through a transport technology.

        .. math::
            \\text{if transport distance set to inf: } F^\\mathrm{l}_{j,e,t} = 0
        .. math::
            \\text{else: } F^\\mathrm{l}_{j,e,t} = h_{j,e} \\rho_{j} F_{j,e,t}

        :math:`F^\\mathrm{l}_{j,e,t}`: Flow losses of carrier through transport
        technology :math:`j` on edge :math:`e` at time :math:`t` \n
        :math:`h_{j,e}`: Transport distance for transport technology :math:`j` on
        edge :math:`e` \n
        :math:`\\rho_{j}`: Loss factor for transport technology :math:`j` \n
        :math:`F_{j,e,t}`: Reference flow of carrier through transport
        technology :math:`j` on edge :math:`e` at time :math:`t`

        """
        if len(self.zen_model.sets["set_transport_technologies"]) == 0:
            return
        flow_transport = self.zen_model.variables["flow_transport"]
        flow_transport_loss = self.zen_model.variables["flow_transport_loss"]
        # This mask checks the distance between nodes
        distance_isfinite = cast(
            xr.DataArray, ~np.isinf(self.zen_model.parameters.distance)
        )
        mask = distance_isfinite.broadcast_like(flow_transport.lower)
        loss_factor = self.zen_model.parameters.transport_loss_factor.broadcast_like(
            flow_transport.lower
        )
        lhs = (flow_transport_loss - loss_factor * flow_transport).where(mask, 0)
        rhs = 0
        constraints = lhs == rhs

        self.zen_model.add_constraint(
            "constraint_transport_technology_losses_flow", constraints
        )
