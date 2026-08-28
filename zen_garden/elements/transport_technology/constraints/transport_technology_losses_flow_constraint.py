from typing import cast

import numpy as np
import xarray as xr

from zen_garden.model.component_types.constraint import GenericConstraint


class TransportTechnologyLossesFlowConstraint(GenericConstraint):
    @classmethod
    def build(cls, model_constructor):
        """Summary:
        Compute the flow losses for a carrier through a transport technology.

        Formulation:

        .. math::
            \\text{if } d^{\\mathrm{dist}}_{h,e}<\\infty:\\quad
            F^{\\mathrm{loss}}_{h,e,t} = \\lambda^{\\mathrm{loss}}_{h,e}
            F^{\\mathrm{trans}}_{h,e,t}

        For infinite transport distances, this constraint does not restrict the loss
        variable.

        .. math::
            \\lambda^{\\mathrm{loss}}_{h,e} =
            d^{\\mathrm{dist}}_{h,e}\\lambda^{\\mathrm{lin}}_h
            \\quad\\text{or}\\quad
            \\lambda^{\\mathrm{loss}}_{h,e} =
            1-\\exp(-d^{\\mathrm{dist}}_{h,e}\\lambda^{\\mathrm{exp}}_h)

        Notation:

        :math:`F^{\\mathrm{loss}}_{h,e,t}`: flow losses through transport technology
        :math:`h` on edge :math:`e` in time step :math:`t` of year :math:`y`
        :math:`d^{\\mathrm{dist}}_{h,e}`: Transport distance for transport technology
        :math:`h` on
        edge :math:`e`
        :math:`\\lambda^{\\mathrm{loss}}_{h,e}`: effective loss factor,
        calculated during preprocessing
        from either a linear or exponential loss-rate input
        :math:`F^{\\mathrm{trans}}_{h,e,t}`: carrier flow through transport
        technology :math:`h` on edge :math:`e` in time step :math:`t` of year :math:`y`
        """
        optimization_model = model_constructor.optimization_model
        if len(optimization_model.sets["set_transport_technologies"]) == 0:
            return
        flow_transport = optimization_model.variables["flow_transport"]
        flow_transport_loss = optimization_model.variables["flow_transport_loss"]
        # This mask checks the distance between nodes
        distance_isfinite = cast(
            xr.DataArray,
            ~np.isinf(optimization_model.parameters.distance),
        )
        mask = distance_isfinite.broadcast_like(flow_transport.lower)
        loss_factor = (
            optimization_model.parameters.transport_loss_factor.broadcast_like(
                flow_transport.lower
            )
        )
        lhs = (flow_transport_loss - loss_factor * flow_transport).where(mask, 0)
        rhs = 0
        constraints = lhs == rhs

        optimization_model.add_constraint(
            "constraint_transport_technology_losses_flow", constraints
        )
