"""Structural masks for transport technology model components."""

from zen_garden.model.component_types.mask import GenericMask

from .finite_transport_route_mask import FiniteTransportRouteMask

TRANSPORT_TECHNOLOGY_MASKS: list[type[GenericMask]] = [FiniteTransportRouteMask]

__all__ = ["FiniteTransportRouteMask", "TRANSPORT_TECHNOLOGY_MASKS"]
