import pandas as pd

from zen_garden.elements.energy_system.parameters.market_share_unbounded import (
    MarketShareUnbounded,
)
from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.technology.variables.capacity_previous import CapacityPrevious
from zen_garden.model.component_types.expression import GenericExpression


class CapacityGrowthFromUnboundedMarket(GenericExpression):
    """Capacity growth supported by peers serving the same reference carrier."""

    name = "capacity_growth_from_unbounded_market"
    indices = ("set_technologies", "set_location", "set_years")

    @staticmethod
    def _class_technologies(model_constructor, technology):
        """Return technologies belonging to the same concrete element class."""
        model = model_constructor.optimization_model
        element = model_constructor.element_registry.get_element_by_name(technology)
        if element is None:
            raise ValueError(f"Element {technology} not found")
        return model.sets[element.label]

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Capacity growth supported by peers serving the same reference carrier.

        Formulation:

        .. math::

             \\sum_{h\\prime\\ne h:c_{h\\prime}^{ref}
            =c_h^{ref}}s^{market}K^{previous}_{h\\prime,q,p,y}

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
        capacity_previous = model.variables[CapacityPrevious]
        shares = {
            (technology, other): (
                model.parameters[MarketShareUnbounded]
                if other != technology
                and model.sets.get_related(
                    "set_technologies", technology, "set_reference_carriers"
                )[0]
                == model.sets.get_related(
                    "set_technologies", other, "set_reference_carriers"
                )[0]
                else 0
            )
            for technology in model.sets[SetTechnologies]
            for other in cls._class_technologies(model_constructor, technology)
        }
        shares = pd.Series(shares)
        shares.index.names = ["set_technologies", "set_other_technologies"]
        shares = shares.to_xarray().broadcast_like(capacity_previous.lower).fillna(0)
        return (
            (
                shares
                * capacity_previous.rename(
                    {"set_technologies": "set_other_technologies"}
                )
            )
            .where(shares != 0)
            .sum("set_other_technologies")
        )
