"""Constructor for the TransportTechnology elements."""

import logging

from typing_extensions import override

from zen_garden.elements.model_constructor import ModelConstructor
from zen_garden.elements.transport_technology import TransportTechnology

logger = logging.getLogger(__name__)


class TransportTechnologyConstructor(ModelConstructor):
    element_class = TransportTechnology

    @override
    def construct_expressions(self):
        """Construct reusable transport coefficients."""
        parameters = self.zen_model.parameters
        transport_technologies = self.zen_model.sets["set_transport_technologies"]

        self.zen_model.add_expression(
            "transport_capex_distance",
            parameters.distance * parameters.capex_per_distance_transport,
        )
        self.zen_model.add_expression(
            "transport_loss_factor_effective",
            parameters.transport_loss_factor,
        )
        self.zen_model.add_expression(
            "transport_carbon_intensity_effective",
            parameters.carbon_intensity_technology.sel(
                set_technologies=transport_technologies
            ),
        )
