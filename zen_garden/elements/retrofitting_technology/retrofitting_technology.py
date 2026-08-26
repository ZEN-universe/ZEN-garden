"""Class defining retrofitting technologies."""

import logging
from typing import ClassVar

from zen_garden.elements.conversion_technology import ConversionTechnology
from zen_garden.elements.retrofitting_technology.parameters import (
    RETROFITTING_TECHNOLOGY_PARAMETERS,
)
from zen_garden.elements.retrofitting_technology.sets import (
    RETROFITTING_TECHNOLOGY_SETS,
)
from zen_garden.topology.generic_parameter import GenericParameter
from zen_garden.topology.generic_set import GenericSet

logger = logging.getLogger(__name__)


class RetrofittingTechnology(ConversionTechnology):
    """Class defining retrofitting technologies."""

    # set label
    label = "set_retrofitting_technologies"
    location_type = "set_nodes"
    own_parameters: ClassVar[list[type[GenericParameter]]] = (
        RETROFITTING_TECHNOLOGY_PARAMETERS
    )
    own_sets: ClassVar[list[type[GenericSet]]] = RETROFITTING_TECHNOLOGY_SETS

    def prepare_input_data(self) -> None:
        """Load the retrofit relationship before generic parameter loading."""
        super().prepare_input_data()
        self.retrofit_base_technology = (
            self.data_input.extract_retrofit_base_technology()
        )
