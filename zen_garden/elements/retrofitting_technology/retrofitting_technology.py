"""Class defining retrofitting technologies."""

import logging
from typing import ClassVar

from zen_garden.elements.conversion_technology import ConversionTechnology
from zen_garden.elements.retrofitting_technology.constraints import (
    RETROFITTING_TECHNOLOGY_CONSTRAINTS,
)
from zen_garden.elements.retrofitting_technology.expressions import (
    RETROFITTING_TECHNOLOGY_EXPRESSIONS,
)
from zen_garden.elements.retrofitting_technology.parameters import (
    RETROFITTING_TECHNOLOGY_PARAMETERS,
)
from zen_garden.elements.retrofitting_technology.sets import (
    RETROFITTING_TECHNOLOGY_SETS,
)
from zen_garden.elements.retrofitting_technology.variables import (
    RETROFITTING_TECHNOLOGY_VARIABLES,
)
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.component_types.parameter import GenericParameter
from zen_garden.model.component_types.set import GenericSet
from zen_garden.model.component_types.variable import GenericVariable

logger = logging.getLogger(__name__)


class RetrofittingTechnology(ConversionTechnology):
    """Class defining retrofitting technologies."""

    # set label
    label = "set_retrofitting_technologies"
    location_type = "set_nodes"
    # Optional, self-contained type: only built when retrofitting technologies
    # are configured.
    always_construct: ClassVar[bool] = False
    own_parameters: ClassVar[list[type[GenericParameter]]] = (
        RETROFITTING_TECHNOLOGY_PARAMETERS
    )
    variables: ClassVar[list[type[GenericVariable]]] = RETROFITTING_TECHNOLOGY_VARIABLES
    own_sets: ClassVar[list[type[GenericSet]]] = RETROFITTING_TECHNOLOGY_SETS
    expressions: ClassVar[list[type[GenericExpression]]] = (
        RETROFITTING_TECHNOLOGY_EXPRESSIONS
    )
    constraints: ClassVar[list[type[GenericConstraint]]] = (
        RETROFITTING_TECHNOLOGY_CONSTRAINTS
    )

    def prepare_input_data(self) -> None:
        """Load the retrofit relationship before generic parameter loading."""
        super().prepare_input_data()
        self.retrofit_base_technology = (
            self.element_data_loader.extract_retrofit_base_technology()
        )
