"""Energy-system element definition."""

from pathlib import Path
from typing import ClassVar

import pandas as pd

from zen_garden.elements.energy_system.constraints import ENERGY_SYSTEM_CONSTRAINTS
from zen_garden.elements.energy_system.expressions import ENERGY_SYSTEM_EXPRESSIONS
from zen_garden.elements.energy_system.parameters import ENERGY_SYSTEM_PARAMETERS
from zen_garden.elements.energy_system.sets import ENERGY_SYSTEM_SETS
from zen_garden.elements.energy_system.variables import ENERGY_SYSTEM_VARIABLES
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.component_types.parameter import GenericParameter
from zen_garden.model.component_types.set import GenericSet
from zen_garden.model.component_types.variable import GenericVariable
from zen_garden.model.element import Element


class EnergySystem(Element):
    """Class defining a standard energy system."""

    name = "EnergySystem"
    label = "energy_system"
    own_parameters: ClassVar[list[type[GenericParameter]]] = ENERGY_SYSTEM_PARAMETERS
    own_sets: ClassVar[list[type[GenericSet]]] = ENERGY_SYSTEM_SETS
    carbon_emissions_annual_limit: pd.Series
    variables: ClassVar[list[type[GenericVariable]]] = ENERGY_SYSTEM_VARIABLES
    expressions: ClassVar[list[type[GenericExpression]]] = ENERGY_SYSTEM_EXPRESSIONS
    constraints: ClassVar[list[type[GenericConstraint]]] = ENERGY_SYSTEM_CONSTRAINTS
    time_steps_operation_duration: pd.Series | None
    time_steps_storage_duration: pd.Series | None

    def _initialize(self) -> None:
        """Initialize values populated during time-series processing."""
        self.time_steps_operation_duration = None
        self.time_steps_storage_duration = None

    def _get_input_path(self) -> Path:
        """Return the singleton energy-system input folder."""
        return Path(self.dataset_path_resolver.folder_of_set(self.label))

    def finalize_input_data(self) -> None:
        """Scale annual emissions limits to the modeled fraction of a year."""
        fraction_year = (
            self.config.system.unaggregated_time_steps_per_year
            / self.config.system.total_hours_per_year
        )
        self.carbon_emissions_annual_limit *= fraction_year
