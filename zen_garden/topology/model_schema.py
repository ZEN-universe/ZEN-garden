"""Configuration-only blueprint for a ZEN-garden optimization model."""

from typing import Any

import numpy as np

from zen_garden.default_config import Config as DefaultConfig
from zen_garden.elements import ELEMENT_TYPE_CLASSES
from zen_garden.elements.element import Element
from zen_garden.elements.energy_system import EnergySystem
from zen_garden.model.config import Config as RuntimeConfig


class ModelSchema:
    """Describe the complete element and index structure of a model."""

    def __init__(self, config: DefaultConfig | RuntimeConfig):
        """Construct a model blueprint using configuration only."""
        self.config = config
        self.element_classes: tuple[type[Element], ...] = (
            EnergySystem,
            *ELEMENT_TYPE_CLASSES.values(),
        )
        self.element_type_classes = dict(ELEMENT_TYPE_CLASSES)
        self.parameters_interpolation_off: dict[str, Any] | None = None
        self.dict_technology_of_carrier: dict[str, list[str]] = {}
        self.set_carriers: list[str] = []
        self._set_hours_all_years: list[int] | None = None
        self._set_years: list[int] | None = None

    @property
    def set_technologies(self) -> list[str]:
        return self.config.system.set_technologies

    @property
    def set_conversion_technologies(self) -> list[str]:
        return self.config.system.set_conversion_technologies

    @property
    def set_transport_technologies(self) -> list[str]:
        return self.config.system.set_transport_technologies

    @property
    def set_storage_technologies(self) -> list[str]:
        return self.config.system.set_storage_technologies

    @property
    def set_retrofitting_technologies(self) -> list[str]:
        return self.config.system.set_retrofitting_technologies

    @property
    def set_hours(self) -> list[int]:
        return list(range(self.config.system.unaggregated_time_steps_per_year))

    @property
    def set_hours_all_years(self) -> list[int]:
        if self._set_hours_all_years is not None:
            return self._set_hours_all_years
        return list(
            range(
                self.config.system.unaggregated_time_steps_per_year
                * self.config.system.optimized_years
            )
        )

    @set_hours_all_years.setter
    def set_hours_all_years(self, value: list[int]) -> None:
        self._set_hours_all_years = value

    @property
    def set_years(self) -> list[int]:
        if self._set_years is not None:
            return self._set_years
        return list(range(self.config.system.optimized_years))

    @set_years.setter
    def set_years(self, value: list[int]) -> None:
        self._set_years = value

    @property
    def set_years_entire_horizon(self) -> list[int]:
        return list(range(self.config.system.optimized_years))

    @property
    def set_time_steps_years(self) -> list[int]:
        system = self.config.system
        return list(
            range(
                system.reference_year,
                system.reference_year
                + system.optimized_years * system.interval_between_years,
                system.interval_between_years,
            )
        )

    @property
    def sequence_time_steps_yearly(self) -> np.ndarray:
        """Map every unaggregated time step to its modeled year."""
        duration = len(self.set_hours_all_years) / len(self.set_years_entire_horizon)
        durations = {year: int(duration) for year in self.set_years_entire_horizon}
        if not duration.is_integer():
            durations[self.set_years_entire_horizon[-1]] = len(
                self.set_hours_all_years
            ) - sum(list(durations.values())[:-1])
        return np.concatenate(
            [[year] * durations[year] for year in self.set_years_entire_horizon]
        )

    def set_technology_of_carrier(self, technology: str, carriers: list[str]) -> None:
        """Record carrier relationships declared by technology elements."""
        for carrier in carriers:
            if carrier not in self.dict_technology_of_carrier:
                self.dict_technology_of_carrier[carrier] = [technology]
                self.set_carriers.append(carrier)
            elif technology not in self.dict_technology_of_carrier[carrier]:
                self.dict_technology_of_carrier[carrier].append(technology)
