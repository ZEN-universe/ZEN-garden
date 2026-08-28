"""Configuration-only blueprint for a ZEN-garden optimization model."""

from collections import defaultdict
from typing import Any, TypeVar, cast

import numpy as np

from zen_garden.config import Config
from zen_garden.elements import ELEMENT_TYPE_CLASSES
from zen_garden.elements.energy_system import EnergySystem
from zen_garden.model.element import Element

T = TypeVar("T", bound=Element)


class ModelSchema:
    """Describe the complete element and index structure of a model."""

    def __init__(self, config: Config):
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
        self._elements: defaultdict[str, list[Element]] = defaultdict(list)

    def register_element(self, element: Element) -> None:
        """Register an element under every element type in its inheritance tree."""
        for element_class in type(element).__mro__:
            if issubclass(element_class, Element):
                self._elements[element_class.__name__].append(element)

    def all_elements(self) -> list[Element]:
        """Return all registered model elements."""
        return list(self._elements[Element.__name__])

    def all_elements_of_type(self, element_class: type[T]) -> list[T]:
        """Return registered elements of a particular type."""
        return cast(list[T], self._elements[element_class.__name__])

    def all_elements_by_type_name(self, type_name: str) -> list[Element]:
        """Return registered elements of a type, addressed by its class name.

        Lets callers that must not import an element class (to avoid an import
        cycle) still filter by type.
        """
        return list(self._elements[type_name])

    def get_element(self, element_class: type[T], name: str) -> T | None:
        """Return one named element of a particular type."""
        return next(
            (
                element
                for element in self.all_elements_of_type(element_class)
                if element.name == name
            ),
            None,
        )

    @property
    def energy_system(self) -> EnergySystem:
        """Return the singleton energy-system element."""
        energy_systems = self.all_elements_of_type(EnergySystem)
        if len(energy_systems) != 1:
            raise RuntimeError("ModelSchema requires exactly one EnergySystem")
        return energy_systems[0]

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
