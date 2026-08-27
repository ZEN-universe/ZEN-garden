"""Global model schema and orchestration context."""

import copy
from typing import TYPE_CHECKING, Any

import numpy as np

from zen_garden.elements.energy_system import EnergySystem
from zen_garden.services.network_topology import NetworkTopology

if TYPE_CHECKING:
    from zen_garden.elements.element import Element
    from zen_garden.model.config import Config
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.element_registry import ElementRegistry
    from zen_garden.services.input_repository import InputRepository
    from zen_garden.types import YearSpecificTs
    from zen_garden.utils.input_data_checks import InputDataChecks


class ModelSchema:
    """Own global model structure and coordinate all model elements."""

    def __init__(
        self,
        config: "Config",
        unit_handling: "UnitHandling",
        input_data_checks: "InputDataChecks",
        time_steps: "TimeStepsDicts",
        year_specific_ts: "YearSpecificTs",
        input_repository: "InputRepository",
    ):
        """Initialize global model state before any elements are created."""
        self.config = config
        self.unit_handling = unit_handling
        self.input_data_checks = input_data_checks
        self.time_steps = time_steps
        self.year_specific_ts = year_specific_ts
        self.input_repository = input_repository
        self.element_registry: ElementRegistry | None = None
        self.dict_technology_of_carrier: dict[str, list[str]] = {}
        self.set_carriers: list[str] = []
        self.parameters_interpolation_off: dict[str, Any] | None = None
        self.sequence_time_steps_yearly: np.ndarray
        self.set_conversion_technologies: list[str]
        self.set_hours: list[int]
        self.set_hours_all_years: list[int]
        self.set_retrofitting_technologies: list[str]
        self.set_storage_technologies: list[str]
        self.set_technologies: list[str]
        self.set_time_steps_years: list[int]
        self.set_transport_technologies: list[str]
        self.set_years: list[int]
        self.set_years_entire_horizon: list[int]
        self.network_topology = NetworkTopology(
            config=config,
            input_repository=input_repository,
            input_data_checks=input_data_checks,
            unit_handling=unit_handling,
        )
        self._prepare_structure()

    def register_element_registry(self, element_registry: "ElementRegistry") -> None:
        """Attach the registry after it has been initialized."""
        self.element_registry = element_registry

    @property
    def elements(self) -> list["Element"]:
        """Return every element participating in the model schema."""
        if self.element_registry is None:
            return []
        return self.element_registry.all_elements()

    @property
    def energy_system(self) -> EnergySystem:
        """Return the singleton energy-system element."""
        if self.element_registry is None:
            raise RuntimeError("The element registry has not been attached")
        energy_systems = self.element_registry.all_elements_of_type(EnergySystem)
        if len(energy_systems) != 1:
            raise RuntimeError("The model schema requires one EnergySystem element")
        return energy_systems[0]

    def set_technology_of_carrier(self, technology: str, carriers: list[str]) -> None:
        """Associate a technology with its input and output carriers."""
        for carrier in carriers:
            if carrier not in self.dict_technology_of_carrier:
                self.dict_technology_of_carrier[carrier] = [technology]
                self.set_carriers.append(carrier)
            elif technology not in self.dict_technology_of_carrier[carrier]:
                self.dict_technology_of_carrier[carrier].append(technology)

    def calculate_connected_edges(self, *args: Any):
        """Calculate connected edges using the global network topology."""
        return self.network_topology.calculate_connected_edges(*args)

    @property
    def set_nodes(self):
        """Return nodes from the global network topology."""
        return self.network_topology.set_nodes

    @property
    def set_edges(self):
        """Return edges from the global network topology."""
        return self.network_topology.set_edges

    @property
    def set_nodes_on_edges(self):
        """Return nodes on edges from the global network topology."""
        return self.network_topology.set_nodes_on_edges

    @property
    def set_haversine_distances_edges(self):
        """Return edge distances from the global network topology."""
        return self.network_topology.set_haversine_distances_edges

    def _prepare_structure(self) -> None:
        """Initialize global time and technology sets from configuration."""
        system = self.config.system
        self.set_technologies = system.set_technologies
        self.set_hours_all_years = list(
            range(system.unaggregated_time_steps_per_year * system.optimized_years)
        )
        self.set_hours = list(range(system.unaggregated_time_steps_per_year))
        self.set_years = list(range(system.optimized_years))
        self.set_years_entire_horizon = copy.deepcopy(self.set_years)
        yearly_duration = self.time_steps.calculate_time_step_duration(
            self.set_years, self.set_hours_all_years
        )
        self.sequence_time_steps_yearly = np.concatenate(
            [[step] * yearly_duration[step] for step in yearly_duration]
        )
        self.time_steps.sequence_time_steps_yearly = self.sequence_time_steps_yearly
        self.set_time_steps_years = list(
            range(
                system.reference_year,
                system.reference_year
                + system.optimized_years * system.interval_between_years,
                system.interval_between_years,
            )
        )
        self.parameters_interpolation_off = self.input_repository.read_mapping_file(
            "parameters_interpolation_off"
        )
        self.set_conversion_technologies = system.set_conversion_technologies
        self.set_transport_technologies = system.set_transport_technologies
        self.set_storage_technologies = system.set_storage_technologies
        self.set_retrofitting_technologies = system.set_retrofitting_technologies
