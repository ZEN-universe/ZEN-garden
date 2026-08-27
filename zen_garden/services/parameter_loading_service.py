"""Global, dependency-ordered parameter loading service."""

import logging
from typing import Any

from zen_garden.elements import ELEMENT_TYPE_CLASSES
from zen_garden.elements.energy_system import EnergySystem
from zen_garden.services.element_registry import ElementRegistry
from zen_garden.topology.generic_parameter import GenericParameter

logger = logging.getLogger(__name__)


class ParameterLoadingService:
    """Prepare targets and load every parameter using one schema-wide DAG."""

    def __init__(self, energy_system: EnergySystem, element_registry: ElementRegistry):
        self.energy_system = energy_system
        self.element_registry = element_registry

    def load_parameters(self) -> None:
        """Load parameters globally in dependency order."""
        targets: list[Any] = [
            self.energy_system,
            *self.element_registry.all_elements(),
        ]
        for target in targets:
            target.prepare_input_data()

        for parameter in self._parameter_order():
            for target in targets:
                if parameter not in target.parameters:
                    continue
                parameter.store_input_data(target)

        for target in targets:
            finalize = getattr(target, "finalize_input_data", None)
            if finalize is not None:
                finalize()

    @staticmethod
    def _parameter_order() -> list[type[GenericParameter]]:
        """Collect the schema parameter classes and topologically order them."""
        parameters = list(EnergySystem.parameters)
        for element_class in ELEMENT_TYPE_CLASSES.values():
            parameters.extend(element_class.parameters)
        return GenericParameter.construction_order(parameters)
