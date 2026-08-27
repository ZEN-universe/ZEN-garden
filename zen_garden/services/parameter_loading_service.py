"""Dependency-ordered parameter loading across the model schema."""

from zen_garden.elements.energy_system import EnergySystem
from zen_garden.services.element_registry import ElementRegistry
from zen_garden.topology.generic_parameter import GenericParameter
from zen_garden.topology.model_schema import ModelSchema


class ParameterLoadingService:
    """Prepare elements and load their parameters in dependency order."""

    def __init__(
        self, model_schema: ModelSchema, element_registry: ElementRegistry
    ):
        """Initialize the service for a fully registered model schema."""
        self.model_schema = model_schema
        self.element_registry = element_registry

    def load_parameters(self) -> None:
        """Prepare, load, and finalize every element in the schema."""
        elements = self.element_registry.all_elements()
        energy_systems = self.element_registry.all_elements_of_type(EnergySystem)
        assert len(energy_systems) == 1
        self.model_schema.parameters_interpolation_off = (
            energy_systems[0].input_repository.read_mapping_file(
                "parameters_interpolation_off"
            )
        )
        for element in elements:
            element.prepare_input_data()

        for parameter in self._parameter_order():
            for element in elements:
                if parameter in element.parameters:
                    parameter.store_input_data(element)

        for element in elements:
            element.finalize_input_data()

    def _parameter_order(self) -> list[type[GenericParameter]]:
        """Topologically order every parameter declaration in the schema."""
        parameters: list[type[GenericParameter]] = list(EnergySystem.parameters)
        for element_class in self.model_schema.element_classes:
            parameters.extend(element_class.parameters)
        return GenericParameter.construction_order(parameters)
