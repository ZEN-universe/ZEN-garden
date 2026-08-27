"""Dependency-ordered parameter loading across the model schema."""

from zen_garden.elements import ELEMENT_TYPE_CLASSES
from zen_garden.elements.energy_system import EnergySystem
from zen_garden.topology.generic_parameter import GenericParameter
from zen_garden.topology.model_schema import ModelSchema


class ParameterLoadingService:
    """Prepare elements and load their parameters in dependency order."""

    def __init__(self, model_schema: ModelSchema):
        """Initialize the service for a fully registered model schema."""
        self.model_schema = model_schema

    def load_parameters(self) -> None:
        """Prepare, load, and finalize every element in the schema."""
        for element in self.model_schema.elements:
            element.prepare_input_data()

        for parameter in self._parameter_order():
            for element in self.model_schema.elements:
                if parameter in element.parameters:
                    parameter.store_input_data(element)

        for element in self.model_schema.elements:
            element.finalize_input_data()

    @staticmethod
    def _parameter_order() -> list[type[GenericParameter]]:
        """Topologically order every parameter declaration in the schema."""
        parameters: list[type[GenericParameter]] = list(EnergySystem.parameters)
        for element_class in ELEMENT_TYPE_CLASSES.values():
            parameters.extend(element_class.parameters)
        return GenericParameter.construction_order(parameters)
