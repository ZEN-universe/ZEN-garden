"""Dependency-ordered parameter loading across the model schema."""

from zen_garden.elements.energy_system import EnergySystem
from zen_garden.model.component_types.parameter import GenericParameter
from zen_garden.model.schema import ModelSchema


class DataLoadingService:
    """Prepare elements and load their parameters in dependency order."""

    def __init__(self, model_schema: ModelSchema):
        """Initialize the service for a fully registered model schema."""
        self.model_schema = model_schema

    def load_parameters(self) -> None:
        """Prepare, load, and finalize every element in the schema."""
        elements = self.model_schema.all_elements()
        self.model_schema.parameters_interpolation_off = (
            self.model_schema.energy_system.attribute_data_loader.read_mapping_file(
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
