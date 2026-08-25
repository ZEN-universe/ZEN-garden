"""Abstract class defining a standard Element."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from zen_garden.preprocess.data_input import DataInput
from zen_garden.services.input_repository import InputRepository

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.model.config import Config
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.dataset_path_resolver import DatasetPathResolver
    from zen_garden.services.element_registry import ElementRegistry
    from zen_garden.services.scenario_dict import ScenarioDict
    from zen_garden.topology.generic_parameter import (
        GenericComputedParameters,
        GenericParameter,
    )
    from zen_garden.types import YearSpecificTs
    from zen_garden.utils.input_data_checks import InputDataChecks


class Element:
    """Class defining a standard Element."""

    # set label
    name: str = "Element"
    label: str = "set_elements"
    raw_time_series: dict[str, pd.Series | pd.DataFrame | None]
    own_parameters: ClassVar[list[type["GenericParameter"]]] = []
    parameters: ClassVar[list[type["GenericParameter"]]] = []

    def __init_subclass__(cls, **kwargs):
        """Compose parameter declarations inherited from element base classes."""
        super().__init_subclass__(**kwargs)
        inherited: list[type["GenericParameter"]] = []
        for base in cls.__bases__:
            inherited.extend(getattr(base, "parameters", ()))
        own = cls.__dict__.get("own_parameters", ())
        cls.parameters = list(dict.fromkeys([*inherited, *own]))

    def __init__(
        self,
        element_name: str,
        config: "Config",
        energy_system: "EnergySystem",
        element_registry: "ElementRegistry",
        unit_handling: "UnitHandling",
        dataset_path_resolver: "DatasetPathResolver",
        scenario_dict: "ScenarioDict",
        input_data_checks: "InputDataChecks",
        time_steps: "TimeStepsDicts",
        year_specific_ts: "YearSpecificTs",
    ):
        """Initialization of an element.

        :param element_name: Name of the element
        :param config: Config object
        :param energy_system: EnergySystem object
        :param element_registry: ElementRegistry object
        :param unit_handling: UnitHandling object
        :param dataset_path_resolver: DatasetPathResolver object
        :param scenario_dict: ScenarioDict object
        :param input_data_checks: InputDataChecks object
        """
        # set attributes
        self.name = element_name
        # optimization setup
        self.config = config
        # energy system
        self.energy_system = energy_system
        self.element_registry = element_registry
        self.unit_handling = unit_handling
        self.dataset_path_resolver = dataset_path_resolver
        self.input_data_checks = input_data_checks
        self.time_steps = time_steps
        # set if aggregated
        self.aggregated = False
        # create DataInput object
        folder_path = self._get_input_path()
        self.input_repository = InputRepository(folder_path)
        self.data_input = DataInput(
            element=self,
            energy_system=self.energy_system,
            unit_handling=self.unit_handling,
            config=self.config,
            scenario_dict=scenario_dict,
            input_data_checks=self.input_data_checks,
            year_specific_ts=year_specific_ts,
            folder_path=folder_path,
            input_repository=self.input_repository,
        )
        # dict to save the parameter units element-wise and to save them in the results
        self.units: dict[str, Any] = {}
        self.raw_time_series = {}

        self._initialize()

    def _initialize(self):
        """Initialize the element."""
        pass

    def store_input_data(self) -> None:
        """Load all declared parameters through the shared input-loader service."""
        from zen_garden.services.parameter_input_loader import ParameterInputLoader
        from zen_garden.topology.generic_parameter import GenericComputedParameters

        self.prepare_input_data()
        loader = ParameterInputLoader()
        for parameter in self.parameters:
            if issubclass(parameter, GenericComputedParameters):
                continue
            loader.load_into(parameter, self)
        for parameter in self._ordered_computed_parameters():
            loader.load_into(parameter, self)
        self.postprocess_input_data()

    @classmethod
    def _ordered_computed_parameters(
        cls,
    ) -> list[type["GenericComputedParameters"]]:
        """Topologically order computed parameters using their dependency DAG."""
        from zen_garden.topology.generic_parameter import GenericComputedParameters

        return GenericComputedParameters.construction_order(cls.parameters)

    def prepare_input_data(self) -> None:
        """Prepare structural information required to load parameters."""

    def postprocess_input_data(self) -> None:
        """Handle stateful data that must persist outside a model instance."""

    def _get_input_path(self):
        """Get input path where input data is stored input_path."""
        # get technology type
        class_label = self.label
        # check if class is a subset
        if class_label not in self.dataset_path_resolver.all_sets():
            subsets = self.config.analysis.subsets
            # iterate through subsets and check if class belongs to any of the subsets
            for set_name, subsets_list in subsets.items():
                if class_label in subsets_list:
                    class_label = set_name
                    break
        # get input path for current class_label
        return Path(
            self.dataset_path_resolver.folder_of_element(class_label, self.name)
        )
