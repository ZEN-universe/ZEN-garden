"""Abstract class defining a standard Element."""

from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from zen_garden.preprocess.data_input import DataInput

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.model.config import Config
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.dataset_path_resolver import DatasetPathResolver
    from zen_garden.services.element_registry import ElementRegistry
    from zen_garden.services.scenario_dict import ScenarioDict
    from zen_garden.types import YearSpecificTs
    from zen_garden.utils.input_data_checks import InputDataChecks


class Element:
    """Class defining a standard Element."""

    # set label
    label: str = "set_elements"
    raw_time_series: dict[str, pd.Series | pd.DataFrame | None]

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
        self.data_input = DataInput(
            element=self,
            energy_system=self.energy_system,
            unit_handling=self.unit_handling,
            config=self.config,
            scenario_dict=scenario_dict,
            input_data_checks=self.input_data_checks,
            year_specific_ts=year_specific_ts,
            folder_path=self._get_input_path(),
        )
        # dict to save the parameter units element-wise and to save them in the results
        self.units = {}
        self.raw_time_series = {}

        self._initialize()

    def _initialize(self):
        """Initialize the element."""
        pass

    @abstractmethod
    def store_input_data(self):
        """Retrieves and stores input data for element as attributes. Each Child class
        overwrites method to store different attributes.
        """
        pass

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
