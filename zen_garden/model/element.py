"""Abstract class defining a standard Element."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd

from zen_garden.input.attribute_data_loader import AttributeDataLoader
from zen_garden.input.element_data_loader import ElementDataLoader
from zen_garden.model.component_types.parameter import GenericParameter

if TYPE_CHECKING:
    from zen_garden.input.dataset_path_resolver import DatasetPathResolver
    from zen_garden.input.input_data_checks import InputDataChecks
    from zen_garden.input.network_topology import NetworkTopology
    from zen_garden.input.scenario_dict import ScenarioDict
    from zen_garden.input.unit_converter import UnitConverter
    from zen_garden.model.component_types.set import GenericSet
    from zen_garden.model.element_registry import ElementRegistry
    from zen_garden.model.schema import ModelSchema
    from zen_garden.types import YearSpecificTs


class Element:
    """Class defining a standard Element."""

    # set label
    name: str = "Element"
    label: str = "set_elements"
    raw_time_series: dict[str, pd.Series | pd.DataFrame | None]
    own_parameters: ClassVar[list[type["GenericParameter"]]] = []
    parameters: ClassVar[list[type["GenericParameter"]]] = []
    own_sets: ClassVar[list[type["GenericSet"]]] = []
    sets: ClassVar[list[type["GenericSet"]]] = []
    # If False, this type's model components are only built when at least one
    # element of it is configured (see ModelConstructor.has_elements). Mandatory
    # types are referenced unconditionally by others (e.g. the carrier balance).
    always_construct: ClassVar[bool] = True

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    def __init_subclass__(cls, **kwargs):
        """Compose parameter declarations inherited from element base classes."""
        super().__init_subclass__(**kwargs)
        inherited: list[type["GenericParameter"]] = []
        for base in cls.__bases__:
            inherited.extend(getattr(base, "parameters", ()))
        own = cls.__dict__.get("own_parameters", ())
        cls.parameters = list(dict.fromkeys([*inherited, *own]))

        inherited_sets: list[type["GenericSet"]] = []
        for base in cls.__bases__:
            inherited_sets.extend(getattr(base, "sets", ()))
        own_sets = cls.__dict__.get("own_sets", ())
        cls.sets = list(dict.fromkeys([*inherited_sets, *own_sets]))

    def __init__(
        self,
        element_name: str,
        model_schema: "ModelSchema",
        network_topology: "NetworkTopology",
        element_registry: "ElementRegistry",
        unit_converter: "UnitConverter",
        dataset_path_resolver: "DatasetPathResolver",
        scenario_dict: "ScenarioDict",
        input_data_checks: "InputDataChecks",
        year_specific_ts: "YearSpecificTs",
    ):
        """Initialization of an element.

        :param element_name: Name of the element
        :param model_schema: Global model schema
        :param element_registry: ElementRegistry object
        :param unit_converter: UnitConverter object
        :param dataset_path_resolver: DatasetPathResolver object
        :param scenario_dict: ScenarioDict object
        :param input_data_checks: InputDataChecks object
        """
        # set attributes
        self.name = element_name
        self.model_schema = model_schema
        self.network_topology = network_topology
        self.element_registry = element_registry
        self.unit_converter = unit_converter
        self.dataset_path_resolver = dataset_path_resolver
        self.input_data_checks = input_data_checks
        # set if aggregated
        self.aggregated = False
        # create ElementDataLoader object
        folder_path = self._get_input_path()
        self.attribute_data_loader = AttributeDataLoader(folder_path)
        self.element_data_loader = ElementDataLoader(
            element=self,
            model_schema=self.model_schema,
            network_topology=self.network_topology,
            unit_converter=self.unit_converter,
            scenario_dict=scenario_dict,
            input_data_checks=self.input_data_checks,
            year_specific_ts=year_specific_ts,
            folder_path=folder_path,
            attribute_data_loader=self.attribute_data_loader,
        )
        # dict to save the parameter units element-wise and to save them in the results
        self.units: dict[str, Any] = {}
        self.raw_time_series = {}

        self._initialize()

    def _initialize(self):
        """Initialize the element."""
        pass

    def prepare_input_data(self) -> None:
        """Prepare structural information required to load parameters."""

    def finalize_input_data(self) -> None:
        """Apply transformations that require all parameters to be loaded."""

    def _get_input_path(self):
        """Get input path where input data is stored input_path."""
        # get technology type
        class_label = self.label
        # check if class is a subset
        if class_label not in self.dataset_path_resolver.all_sets():
            subsets = self.model_schema.config.analysis.subsets
            # iterate through subsets and check if class belongs to any of the subsets
            for set_name, subsets_list in subsets.items():
                if class_label in subsets_list:
                    class_label = set_name
                    break
        # get input path for current class_label
        return Path(
            self.dataset_path_resolver.folder_of_element(class_label, self.name)
        )
