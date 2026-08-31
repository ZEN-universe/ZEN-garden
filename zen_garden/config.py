"""Set default configurations in ZEN_garden.

This module defines default values for all configurations in ZEN_garden. The
class :class:`Config` serves as a container grouping all model configurations.
The configurations are further organized in a class structure that resembles
that of the ZEN-garden input data. The :class:`Config` class thus links to the four
main configuration types (``analysis``, ``solver``, ``system``, and ``scenario``),
each defined using separate class. Default configurations for the ``system.json``
configurations are located in the class :class:`System`. Whenever a
configuration consists of a dictionary, a new class is defined
to provide a template for the configuration and define all required
default values.

The current structure of classes in which defaults are set is as follows:

.. code-block::

    Config
    |--Analysis
    |  |--Subsets
    |  |--HeaderDataInputs
    |  `--TimeSeriesAggregation
    |
    |--Solver
    |--System
    `--Scenario


Default values are overwritten by any changes specified in the input files
``system.json``, ``scenarios.json``, and ``config.json``.
"""

import json
import os
import warnings
from collections.abc import ItemsView, KeysView, ValuesView
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import override

from zen_garden.workflow_step import workflow_step

PROHIBITED_DATASET_CHARACTERS = [
    " ",
    ".",
    ":",
    ",",
    ";",
    "!",
    "?",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "<",
    ">",
    "&",
    "|",
    "*",
    "^",
    "%",
    "$",
    "#",
    "@",
    "`",
    "~",
    "\\",
    "/",
]


class ConfigBase(BaseModel):
    """Base class for configuration schema objects.

    Provides a shared base for configuration objects in ZEN-garden.
    It supports dictionary-style access to attributes while relying on
    :class:`pydantic.BaseModel` for schema validation and typed fields.

    Attributes:
        model_config: Pydantic configuration for the model. The class allows extra
            fields and enforces strict validation for incoming data.

    Methods:
        __getitem__: Access a field by key as if the model were a dict.
        __setitem__: Set a field by key.
        keys: Return the model field names.
        items: Return the model field items.
        values: Return the model field values.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    @classmethod
    def from_file(cls, path: str | Path) -> "ConfigBase":
        """Create a config object from a JSON or YAML file.

        Args:
            path: Path to a JSON or YAML configuration file.

        Returns:
            An instantiated subclass of :class:`ConfigBase` populated from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is empty, malformed, not a mapping, or fails
                schema validation.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: '{path}'. Expected a JSON or YAML configuration file."
            )

        try:
            with path.open("r", encoding="utf-8") as file:
                if path.suffix.lower() in {".yaml", ".yml"}:
                    data = yaml.safe_load(file)
                else:
                    warnings.warn(
                        f"Loading JSON from '{path}' is deprecated. Convert the "
                        "file to YAML.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    data = json.load(file)
        except (json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ValueError(
                f"Failed to parse configuration file '{path}': {exc}"
            ) from exc

        if data is None:
            raise ValueError(f"File '{path}' is empty or contains no data.")
        if not isinstance(data, dict):
            raise ValueError(
                f"File '{path}' must contain a top-level object mapping keys to "
                f"values. Got {type(data).__name__} instead."
            )

        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise ValueError(
                f"Failed to validate configuration from '{path}' against "
                f"the {cls.__name__} schema."
            ) from exc

    def __getitem__(self, __name: str) -> Any:
        return getattr(self, __name)

    def __setitem__(self, __name: str, __value: Any) -> None:
        setattr(self, __name, __value)

    def keys(self) -> KeysView[str]:
        return self.model_dump().keys()

    def items(self) -> ItemsView[str, Any]:
        return self.model_dump().items()

    def values(self) -> ValuesView[Any]:
        return self.model_dump().values()


class Subsets(ConfigBase):
    set_carriers: list[str] = []
    set_technologies: dict[str, list[str]] | list[str] = {
        "set_conversion_technologies": ["set_retrofitting_technologies"],
        "set_transport_technologies": [],
        "set_storage_technologies": [],
    }


class HeaderDataInputs(ConfigBase):
    """Maps input/output headers to internal set names used in ZEN-garden.

    This class defines standard header names for the input and
    output files of ZEN-garden. It provides a mapping between the column headers
    of input/output files and internal set names used in the code. For
    example, the class attribute "set_nodes" (default value "node") means that
    any input csv file with column header "node" will be interpreted as
    containing elements of the internal set "set_nodes".

    Inherits from:
        :class:`Subscriptable` - Provides dictionary-like access to attributes
        and allows input data handling via Pydantic's BaseModel
    """

    set_nodes: str = "node"
    set_edges: str = "edge"
    set_location: str = "location"
    set_hours: str = "time"  # IMPORTANT: time must be unique
    set_time_steps_operation: str = "time_operation"
    set_time_steps_storage_level: str = "time_storage_level"
    set_years: str = "year"  # IMPORTANT: year must be unique
    set_years_entire_horizon: str = "year_entire_horizon"
    set_carriers: str = "carrier"
    set_input_carriers: str = "carrier"
    set_output_carriers: str = "carrier"
    set_time_steps_storage: str = "time_storage_level"
    set_dependent_carriers: str = "carrier"
    set_elements: str = "element"
    set_conversion_technologies: str = "technology"
    set_transport_technologies: str = "technology"
    set_transport_technologies_loss_exponential: str = "technology"
    set_storage_technologies: str = "technology"
    set_technologies: str = "technology"
    set_technologies_existing: str = "technology_existing"
    set_capacity_types: str = "capacity_type"
    set_retrofitting_technologies: str = "technology"


class System(ConfigBase):
    """Class which contains the system configuration.

    This defines for example the set of carriers, technologies, etc.
    """

    set_carriers: list[str] = []
    set_capacity_types: list[str] = ["power", "energy"]
    set_technologies: list[str] = []
    set_conversion_technologies: list[str] = []
    set_storage_technologies: list[str] = []
    set_retrofitting_technologies: list[str] = []
    storage_periodicity: bool = True
    multiyear_periodicity: bool = False
    set_transport_technologies: list[str] = []
    set_transport_technologies_loss_exponential: list[str] = []
    double_capex_transport: bool = False
    set_nodes: list[str] = []
    coords: dict[str, dict[str, float]] = {}
    exclude_parameters_from_TSA: bool = True
    conduct_scenario_analysis: bool = False
    run_default_scenario: bool = True
    clean_sub_scenarios: bool = False
    total_hours_per_year: int = 8760
    knowledge_depreciation_rate: float = 0.1
    reference_year: int = 2024
    unaggregated_time_steps_per_year: int = 8760
    aggregated_time_steps_per_year: int = 8760
    conduct_time_series_aggregation: bool = False
    optimized_years: int = 1
    interval_between_years: int = 1
    use_rolling_horizon: bool = False
    years_in_rolling_horizon: int = 1
    years_in_decision_horizon: int = 1
    use_capacities_existing: bool = True
    allow_investment: bool = True
    storage_charge_discharge_binary: bool = False


class Solver(ConfigBase):
    """Class which contains the solver configuration.

    This defines for example the solver options, scaling, etc.
    """

    name: str = "highs"
    solver_options: dict[str, Any] = {}
    check_unit_consistency: bool = True
    solver_dir: str = ".//outputs//solver_files"
    keep_files: bool = False
    io_api: str = "lp"
    save_duals: bool = False
    save_reduced_costs: bool = False
    save_parameters: bool = True
    selected_saved_parameters: list[str] = []  # if empty, all parameters are saved
    selected_saved_variables: list[str] = []  # if empty, all variables are saved
    selected_saved_duals: list[str] = (
        []
    )  # if empty, all duals are saved (if save_duals is True)
    selected_saved_reduced_costs: list[str] = (
        []
    )  # if empty, all reduced costs are saved (if save_reduced_costs is True)
    round_parameters: bool = False
    rounding_decimal_points_units: int = 6
    rounding_decimal_points_capacity: int = 4
    rounding_decimal_points_tsa: int = 4
    analyze_numerics: bool = True
    run_diagnostics: bool = False
    use_scaling: bool = True
    scaling_include_rhs: bool = True
    scaling_algorithm: Union[list[str], str] = ["geom", "geom", "geom"]


class TimeSeriesAggregation(ConfigBase):
    """Class which contains the time series aggregation configuration.

    This defines for example the clustering method, etc.
    """

    clusterMethod: str = "hierarchical"
    solver: str = "highs"
    hoursPerPeriod: int = 1  # keep this at 1
    extremePeriodMethod: Optional[str] = "None"
    rescaleClusterPeriods: bool = False
    representationMethod: str = "mean"
    resolution: int = 1


class Analysis(ConfigBase):
    """Class which contains the analysis configuration.

    This defines for example the objective function, output settings, etc.
    """

    dataset: str = ""
    objective: Literal["total_cost", "total_carbon_emissions"] = "total_cost"
    sense: str = "min"
    subsets: Subsets = Subsets()
    header_data_inputs: HeaderDataInputs = HeaderDataInputs()
    time_series_aggregation: TimeSeriesAggregation = TimeSeriesAggregation()
    folder_output: str = "./outputs/"
    overwrite_output: bool = True
    output_format: str = "h5"
    output_version: int = 4
    earliest_year_of_data: int = 1900
    zen_garden_version: str | None = None


class Config(ConfigBase):
    """Class which contains the configuration of the model.

    This includes the configurations of the system, solver, and analysis as
    well as the dictionary of scenarios.
    """

    analysis: Analysis = Analysis()
    solver: Solver = Solver()
    system: System = System()
    plugins: dict[str, Any] = {}

    scenarios: dict[str, Any] = {"": {}}

    @classmethod
    @override
    @workflow_step(
        order=1, phase="Setup", label="Load config.yaml and system.yaml"
    )
    def from_file(
        cls,
        config_path: str | Path,
        dataset_path: str | Path | None = None,
        folder_output: str | Path | None = None,
    ) -> "Config":
        """Load analysis, solver, and dataset system configuration.

        Relative paths are resolved against the main configuration file. A
        supplied dataset or output folder overrides the corresponding analysis
        setting before paths and the dataset's system file are loaded.
        """
        config = super().from_file(config_path)
        assert isinstance(config, cls)
        config_path = Path(config_path)
        config_dir = config_path.parent

        if dataset_path is not None:
            config.analysis.dataset = str(dataset_path)
        if folder_output is not None:
            config.analysis.folder_output = str(folder_output)
            config.solver.solver_dir = str(folder_output)

        dataset_path = Path(config.analysis.dataset)
        if not dataset_path.is_absolute():
            dataset_path = (config_dir / dataset_path).resolve()
        if dataset_path.exists():
            config.analysis.dataset = str(dataset_path)
            system_path = next(
                (
                    dataset_path / name
                    for name in ("system.yaml", "system.yml", "system.json")
                    if (dataset_path / name).exists()
                ),
                None,
            )
            if system_path is None:
                raise FileNotFoundError(
                    f"No system definition file found in dataset '{dataset_path}'. "
                    "Expected one of: system.yaml, system.yml, system.json."
                )
            loaded_system = System.from_file(system_path)
            config.system = config.system.model_copy(update=loaded_system.model_dump())

        output_path = Path(config.analysis.folder_output)
        if not output_path.is_absolute():
            output_path = (config_dir / output_path).resolve()
        config.analysis.folder_output = str(output_path)

        solver_path = Path(config.solver.solver_dir)
        if not solver_path.is_absolute():
            solver_path = (config_dir / solver_path).resolve()
        config.solver.solver_dir = str(solver_path)
        config.analysis.zen_garden_version = version("zen-garden")
        return config

    def validate_configurations(self) -> None:
        """Validate the configuration for internal consistency and against the dataset.

        Ensures the selected dataset exists and is well-formed, that at least one
        technology is selected (removing duplicate selections), and that the
        year-related parameters are defined consistently. Raises ``AssertionError``,
        ``ValueError`` or ``FileNotFoundError`` on the first problem encountered.
        """
        self._validate_dataset()
        self._validate_technology_selections()
        self._validate_year_definitions()

    def _validate_dataset(self) -> None:
        """Ensure the chosen dataset exists and contains a system definition file."""
        dataset = os.path.basename(self.analysis.dataset)
        dirname = os.path.dirname(self.analysis.dataset)
        assert os.path.exists(
            dirname
        ), f"Requested folder {dirname} is not a valid path"
        assert os.path.exists(self.analysis.dataset), (
            f"The chosen dataset {dataset} does not exist at "
            f"{self.analysis.dataset} as it is specified in the config"
        )
        # check if any character in the dataset name is prohibited
        for char in PROHIBITED_DATASET_CHARACTERS:
            if char in dataset:
                raise ValueError(
                    f"Character {char} is not allowed in the dataset name "
                    f"{dataset}\nProhibited characters: "
                    f"{PROHIBITED_DATASET_CHARACTERS}"
                )
        system_files = [
            "system.yaml",
            "system.yml",
            "system.json",
        ]
        if not any(
            os.path.exists(os.path.join(self.analysis.dataset, filename))
            for filename in system_files
        ):
            raise FileNotFoundError(
                f"No system definition file found in dataset "
                f"'{self.analysis.dataset}'. "
                "Expected one of: system.yaml, system.yml, system.json."
            )

    def _validate_technology_selections(self) -> None:
        """Check the technology selection and drop any duplicate entries."""
        # Checks if at least one technology is selected in the system file
        assert (
            len(
                self.system.set_conversion_technologies
                + self.system.set_transport_technologies
                + self.system.set_storage_technologies
            )
            > 0
        ), "No technology selected in system"
        # Remove possible duplicates from the technology selections
        for tech_list in [
            "set_conversion_technologies",
            "set_transport_technologies",
            "set_storage_technologies",
        ]:
            techs_selected = getattr(self.system, tech_list)
            unique_elements = sorted(set(techs_selected))
            self.system = self.system.model_copy(update={tech_list: unique_elements})

    def _validate_year_definitions(self) -> None:
        """Check that year-related parameters are defined correctly."""
        # assert that number of optimized years is a positive integer
        assert (
            isinstance(self.system.optimized_years, int)
            and self.system.optimized_years > 0
        ), (
            "Number of optimized years must be a positive integer, however it "
            f"is {self.system.optimized_years}"
        )
        # assert that interval between years is a positive integer
        assert (
            isinstance(self.system.interval_between_years, int)
            and self.system.interval_between_years > 0
        ), (
            "Interval between years must be a positive integer, however it is "
            f"{self.system.interval_between_years}"
        )
        assert (
            isinstance(self.system.reference_year, int)
            and self.system.reference_year >= self.analysis.earliest_year_of_data
        ), (
            "Reference year must be an integer and larger than the defined "
            f"earliest_year_of_data: {self.analysis.earliest_year_of_data}"
        )
        # check if the number of years in the rolling horizon isn't larger than
        # the number of optimized years
        if (
            self.system.years_in_rolling_horizon > self.system.optimized_years
            and self.system.use_rolling_horizon
        ):
            warnings.warn(
                "The chosen number of years in the rolling horizon step is "
                "larger than the total number of years optimized!",
                stacklevel=2,
            )
