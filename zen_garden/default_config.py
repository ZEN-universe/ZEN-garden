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
from collections.abc import ItemsView, KeysView, ValuesView
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError


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
