"""
Default configuration.

Changes from the default values are specified in config.py (folders data/tests) and system.py (individual datasets)
"""

from pydantic import BaseModel, ConfigDict
from typing import Any, Optional, Union
import importlib.metadata

class Subscriptable(BaseModel, extra="allow"):
    def __getitem__(self, __name: str) -> Any:
        return getattr(self, __name)

    def __setitem__(self, __name: str, __value: Any) -> None:
        setattr(self, __name, __value)

    def keys(self) -> Any:
        return self.model_dump().keys()

    def update(self, new_values: dict[Any, Any]) -> None:
        for key, val in new_values.items():
            if isinstance(val, dict):
                getattr(self, key).update(val)
            else:
                setattr(self, key, val)

    def items(self) -> Any:
        return self.model_dump().items()

    def values(self) -> Any:
        return self.model_dump().values()

    def __iter__(self) -> Any:
        self.fix_keys = list(self.model_dump().keys())
        self.i = 0
        return self

    def __next__(self) -> Any:
        if self.i < len(self.fix_keys):
            ans = self.fix_keys[self.i]
            self.i += 1
            return ans
        else:
            del self.i
            del self.fix_keys
            raise StopIteration


class Subsets(Subscriptable):
    set_carriers: list[str] = []
    set_technologies: dict[str, list[str]] | list[str] = {
        "set_conversion_technologies": ["set_retrofitting_technologies"],
        "set_transport_technologies": [],
        "set_storage_technologies": [],
    }


class HeaderDataInputs(Subscriptable):
    """
    Header data inputs for the model
    """
    set_nodes: str = "node"
    set_edges: str = "edge"
    set_location: str = "location"
    set_time_steps: str = "time"  # IMPORTANT: time must be unique
    set_time_steps_operation: str = "time_operation"
    set_time_steps_storage_level: str = "time_storage_level"
    set_time_steps_yearly: str = "year"  # IMPORTANT: year must be unique
    set_time_steps_yearly_entire_horizon: str = "year_entire_horizon"
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

class System(Subscriptable):
    """
    Class which contains the system configuration. This defines for example the set of carriers, technologies, etc.
    """
    set_carriers: list[str] = []
    set_capacity_types: list[str] = ["power", "energy"]
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
    aggregated_time_steps_per_year: int = 10
    conduct_time_series_aggregation: bool = False
    optimized_years: int = 1
    interval_between_years: int = 1
    use_rolling_horizon: bool = False
    years_in_rolling_horizon: int = 1
    years_in_decision_horizon: int = 1
    use_capacities_existing: bool = True
    allow_investment: bool = True

class SolverOptions(Subscriptable):
    pass

class Solver(Subscriptable):
    """
    Class which contains the solver configuration. This defines for example the solver options, scaling, etc.
    """
    name: str = "highs"
    solver_options: SolverOptions = SolverOptions()
    check_unit_consistency: bool = True
    solver_dir: str = ".//outputs//solver_files"
    keep_files: bool = False
    io_api: str = "lp"
    save_duals: bool = False
    save_parameters: bool = True
    selected_saved_parameters: list = [] # if empty, all parameters are saved
    selected_saved_variables: list = [] # if empty, all variables are saved
    selected_saved_duals: list = [] # if empty, all duals are saved (if save_duals is True)
    linear_regression_check: dict[str, float] = {
        "eps_intercept": 0.1,
        "epsRvalue": 1 - (1e-5),
    }
    round_parameters: bool = False
    rounding_decimal_points_units: int = 6
    rounding_decimal_points_capacity: int = 4
    rounding_decimal_points_tsa: int = 4
    analyze_numerics: bool = True
    run_diagnostics: bool = False
    use_scaling: bool = True
    scaling_include_rhs: bool = True
    scaling_algorithm: Union[list[str],str] = ["geom","geom","geom"]


class TimeSeriesAggregation(Subscriptable):
    """
    Class which contains the time series aggregation configuration. This defines for example the clustering method, etc.
    """
    slv: Solver = Solver()
    clusterMethod: str = "hierarchical"
    solver: str = slv.name
    hoursPerPeriod: int = 1 # keep this at 1
    extremePeriodMethod: Optional[str] = "None"
    rescaleClusterPeriods: bool = False
    representationMethod: str = "meanRepresentation"
    resolution: int = 1

class Analysis(Subscriptable):
    """
    Class which contains the analysis configuration. This defines for example the objective function, output settings, etc.
    """
    dataset: str = ""
    objective: str = "total_cost"
    sense: str = "min"
    subsets: Subsets = Subsets()
    header_data_inputs: HeaderDataInputs = HeaderDataInputs()
    time_series_aggregation: TimeSeriesAggregation = TimeSeriesAggregation()
    folder_output: str = "./outputs/"
    overwrite_output: bool = True
    output_format: str = "h5"
    earliest_year_of_data: int = 1900
    zen_garden_version: str = None

class Config(Subscriptable):
    """
    Class which contains the configuration of the model. This includes the configuratins of the system, solver, and analysis as well as the dictionary of scenarios.
    """
    analysis: Analysis = Analysis()
    solver: Solver = Solver()
    system: System = System()

    scenarios: dict[str, Any] = {"": {}}
