"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Alissa Ganter (aganter@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Default configuration. Changes from the default values are specified in config.py (folders data/tests) and system.py (individual datasets)
"""

from pydantic import BaseModel, ConfigDict
from typing import Any, Optional


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
    model_config = ConfigDict(extra="allow")
    set_carriers: list[str] = []
    set_capacity_types: list[str] = ["power", "energy"]
    set_conversion_technologies: list[str] = []
    set_storage_technologies: list[str] = []
    set_retrofitting_technologies: list[str] = []
    storage_periodicity: bool = True
    set_transport_technologies: list[str] = []
    set_transport_technologies_loss_exponential: list[str] = []
    double_capex_transport: bool = False
    set_nodes: list[str] = []
    exclude_parameters_from_TSA: bool = True
    conduct_scenario_analysis: bool = False
    run_default_scenario: bool = True
    clean_sub_scenarios: bool = False
    total_hours_per_year: int = 8760
    knowledge_depreciation_rate: float = 0.1
    enforce_selfish_behavior: bool = False
    reference_year: int = 2023
    unaggregated_time_steps_per_year: int = 8760
    aggregated_time_steps_per_year: int = 10
    conduct_time_series_aggregation: bool = True
    optimized_years: int = 3
    interval_between_years: int = 1
    use_rolling_horizon: bool = False
    years_in_rolling_horizon: int = 5
    interval_between_optimizations: int = 1
    use_capacities_existing: bool = True

class SolverOptions(Subscriptable):
    pass

class Solver(Subscriptable):
    name: str = "highs"
    solver_options: SolverOptions = SolverOptions()
    check_unit_consistency: bool = True
    solver_dir: str = ".//outputs//solver_files"
    keep_files: bool = False
    io_api: str = "lp"
    add_duals: bool = False
    recommend_base_units: bool = False
    immutable_unit: list[str] = []
    range_unit_exponents: dict[str, int] = {"min": -1, "max": 1, "step_width": 1}
    rounding_decimal_points: int = 5
    rounding_decimal_points_ts: int = 4
    linear_regression_check: dict[str, float] = {
        "eps_intercept": 0.1,
        "epsRvalue": 1 - (1e-5),
    }
    rounding_decimal_points_units: int = 6
    round_parameters: bool = True
    rounding_decimal_points_capacity: int = 4
    analyze_numerics: bool = True
    use_scaling: bool = True
    scaling_include_rhs: bool = False
    scaling_algorithm: list[str] = ["geom","geom","geom"]



class TimeSeriesAggregation(Subscriptable):
    slv: Solver = Solver()
    clusterMethod: str = "hierarchical"
    solver: str = slv.name
    hoursPerPeriod: int = 1
    extremePeriodMethod: Optional[str] = "None"
    rescaleClusterPeriods: bool = False
    representationMethod: str = "meanRepresentation"
    resolution: int = 1
    segmentation: bool = False
    noSegments: int = 12

class Analysis(Subscriptable):
    dataset: str = ""
    objective: str = "total_cost"
    sense: str = "minimize"
    transport_distance: str = "Euclidean"
    subsets: Subsets = Subsets()
    header_data_inputs: HeaderDataInputs = HeaderDataInputs()
    time_series_aggregation: TimeSeriesAggregation = TimeSeriesAggregation()
    folder_output: str = "./outputs/"
    overwrite_output: bool = True
    output_format: str = "h5"
    write_results_yml: bool = False
    max_output_size_mb: int = 500
    folder_name_system_specification: str = "system_specification"
    earliest_year_of_data: int = 1900

class Config(Subscriptable):
    analysis: Analysis = Analysis()
    solver: Solver = Solver()
    system: System = System()

    scenarios: dict[str, Any] = {"": {}}
