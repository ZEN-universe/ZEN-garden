"""
:Title:        ZEN-GARDEN
:Created:      October-2021
:Authors:      Alissa Ganter (aganter@ethz.ch)
:Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Default configuration. Changes from the default values are specified in config.py (folders data/tests) and system.py (individual datasets)
"""

from pydantic import BaseModel, ConfigDict
from typing import Any, Optional


class Subscriptable(BaseModel):
    def __getitem__(self, __name) -> Any:
        return getattr(self, __name)

    def __setitem__(self, __name: str, __value: Any) -> None:
        setattr(self, __name, __value)

    def keys(self):
        return self.model_dump().keys()

    def update(self, new_values: dict):
        for key, val in new_values.items():
            if type(val) is dict:
                getattr(self, key).update(val)
            else:
                setattr(self, key, val)

    def items(self):
        return self.model_dump().items()

    def __iter__(self):
        self.fix_keys = list(self.model_dump().keys())
        self.i = 0
        return self

    def __next__(self):
        if self.i < len(self.fix_keys):
            ans = self.fix_keys[self.i]
            self.i += 1
            return ans
        else:
            raise StopIteration


class Subsets(Subscriptable):
    set_carriers: list[str] = []
    set_technologies: list[str] = [
        "set_conversion_technologies",
        "set_transport_technologies",
        "set_storage_technologies",
    ]
    set_conversion_technologies: list[str] = ["set_conditioning_technologies"]


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
    set_dependent_carriers: str = "carrier"
    set_conditioning_carriers: str = "carrier"
    set_conditioning_carrier_parents: str = "carrier"
    set_elements: str = "element"
    set_conversion_technologies: str = "technology"
    set_transport_technologies: str = "technology"
    set_storage_technologies: str = "technology"
    set_technologies: str = "technology"
    set_technologies_existing: str = "technology_existing"
    set_capacity_types: str = "capacity_type"


class TimeSeriesAggregation(Subscriptable):
    clusterMethod: str = "hierarchical"
    solver: str = "gurobi"
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
    postprocess: bool = False
    folder_output: str = "./outputs/"
    overwrite_output: bool = True
    output_format: str = "h5"
    write_results_yml: bool = False
    max_output_size_mb: int = 500
    folder_name_system_specification: str = "system_specification"
    earliest_year_of_data: int = 1900
    use_capacities_existing: bool = False


class System(Subscriptable):
    model_config = ConfigDict(extra="allow")
    set_carriers: list = []
    set_conditioning_carriers: list = []
    set_capacity_types: list[str] = ["power", "energy"]
    set_conversion_technologies: list[str] = []
    set_conditioning_technologies: list = []
    set_storage_technologies: list = []
    storage_periodicity: list = True
    set_transport_technologies: list = []
    double_capex_transport: bool = False
    set_bidirectional_transport_technologies: list = []
    set_nodes: list[str] = []
    conduct_time_series_aggregation: bool = False
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
    use_rolling_horizon: int = False
    years_in_rolling_horizon: int = 5


class SolverOptions(Subscriptable):
    logfile: str = ".//outputs//logs//GurobiLogFile.log"
    MIPGap: Optional[str] = None
    TimeLimit: Optional[int] = None
    Method: Optional[str] = None


class Solver(Subscriptable):
    name: str = "glpk"
    solver_options: SolverOptions = SolverOptions()
    solver_dir: str = ".//outputs//solver_files"
    keep_files: bool = False
    io_api: str = "direct"
    add_duals: bool = False
    analyze_numerics: bool = False
    recommend_base_units: bool = False
    immutable_unit: list = []
    range_unit_exponents: dict = {"min": -1, "max": 1, "step_width": 1}
    define_ton_as_metric_ton: bool = True
    rounding_decimal_points: int = 5
    rounding_decimal_points_ts: int = 5
    linear_regression_check: dict = {
        "eps_intercept": 0.1,
        "epsRvalue": 1 - (1e-5),
    }
    analyze_numerics: bool = True


class Config(Subscriptable):
    # analysis: dict = Analysis().model_dump()
    analysis: Analysis = Analysis()

    # solver: dict = Solver().model_dump()
    solver: Solver = Solver()

    system: dict = System()
    # system: System = System()

    scenarios: dict = {"": {}}


class _Config(object):
    """
    A class that contains all relevant parameters for the ZEN-Garden
    """

    def __init__(self, analysis=None, solver=None, system=None, scenarios=None):
        """
        Initializes an instance of the parameters containing all defaults. If dictionaries are provided the defaults
        are overwritten.

        :param analysis: A dictionary used to update the default values in analysis
        :param solver: A dictionary used to update the default values in solver
        :param system: A dictionary used to update the default values in system
        :param scenarios: A dictionary used to update the default values in scenarios
        :return: A class instance of Config
        """

        ## Analysis - dictionary declaration
        # This dictionary contains all the settings related to the typology of analysis given a specific system configuration.
        # The dictionary also contains default settings related to the input data.
        self.analysis = dict()

        ## Solver - dictionary declaration
        # This dictionary contains all the settings related to the solver of the optimisation problem.
        self.solver = dict()

        ## System - dictionary declaration
        # This dictionary defines the configuration of the system by selecting the subset of technologies to be included into the analysis.
        self.system = dict()

        ## Scenarios - dictionary declaration
        # This dictionary defines the set of scenarios that is evaluated.
        self.scenarios = dict()

        # set the defaults
        self._set_defaults()

        # update
        if analysis is not None:
            self.analysis.update(analysis)
        if solver is not None:
            self.solver.update(solver)
        if system is not None:
            self.system.update(system)
        if scenarios is not None:
            self.scenarios.update(scenarios)

    def _set_defaults(self):
        """
        Initializes all the default parameters
        """

        ## Analysis - Items assignment
        # objective function definition
        self.analysis["objective"] = "total_cost"
        # typology of optimisation: minimize or maximize
        self.analysis["sense"] = "minimize"
        # transport distance (euclidean or actual)
        self.analysis["transport_distance"] = "Euclidean"
        # dictionary with subsets related to set
        self.analysis["subsets"] = {
            "set_carriers": [],
            "set_technologies": [
                "set_conversion_technologies",
                "set_transport_technologies",
                "set_storage_technologies",
            ],
            "set_conversion_technologies": ["set_conditioning_technologies"],
        }
        # headers for the generation of input files
        self.analysis["header_data_inputs"] = {
            "set_nodes": "node",
            "set_edges": "edge",
            "set_location": "location",
            "set_time_steps": "time",  # IMPORTANT: time must be unique
            "set_time_steps_operation": "time_operation",
            "set_time_steps_storage_level": "time_storage_level",
            "set_time_steps_yearly": "year",  # IMPORTANT: year must be unique
            "set_time_steps_yearly_entire_horizon": "year_entire_horizon",
            "set_carriers": "carrier",
            "set_input_carriers": "carrier",
            "set_output_carriers": "carrier",
            "set_dependent_carriers": "carrier",
            "set_conditioning_carriers": "carrier",
            "set_conditioning_carrier_parents": "carrier",
            "set_elements": "element",
            "set_conversion_technologies": "technology",
            "set_transport_technologies": "technology",
            "set_storage_technologies": "technology",
            "set_technologies": "technology",
            "set_technologies_existing": "technology_existing",
            "set_capacity_types": "capacity_type",
        }
        # time series aggregation
        self.analysis["time_series_aggregation"] = {
            "clusterMethod": "hierarchical",
            "solver": "gurobi",
            "hoursPerPeriod": 1,
            "extremePeriodMethod": "None",
            "rescaleClusterPeriods": False,
            "representationMethod": "meanRepresentation",
            "resolution": 1,
            "segmentation": False,
            "noSegments": 12,
        }
        self.analysis["postprocess"] = False
        self.analysis["folder_output"] = "./outputs/"
        self.analysis["overwrite_output"] = True
        # output format, can be h5, json or gzip
        self.analysis["output_format"] = "h5"
        self.analysis["write_results_yml"] = False
        self.analysis["max_output_size_mb"] = 500
        # name of data folder for energy system specification
        self.analysis["folder_name_system_specification"] = "system_specification"
        # earliest possible year of input data, needed to differentiate between yearly and generic time indices
        self.analysis["earliest_year_of_data"] = 1900
        self.analysis["use_capacities_existing"] = False
        ## System - Items assignment
        # set of energy carriers
        self.system["set_carriers"] = []
        # set of conditioning carriers
        self.system["set_conditioning_carriers"] = []
        # set of capacity types: power-rated or energy-rated
        self.system["set_capacity_types"] = ["power", "energy"]
        # set of conversion technologies
        self.system["set_conversion_technologies"] = []
        # set of conditioning technologies
        self.system["set_conditioning_technologies"] = []
        # set of storage technologies
        self.system["set_storage_technologies"] = []
        self.system["storage_periodicity"] = True
        # set of transport technologies
        self.system["set_transport_technologies"] = []
        self.system["double_capex_transport"] = False
        self.system["set_bidirectional_transport_technologies"] = []
        # set of nodes
        self.system["set_nodes"] = []
        # toggle to use time_series_aggregation
        self.system["conduct_time_series_aggregation"] = False
        # toggle to exclude parameters from TSA, specified in system_specification/exclude_parameter_from_TSA
        self.system["exclude_parameters_from_TSA"] = True
        # toggle to perform analysis for multiple scenarios
        self.system["conduct_scenario_analysis"] = False
        # toggle to disable the default scenario (empty string), only considered if conduct_scenario_analysis is True
        self.system["run_default_scenario"] = True
        # toggle to delete all sub-scenarios that are not in the current scenario dict
        self.system["clean_sub_scenarios"] = False
        # total hours per year
        self.system["total_hours_per_year"] = 8760
        # rate at which the knowledge stock of existing capacities is depreciated annually
        self.system["knowledge_depreciation_rate"] = 0.1
        # enforce selfish behavior
        self.system["enforce_selfish_behavior"] = False

        ## Solver - Items assignment
        # solver selection (find more solver options for gurobi here: https://www.gurobi.com/documentation/9.1/refman/parameters.html)
        self.solver["name"] = "glpk"
        # gurobi options
        self.solver["solver_options"] = {
            "logfile": ".//outputs//logs//GurobiLogFile.log",
            "MIPGap": None,
            "TimeLimit": None,
            "Method": None,
        }
        # Directory for solver output
        self.solver["solver_dir"] = ".//outputs//solver_files"
        self.solver["keep_files"] = False
        self.solver["io_api"] = "direct"
        # This is not yet supported in linopy
        self.solver["add_duals"] = False
        # analyze numerics
        self.solver["analyze_numerics"] = False
        self.solver["recommend_base_units"] = False
        self.solver["immutable_unit"] = []
        self.solver["range_unit_exponents"] = {"min": -1, "max": 1, "step_width": 1}
        # assumes "ton" to be metric ton, not imperial ton
        self.solver["define_ton_as_metric_ton"] = True
        # round down to number of decimal points, for new capacity and unit multipliers
        self.solver["rounding_decimal_points"] = 5
        # round down to number of decimal points, for time series after TSA
        self.solver["rounding_decimal_points_ts"] = 5
        # settings for selection of x-y relationships, which are modeled as PWA, and which are modeled linearly:
        # linear regression of x-y values: if relative intercept (intercept/slope) below threshold and rvalue above threshold, model linear with slope
        self.solver["linear_regression_check"] = {
            "eps_intercept": 0.1,
            "epsRvalue": 1 - (1e-5),
        }

        ## Scenarios - dictionary declaration
        self.scenarios[""] = {}
