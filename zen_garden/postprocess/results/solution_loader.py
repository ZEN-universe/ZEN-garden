"""Contains the implementation of a SolutionLoader that reads the solution."""

import copy
import json
import logging
import os
import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, cast, override

import h5py  # type: ignore
import numpy as np
import pandas as pd
import pint
import xarray as xr

from zen_garden.default_config import Analysis, Solver, System
from zen_garden.utils import slice_df_by_index

logger = logging.getLogger(__name__)


class ComponentType(Enum):
    parameter = "parameter"
    variable = "variable"
    dual = "dual"
    sets = "sets"

    @classmethod
    def get_component_type_names(cls) -> list[str]:
        return [component_type.value for component_type in cls]

    @classmethod
    def get_file_names_maps(cls) -> dict[str, "ComponentType"]:
        """Method that returns a dictionary that maps the file names to the
        component types.
        """
        return {
            "param_dict.h5": ComponentType.parameter,
            "var_dict.h5": ComponentType.variable,
            "set_dict.h5": ComponentType.sets,
            "dual_dict.h5": ComponentType.dual,
        }


class TimestepType(Enum):
    yearly = "year"
    operational = "time_operation"
    storage = "time_storage_level"

    @classmethod
    def get_time_steps_names(cls) -> list[str]:
        """Method that returns a list of timestep names.
        :return: get_time_steps_names.
        """
        return [time_step_type.value for time_step_type in cls]

    @classmethod
    def get_time_step_type(cls, time_step: str | None) -> Optional["TimestepType"]:
        """Method that returns the timestep type given a timestep name.
        :param time_step: The name of the timestep.
        :return: The timestep type.
        """
        for member in cls:
            if member.value == time_step:
                return member
        return None


class Component:
    """Class that defines a component."""

    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        index_names: list[str],
        ts_type: Optional[TimestepType],
        ts_name: Optional[str],
        file_name: str,
        doc: str,
        has_units: bool,
    ) -> None:
        self._component_type = component_type
        self._name = name
        self._index_names = index_names
        self._ts_type = ts_type
        self._file_name = file_name
        self._ts_name = ts_name
        self._doc = doc
        self._has_units = has_units

    @property
    def component_type(self) -> ComponentType:
        return self._component_type

    @property
    def index_names(self) -> list[str]:
        return self._index_names

    @property
    def timestep_type(self) -> Optional[TimestepType]:
        return self._ts_type

    @property
    def timestep_name(self) -> Optional[str]:
        return self._ts_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def file_name(self) -> str:
        return self._file_name

    @property
    def doc(self) -> str:
        return self._doc

    @property
    def has_units(self) -> bool:
        return self._has_units

    @override
    def __repr__(self) -> str:
        return (
            f"Component("
            f"name={self.name}, "
            f"component_type={self.component_type}, "
            f"index_names={self.index_names}, "
            f"ts_type={self._ts_type}, "
            f"ts_name={self._ts_name}, "
            f"file_name={self.file_name}, "
            f"doc={self.doc}, "
            f"has_units={self.has_units}"
            f")"
        )


class Scenario:
    """Implementation of the scenario. In this solution version, the analysis and
    system configs are stored as jsons for each of the scenario in the
    corresponding folder.
    """

    def __init__(
        self, path: str | os.PathLike[str], name: str, base_scenario: str
    ) -> None:
        self.name = name
        self.base_name = base_scenario
        self._exists = True
        self._path = os.fspath(path)
        self._analysis: Analysis = self._read_analysis()
        self._system: System = self._read_system()
        self._solver: Solver = self._read_solver()
        self._benchmarking: dict[str, Any] = self._read_benchmarking()
        self._component_types: dict[str, list[str]] = {}
        self._components: dict[str, tuple[ComponentType, str, str]] = {}
        self._read_components()

    def _read_analysis(self) -> Analysis:
        analysis_path = os.path.join(self.path, "analysis.json")
        if not os.path.exists(analysis_path):
            logger.warning(f"analysis.json does not exist for scenario {self.name}")
            self._exists = False
            return Analysis()

        with open(analysis_path, "r") as f:
            return Analysis(**json.load(f))

    def _read_system(self) -> System:
        system_path = os.path.join(self.path, "system.json")
        if not os.path.exists(system_path):
            logger.warning(f"system.json does not exist for scenario {self.name}")
            return System()

        with open(system_path, "r") as f:
            return System(**json.load(f))

    def _read_solver(self) -> Solver:
        solver_path = os.path.join(self.path, "solver.json")
        if not os.path.exists(solver_path):
            logger.warning(f"solver.json does not exist for scenario {self.name}")
            return Solver()

        with open(solver_path, "r") as f:
            return Solver(**json.load(f))

    def _read_benchmarking(self) -> dict[str, Any]:
        benchmarking_path = os.path.join(self.path, "benchmarking.json")
        if os.path.exists(benchmarking_path):
            with open(benchmarking_path, "r") as f:
                return json.load(f)
        else:
            return {}

    def _read_ureg(self) -> pint.UnitRegistry:
        # suppress pint output about redefining units
        logging.getLogger("pint").setLevel(logging.ERROR)
        # load ureg
        ureg: pint.UnitRegistry = copy.copy(pint.UnitRegistry())
        unit_path = os.path.join(self.path, "unit_definitions.txt")
        if os.path.exists(unit_path):
            ureg.load_definitions(unit_path)
        return ureg

    def convert_ts2year(
        self, df: "pd.DataFrame | pd.Series[Any]"
    ) -> "pd.DataFrame | pd.Series[Any]":
        """Converts the yearly ts column to the corresponding year."""
        df = df.copy()
        if isinstance(df, pd.Series):
            year_index = df.index
        else:
            year_index = df.columns
        assert pd.api.types.is_any_real_numeric_dtype(year_index), (
            f"DataFrame columns must be numeric to convert to year, not "
            f"{year_index.to_list()}."
        )
        ry = self.system.reference_year
        del_y = self.system.interval_between_years
        years = [ry + int(i) * del_y for i in year_index]
        if isinstance(df, pd.Series):
            df.index = years
        else:
            df.columns = years
        return df

    def rename_index(
        self, df: "pd.DataFrame | pd.Series[Any]"
    ) -> "pd.DataFrame | pd.Series[Any]":
        """Renames the index of the dataframe."""
        if isinstance(df, pd.Series) and df.index.name == "scalar":
            return df.reset_index(drop=True)
        map = self.analysis.header_data_inputs
        df = df.copy()
        renamed_index = [
            map[str(idx)] if idx in map.keys() else idx for idx in df.index.names
        ]
        df.index.names = renamed_index
        return df

    def convert_year2ts(self, year: int) -> int:
        """Converts the year to the corresponding time step."""
        assert isinstance(year, int), f"Year must be an integer, not {type(year)}."
        ry = self.system.reference_year
        del_y = self.system.interval_between_years
        all_years = [ry + i * del_y for i in range(self.system.optimized_years)]
        if year in all_years:
            ts = (year - ry) // del_y
        elif year <= self.analysis.earliest_year_of_data and year in range(
            self.system.optimized_years
        ):
            warnings.warn(
                f"Selecting the yearly time steps ({year}) instead of the "
                f"actual year ({ry + del_y * year}) is deprecated. Please use "
                "the actual year.",
                DeprecationWarning,
                stacklevel=2,
            )
            ts = year
        else:
            raise KeyError(f"Year {year} not in optimized years {all_years}.")
        return ts

    def _read_components(self) -> None:
        """Create the component instances.

        The components are stored in three files and the file-names define
        the types of the component. Furthermore, the timestep name and type
        are derived by checking if any of the defined time steps name is
        in the index of the dataframe.
        """
        component_types: dict[str, list[str]] = {
            t: [] for t in ComponentType.get_component_type_names()
        }
        components: dict[str, tuple[ComponentType, str, str]] = {}

        if not self._exists:
            self._component_types = component_types
            self._components = components
            return

        if self.has_rh:
            mf_name = [i for i in os.listdir(self.path) if "MF_" in i][0]
            component_folder = os.path.join(self.path, mf_name)
        else:
            component_folder = self.path

        if check_if_v1_leq_v2(get_solution_version(self), "v3"):
            for (
                file_name,
                component_type,
            ) in ComponentType.get_file_names_maps().items():
                file_path = os.path.join(component_folder, file_name)

                if not os.path.exists(file_path):
                    continue

                h5_file = h5py.File(file_path)
                component_types[component_type.value] = list(h5_file.keys())
                components.update(
                    {
                        cn: (
                            component_type,
                            file_name,
                            file_path,
                        )
                        for cn in h5_file.keys()
                    }
                )
        else:
            file_name_map = {
                ComponentType.dual: "duals.nc",
                ComponentType.variable: "variables.nc",
                ComponentType.parameter: "parameters.nc",
                ComponentType.sets: "sets.h5",
            }
            for component_type, file_name in file_name_map.items():
                file_path = os.path.join(component_folder, file_name)

                if not os.path.exists(file_path):
                    continue

                if component_type is ComponentType.sets:
                    h5_file = h5py.File(file_path)
                    component_types[component_type.value] = list(h5_file.keys())
                    components.update(
                        {
                            cn: (
                                component_type,
                                file_name,
                                file_path,
                            )
                            for cn in h5_file.keys()
                        }
                    )
                else:
                    nc_file = xr.open_dataset(file_path)
                    component_types[component_type.value] = list(
                        str(cn) for cn in nc_file.data_vars
                    )
                    components.update(
                        {
                            str(cn): (
                                component_type,
                                file_name,
                                file_path,
                            )
                            for cn in nc_file.data_vars
                        }
                    )

        self._component_types = component_types
        self._components = components

    @property
    def components(self) -> dict[str, tuple[ComponentType, str, str]]:
        return self._components

    @property
    def component_types(self) -> dict[str, list[str]]:
        return self._component_types

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    @property
    def solver(self) -> Solver:
        return self._solver

    @property
    def system(self) -> System:
        return self._system

    @property
    def benchmarking(self) -> dict[str, Any]:
        return self._benchmarking

    @property
    def path(self) -> str:
        return self._path

    @property
    def has_rh(self) -> bool:
        return self.system.use_rolling_horizon

    @property
    def ureg(self) -> pint.UnitRegistry:
        return self._read_ureg()

    @property
    def exists(self) -> bool:
        return self._exists

    def get_component(self, component_name: str) -> Component:
        """Method that returns a component given its name.
        :param component_name: The name of the component.
        :return: The component.
        """
        if component_name not in self.components:
            raise KeyError(
                f"Component {component_name} not found in scenario "
                f"{self.name}. Available components: "
                f"{list(self.components.keys())}"
            )

        version = get_solution_version(self)
        component_type, file_name, file_path = self.components[component_name]
        if check_if_v1_leq_v2(version, "v3"):
            h5_file = h5py.File(file_path)
            index_names = get_index_names(h5_file, component_name, version)
            time_index = set(index_names).intersection(
                set(TimestepType.get_time_steps_names())
            )
            timestep_name = time_index.pop() if len(time_index) > 0 else None
            timestep_type = TimestepType.get_time_step_type(timestep_name)

            doc = get_doc(h5_file, component_name, version)

            has_units = get_has_units(h5_file, component_name, version)

            return Component(
                component_name,
                component_type,
                index_names,
                timestep_type,
                timestep_name,
                file_name,
                doc,
                has_units,
            )
        else:
            file_path = Path(file_path)
            if component_type is ComponentType.sets:
                h5_file = h5py.File(file_path)
                index_names = get_index_names(h5_file, component_name, version)
                time_index = set(index_names).intersection(
                    set(TimestepType.get_time_steps_names())
                )
                timestep_name = time_index.pop() if len(time_index) > 0 else None
                timestep_type = TimestepType.get_time_step_type(timestep_name)

                doc = get_doc_from_json(
                    file_path.parent / "sets_docs.json", component_name, version
                )

                has_units = False  # Sets do not have units

                return Component(
                    component_name,
                    component_type,
                    index_names,
                    timestep_type,
                    timestep_name,
                    file_name,
                    doc,
                    has_units,
                )
            else:
                nc_file = xr.open_dataset(file_path)
                index_names = [str(dim) for dim in nc_file[component_name].dims]
                time_index_map = {
                    "set_years": TimestepType.yearly,
                    "set_time_steps_operation": TimestepType.operational,
                    "set_time_steps_storage_level": TimestepType.storage,
                    "set_time_steps_storage": TimestepType.storage,
                }
                time_index = set(index_names).intersection(set(time_index_map.keys()))
                timestep_name = time_index.pop() if len(time_index) > 0 else None
                timestep_type = (
                    time_index_map[timestep_name] if timestep_name is not None else None
                )

                DOCS_FILENAME_MAP = {
                    ComponentType.parameter: "parameters_docs.json",
                    ComponentType.variable: "variables_docs.json",
                    ComponentType.dual: "duals_docs.json",
                }
                doc = get_doc_from_json(
                    file_path.parent / DOCS_FILENAME_MAP[component_type],
                    component_name,
                    version,
                )

                has_units = component_type in [
                    ComponentType.parameter,
                    ComponentType.variable,
                ] and component_name in h5py.File(
                    str(file_path).replace(".nc", "_units.h5")
                )

                return Component(
                    component_name,
                    component_type,
                    index_names,
                    timestep_type,
                    timestep_name,
                    file_name,
                    doc,
                    has_units,
                )


class SolutionLoader:
    """Implementation of a SolutionLoader."""

    def __init__(self, path: str | os.PathLike[str], enable_cache: bool = True) -> None:
        self.path = os.fspath(path)
        assert len(os.listdir(path)) > 0, f"Path {path} is empty."
        self._scenarios: dict[str, Scenario] = self._read_scenarios()
        self._ureg = get_first_scenario(self._scenarios).ureg
        self._series_cache: dict[str, "pd.Series[Any]"] = {}
        self.enable_cache = enable_cache

    @property
    def scenarios(self) -> dict[str, Scenario]:
        return self._scenarios

    @property
    def name(self) -> str:
        scenario = get_first_scenario(self._scenarios)
        name = scenario.analysis.dataset.split("/")[-1]
        return name

    @property
    def has_duals(self) -> bool:
        first_scenario = get_first_scenario(self._scenarios)
        return first_scenario.solver.save_duals

    @property
    def has_parameters(self) -> bool:
        first_scenario = get_first_scenario(self._scenarios)
        if not hasattr(first_scenario.solver, "save_parameters"):
            return True
        return first_scenario.solver.save_parameters

    def _combine_dataseries(
        self,
        component: Component,
        scenario: Scenario,
        pd_dict: dict[int, "pd.Series[Any]"],
    ) -> "pd.DataFrame | pd.Series[Any]":
        """Method that combines the values when a solution is created without
        perfect foresight given a component, a scenario and a dictionary
        containing the name of the MF-data (Format: "MF_{year}").
        """
        series_to_concat = []
        optimized_years = sorted(pd_dict.keys())
        for year in optimized_years:
            if year != optimized_years[-1]:
                next_year = optimized_years[optimized_years.index(year) + 1]
            else:
                next_year = year + 1
            decision_horizon = tuple(range(year, next_year))
            current_mf = pd_dict[year]
            if current_mf.empty:
                continue
            if component.timestep_type is TimestepType.yearly:
                if check_if_v1_leq_v2(get_solution_version(scenario), "v3"):
                    year_index = "year"
                else:
                    year_index = "set_years"
                year_series = current_mf[
                    current_mf.index.get_level_values(year_index).isin(decision_horizon)
                ]
                series_to_concat.append(year_series)
            elif component.timestep_type in [
                TimestepType.operational,
                TimestepType.storage,
            ]:
                assert component.timestep_name is not None

                time_steps = self.get_timesteps_of_years(
                    scenario, component.timestep_type, decision_horizon
                )
                time_step_list = {tstep for tstep in time_steps}
                all_timesteps = current_mf.index.get_level_values(
                    component.timestep_name
                )
                year_series = current_mf[[i in time_step_list for i in all_timesteps]]
                series_to_concat.append(year_series)
            else:
                series_to_concat.append(current_mf)
                break

        if len(series_to_concat) == 0:
            return pd.Series(dtype=float)

        return pd.concat(series_to_concat)

    def _concatenate_raw_dataseries(
        self,
        pd_dict: dict[int | str, "pd.Series[Any]"],
    ) -> "pd.DataFrame | pd.Series[Any]":
        """Method that concatenates the raw values when a solution is created
        without perfect foresight given a component, a scenario and a
        dictionary containing the name of the MF-data (Format: "MF_{year}").
        The raw values are not combined, i.e., the data is kept for all the
        foresight steps.
        """
        series = pd.concat(pd_dict, keys=pd_dict.keys())
        series = series.sort_index(level=0)
        index_names = pd_dict[list(pd_dict.keys())[0]].index.names
        new_index_names = ["mf"] + index_names
        series.index.names = new_index_names
        return series

    def get_component_data(
        self,
        scenario: Scenario,
        component: Component,
        keep_raw: bool = False,
        data_type: Literal["dataframe", "units"] = "dataframe",
        index=None,
    ) -> "pd.DataFrame | pd.Series[Any]":
        """Returns the actual component values given
        a component and a scenario. Already combines the yearly data if the
        solution does not use perfect foresight, unless explicitly desired
        otherwise (keep_raw = True).
        """
        if index is None:
            index = tuple()
        version = get_solution_version(scenario)
        if scenario.has_rh:
            # If solution has rolling horizon, load the values for all the foresight
            # steps and combine them.
            pattern = re.compile(r"^MF_\d+(_.*)?$")
            subfolder_names = list(
                filter(lambda x: pattern.match(x), os.listdir(scenario.path))
            )
            combined_series: dict[int, pd.Series[Any]] = {}
            raw_series: dict[int | str, pd.Series[Any]] = {}

            for subfolder_name in subfolder_names:
                sf_stripped = subfolder_name.replace("MF_", "")
                mf_idx: int | str
                if not sf_stripped.isnumeric():
                    if keep_raw:
                        mf_idx = subfolder_name.replace("MF_", "")
                    else:
                        continue
                else:
                    mf_idx = int(subfolder_name.replace("MF_", ""))
                file_path = os.path.join(
                    scenario.path, subfolder_name, component.file_name
                )
                series = get_df_from_path(
                    file_path, component, version, data_type, index
                )
                if keep_raw:
                    raw_series[mf_idx] = series
                else:
                    assert isinstance(mf_idx, int)
                    combined_series[mf_idx] = series
            if not keep_raw:
                combined_dataseries = self._combine_dataseries(
                    component, scenario, combined_series
                )
            else:
                combined_dataseries = self._concatenate_raw_dataseries(raw_series)
            return combined_dataseries
        else:
            # If solution does not use rolling horizon, simply load the HDF file.
            file_path = os.path.join(scenario.path, component.file_name)
            ans = get_df_from_path(file_path, component, version, data_type, index)
            return ans

    def _read_scenarios(self) -> dict[str, Scenario]:
        """Create the scenario instances. The definitions of the scenarios are
        stored in the scenarios.json files. If the solution does not have
        multiple scenarios, we store the solution as "none".
        """
        scenarios_json_path = os.path.join(self.path, "scenarios.json")
        ans: dict[str, Scenario] = {}
        with open(scenarios_json_path, "r") as f:
            scenario_configs = json.load(f)

        if len(scenario_configs) == 1:
            scenario_name = "none"
            scenario_path = self.path
            ans[scenario_name] = Scenario(scenario_path, scenario_name, "")
        else:
            for scenario_id, scenario_config in scenario_configs.items():
                scenario_name = f"scenario_{scenario_id}"
                scenario_path = os.path.join(
                    self.path, f"scenario_{scenario_config['base_scenario']}"
                )

                base_scenario = scenario_config["base_scenario"]

                # Some scenarios have additional parameter definitions that are
                # stored in subfolders.
                scenario_subfolder = scenario_config["sub_folder"]

                if scenario_subfolder != "":
                    scenario_path = os.path.join(
                        scenario_path, f"scenario_{scenario_subfolder}"
                    )

                scenario = Scenario(scenario_path, scenario_name, base_scenario)

                if scenario.exists:
                    ans[scenario_name] = scenario

        return ans

    def get_timestep_duration(
        self, scenario: Scenario, component: Component
    ) -> "pd.Series[Any]":
        """The timestep duration is stored as any other component, the only thing
        is to define the correct name depending on the component timestep type.
        """
        if component.timestep_type is TimestepType.operational:
            timestep_duration_name = "time_steps_operation_duration"
        else:
            timestep_duration_name = "time_steps_storage_duration"
        version = get_solution_version(scenario)
        if check_if_v1_leq_v2(version, "v0"):
            time_step_duration = self.get_component_data(
                scenario, scenario.get_component(timestep_duration_name)
            )
        else:
            time_steps_file_name = _get_time_steps_file(scenario)
            time_steps_file_name = time_steps_file_name + ".json"
            dict_path = os.path.join(
                scenario.path,
                time_steps_file_name,
            )
            with open(dict_path) as json_file:
                ans = json.load(json_file)
            time_step_duration = pd.Series(ans[timestep_duration_name])
            time_step_duration.index = time_step_duration.index.astype(int)
            time_step_duration = time_step_duration.astype(int)

        assert type(time_step_duration) is pd.Series

        return time_step_duration

    def get_timesteps(
        self, scenario: Scenario, component: Component, year: int
    ) -> "pd.Series[Any]":
        """THe timesteps are stored in a file HDF-File called
        dict_all_sequence_time_steps saved for each scenario. The name of the
        dataframe depends on the timestep type.
        """
        time_steps_file_name = _get_time_steps_file(scenario)

        timesteps_name = (
            "time_steps_year2operation"
            if component.timestep_type is TimestepType.operational
            else "time_steps_year2storage"
        )
        version = get_solution_version(scenario)
        if check_if_v1_leq_v2(version, "v0"):
            time_steps_file_name = time_steps_file_name + ".h5"
            dict_path = os.path.join(
                scenario.path,
                time_steps_file_name,
            )
            ans = pd.read_hdf(dict_path, f"{timesteps_name}/{year}")
        else:
            time_steps_file_name = time_steps_file_name + ".json"
            dict_path = os.path.join(
                scenario.path,
                time_steps_file_name,
            )
            with open(dict_path) as json_file:
                ans = json.load(json_file)
            ans = pd.Series(ans[timesteps_name][str(year)])

        assert type(ans) is pd.Series

        return ans

    def get_timesteps_of_years(
        self, scenario: Scenario, ts_type: TimestepType, years: tuple
    ) -> "pd.DataFrame | pd.Series[Any]":
        """Method that returns the timesteps of the scenario for a given year."""
        sequence_time_steps_name = _get_time_steps_file(scenario)
        version = get_solution_version(scenario)
        if check_if_v1_leq_v2(version, "v0"):
            sequence_time_steps_name = sequence_time_steps_name + ".h5"
            time_step_path = os.path.join(scenario.path, sequence_time_steps_name)
            time_step_file = h5py.File(time_step_path)
        else:
            sequence_time_steps_name = sequence_time_steps_name + ".json"
            time_step_path = os.path.join(scenario.path, sequence_time_steps_name)
            with open(time_step_path) as json_file:
                time_step_file = json.load(json_file)

        if ts_type is TimestepType.storage:
            time_step_name = "time_steps_year2storage"
        elif ts_type is TimestepType.operational:
            time_step_name = "time_steps_year2operation"
        else:
            raise KeyError(f"Time step type {ts_type} not found.")

        time_step_yearly = time_step_file[time_step_name]

        time_steps = []
        for year in years:
            year_series = time_step_yearly[str(year)]
            if check_if_v1_leq_v2(version, "v0"):
                time_steps.append(pd.read_hdf(time_step_path, year_series.name))
            else:
                time_steps.append(pd.Series(time_step_yearly[str(year)]))

        time_steps = pd.concat(time_steps).reset_index(drop=True)
        return time_steps

    def get_sequence_time_steps(
        self, scenario: Scenario, timestep_type: TimestepType
    ) -> "pd.Series[Any]":
        """Method that returns the sequence time steps of a scenario.

        Args:
            scenario
            timestep_type
        """
        time_steps_file_name = _get_time_steps_file(scenario)

        if timestep_type is TimestepType.operational:
            sequence_timesteps_name = "operation"
        elif timestep_type is TimestepType.storage:
            sequence_timesteps_name = "storage"
        else:
            sequence_timesteps_name = "yearly"
        version = get_solution_version(scenario)
        if check_if_v1_leq_v2(version, "v0"):
            time_steps_file_name = time_steps_file_name + ".h5"
            dict_path = os.path.join(
                scenario.path,
                time_steps_file_name,
            )
            ans = pd.read_hdf(dict_path, sequence_timesteps_name)
        else:
            time_steps_file_name = time_steps_file_name + ".json"
            dict_path = os.path.join(
                scenario.path,
                time_steps_file_name,
            )
            with open(dict_path) as json_file:
                ans = json.load(json_file)
            ans = pd.Series(ans[sequence_timesteps_name])
        return ans

    def get_optimized_years(self, scenario: Scenario) -> list[int]:
        """Method that returns the years for which the solution was optimized."""
        time_steps_file_name = _get_time_steps_file(scenario)

        try:
            version = get_solution_version(scenario)
            if check_if_v1_leq_v2(version, "v0"):
                time_steps_file_name = time_steps_file_name + ".h5"
                dict_path = os.path.join(
                    scenario.path,
                    time_steps_file_name,
                )
                ans = pd.read_hdf(dict_path, "optimized_time_steps").tolist()
            else:
                time_steps_file_name = time_steps_file_name + ".json"
                dict_path = os.path.join(
                    scenario.path,
                    time_steps_file_name,
                )
                with open(dict_path) as json_file:
                    ans = json.load(json_file)
                ans = ans["optimized_time_steps"]

        # if old version of the solution
        except Exception:
            if scenario.has_rh:
                pattern = re.compile(r"^MF_\d+$")
                subfolder_names = list(
                    filter(lambda x: pattern.match(x), os.listdir(scenario.path))
                )
                ans = [
                    int(subfolder_name.replace("MF_", ""))
                    for subfolder_name in subfolder_names
                ]
            else:  # if no rolling horizon, single optimized year
                ans = [0]

        return ans

    def get_time_steps_storage_level_startend_year(
        self,
        scenario: Scenario,
    ) -> dict[int, int]:
        """Return time steps that define the start and end of the storage level.

        :param scenario: scenario name.
        """
        version = get_solution_version(scenario)
        if check_if_v1_leq_v2(version, "v1"):
            sequence = self.get_sequence_time_steps(scenario, TimestepType.storage)
            time_steps_per_year = scenario.system.unaggregated_time_steps_per_year
            dict_startend = {}
            for i in np.arange(scenario.system.optimized_years):
                start_idx = i * time_steps_per_year
                end_idx = (i + 1) * time_steps_per_year - 1
                dict_startend[sequence.iloc[start_idx]] = sequence.iloc[end_idx]
        else:
            time_steps_file_name = _get_time_steps_file(scenario)
            time_steps_file_name = time_steps_file_name + ".json"
            dict_path = os.path.join(
                scenario.path,
                time_steps_file_name,
            )
            with open(dict_path) as json_file:
                ans = json.load(json_file)
            dict_startend = ans["time_steps_storage_level_startend_year"]
            dict_startend = {int(k): int(v) for k, v in dict_startend.items()}
        return dict_startend


#### Helper functions
def get_first_scenario(scenarios: dict[str, Scenario]) -> Scenario:
    """Helper-function that returns the first scenario of a dictionary of scenarios.
    :param scenarios: The dictionary of scenarios.

    :return: The first scenario of the dictionary.
    """
    return scenarios[next(iter(scenarios.keys()))]


def get_solution_version(scenario: Scenario) -> str:
    """Helper-function that checks the version of the solution.
    The order in versions is important as the highest version should be checked
    last {v1,v2,...}.

    :param scenario: The scenario for which the version should be checked.

    :return: The version of the solution.
    """
    versions = {"v1": "2.0.14", "v2": "2.2.15", "v3": "2.9.2", "v4": "3.0.0"}
    version = "v0"
    if hasattr(scenario.analysis, "zen_garden_version"):
        zen_garden_version = scenario.analysis.zen_garden_version
        if zen_garden_version is None:
            return version
        for k, v in versions.items():
            if check_if_v1_leq_v2(v, zen_garden_version):
                version = k
    return version


def check_if_v1_leq_v2(version1: str, version2: str) -> bool:
    """Helper-function that compares two versions.

    The comparison is done by checking if version1 <= version2.
    Each version is a string of *.*.* format, where the number of positions is
    arbitrary.

    :param version1: The first version.
    :param version2: The second version.

    :return: True if the version1 <= version2.
    """
    if version1 is None:
        return True
    version1 = version1.replace("v", "")
    version2 = version2.replace("v", "")
    v1 = version1.split(".")
    v2 = version2.split(".")

    for i in range(len(v1)):
        if int(v1[i]) > int(v2[i]):
            return False
        elif int(v1[i]) < int(v2[i]):
            return True
    return True


def get_index_names(h5_file: h5py.File, component_name: str, version: str) -> list[str]:
    """Helper-function that returns the pandas dataframe index names of a h5-Group."""
    if check_if_v1_leq_v2(version, "v0"):
        h5_group = h5_file[component_name + "/dataframe"]
        ans = []
        for val in h5_group.values():
            try:
                name = val.attrs["name"].decode()
            except KeyError:
                continue

            if name != "N.":
                ans.append(name)
    elif check_if_v1_leq_v2(version, "v3"):
        h5_group = h5_file[component_name]
        index_names = h5_group.attrs["index_names"].decode()
        ans = index_names.split(",")
    else:
        series = pd.read_hdf(h5_file.filename, component_name)
        ans = series.index.names
    return ans


def get_doc(h5_file: h5py.File, component_name: str, version: str) -> str:
    """Helper-function that returns the documentation of a h5-Group."""
    if check_if_v1_leq_v2(version, "v0"):
        doc = str(
            np.char.decode(h5_file[component_name + "/docstring"].attrs.get("value"))
        )
    elif check_if_v1_leq_v2(version, "v3"):
        doc = h5_file[component_name].attrs["docstring"].decode()
    else:
        raise ValueError(f"Version {version} not supported for getting docstring.")
    if ";" in doc and ":" in doc:
        doc = "\n".join(
            [f"{v.split(':')[0]}: {v.split(':')[1]}" for v in doc.split(";")]
        )
    return doc


def get_doc_from_json(json_path: str | Path, component_name: str, version: str) -> str:
    """Helper-function that returns the documentation of a component from a json file.

    :param json_path: The path to the json file.
    :param component_name: The name of the component.
    :param version: The version of the component.
    :return: The documentation of the component.
    """
    with open(json_path, "r") as f:
        doc_dict = cast(dict[str, str], json.load(f))

    if component_name not in doc_dict:
        raise KeyError(f"Component {component_name} not found in {json_path}.")

    doc = doc_dict[component_name]
    if ";" in doc and ":" in doc:
        doc = "\n".join(v.replace(":", ": ") for v in doc.split(";"))
    return doc


def get_has_units(h5_file: h5py.File, component_name: str, version: str) -> bool:
    """Helper-function that returns a boolean indicating if the component has
    units.
    """
    if check_if_v1_leq_v2(version, "v0"):
        has_units = "units" in h5_file[component_name]
    else:
        has_units = h5_file[component_name].attrs["has_units"]
    if has_units == 1:
        has_units = True
    elif has_units == 0:
        has_units = False
    else:
        raise ValueError(f"Value {has_units} for has_units not supported.")
    return has_units


def get_df_from_path(
    path: str,
    component: Component,
    version: str,
    data_type: Literal["dataframe", "units"] = "dataframe",
    index: tuple[str, ...] | dict[str, str] | None = None,
) -> "pd.Series[Any]":
    """Helper-function that returns a Pandas series given the path of a file and
    the component name.
    """

    if check_if_v1_leq_v2(version, "v0"):
        pd_read = pd.read_hdf(path, f"{component.name}/{data_type}")
        if len(index) > 0:
            pd_read = slice_df_by_index(pd_read, index)
    elif check_if_v1_leq_v2(version, "v2"):
        if index is None:
            index = tuple()
        if data_type == "dataframe":
            try:
                pd_read = pd.read_hdf(path, component.name, where=index)
            except Exception:
                pd_read = pd.read_hdf(path, component.name)
            if isinstance(pd_read, pd.DataFrame):
                pd_read = pd_read["value"]
        elif data_type == "units":
            try:
                pd_read = pd.read_hdf(
                    path, component.name, where=index, columns=["units"]
                )
            except Exception:
                try:
                    pd_read = pd.read_hdf(path, component.name, columns=["units"])
                except IndexError:
                    logger.warning(
                        "Cannot retrieve units. Make sure you have updated the "
                        "environment to the latest version."
                    )
                    return pd.Series([])
        else:
            raise ValueError(f"Data type {data_type} not supported.")
    elif (
        check_if_v1_leq_v2(version, "v3")
        or component.component_type is ComponentType.sets
    ):
        if index is None:
            index = tuple()
        if data_type == "dataframe":
            try:
                pd_read = pd.read_hdf(path, component.name, where=index)
            except Exception:
                pd_read = pd.read_hdf(path, component.name)
        elif data_type == "units":
            try:
                pd_read = pd.read_hdf(path, component.name + "_units", where=index)
            except Exception:
                pd_read = pd.read_hdf(path, component.name + "_units")
        else:
            raise ValueError(f"Data type {data_type} not supported.")
    else:
        if index == ():
            index = {}
        elif type(index) is not dict:
            raise ValueError(f"Index must be a mapping for version {version}.")
        if data_type == "dataframe":
            pd_read = (
                xr.open_dataset(path)[component.name].query(index).to_series().dropna()
            )
        elif data_type == "units":
            pd_read = pd.read_hdf(path.replace(".nc", "_units.h5"), component.name)
        else:
            raise ValueError(f"Data type {data_type} not supported.")

        # If index is not empty, slice the series by index

    if isinstance(pd_read, pd.DataFrame):
        ans = pd_read.squeeze()
    elif isinstance(pd_read, pd.Series):
        ans = pd_read
    else:
        raise ValueError(f"Data type {type(pd_read)} not supported.")

    if isinstance(ans, (np.float64, str)):
        ans = pd.Series([ans], index=pd_read.index)

    assert type(ans) is pd.Series, f"Type {type(ans)} not supported."

    return ans


def _get_time_steps_file(scenario):
    """Helper-function that returns the name of the time steps file of a scenario.
    :param scenario:
    :return: time_steps_file_name.
    """
    time_steps_file_name = [
        os.path.splitext(i)[0]
        for i in os.listdir(scenario.path)
        if "dict_all_sequence_time_steps" in i and ".lock" not in i
    ]
    time_steps_file_name = np.unique(time_steps_file_name)
    assert (
        len(time_steps_file_name) == 1
    ), f"Multiple time steps files found: {time_steps_file_name}"
    time_steps_file_name = time_steps_file_name[0]
    return time_steps_file_name
