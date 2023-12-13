from zen_garden.postprocess.new_results.solution_loader import (
    MF as AbstractMF,
    SolutionLoader as AbstractLoader,
    Scenario as AbstractScenario,
    Component as AbstractComponent,
    ComponentType,
    TimestepType,
)

from zen_garden.model.default_config import Analysis, System
import json
import os
import h5py  # type: ignore
from typing import Optional, Any
import pandas as pd
import numpy as np


file_names_maps = {
    "param_dict.h5": ComponentType.parameter,
    "var_dict.h5": ComponentType.variable,
    "set_dict.h5": ComponentType.sets,
}

time_steps_map = {
    "year": TimestepType.yearly,
    "time_operation": TimestepType.operational,
    "time_storage_level": TimestepType.storage,
}


class MF(AbstractMF):
    def __init__(
        self,
        path: str,
        component_name: str,
        component_type: ComponentType,
        mf_step: int,
    ) -> None:
        self.path = path
        self.component_name: str = component_name
        self._series_cache: Optional["pd.Series[Any]"] = None
        self.component_type = component_type
        self.step = mf_step

    def get_series(self) -> "pd.Series[Any]":
        if self._series_cache is None:
            pd_read = pd.read_hdf(self.path, self.component_name + "/dataframe")
            if isinstance(pd_read, pd.DataFrame):
                self._series_cache = pd_read.squeeze()
            elif isinstance(pd_read, pd.Series):
                self._series_cache = pd_read
            if isinstance(self._series_cache, (np.float_, str)):
                self._series_cache = pd.Series(
                    [self._series_cache], index=pd_read.index
                )
        assert isinstance(self._series_cache, pd.Series)
        return self._series_cache

    @classmethod
    def from_hdf(cls, hdf_path: str) -> "list[MF]":
        ans: list[MF] = []
        hdf_path_parts = hdf_path.split("/")
        try:
            mf_step = int(hdf_path_parts[-2].replace("MF_", ""))
        except ValueError:
            mf_step = 0
        component_type = file_names_maps[hdf_path_parts[-1]]
        for component_name in h5py.File(hdf_path):
            ans.append(MF(hdf_path, component_name, component_type, mf_step))
        return ans

    def get_index_names(self) -> list[str]:
        h5_file = h5py.File(self.path)[self.component_name + "/dataframe"]
        ans = []

        for key, val in h5_file.items():
            if not key.startswith("axis"):
                continue
            try:
                name = val.attrs["name"].decode()
            except KeyError:
                continue

            if name != "N.":
                ans.append(name)

        return ans


class Component(AbstractComponent):
    def __init__(
        self,
        name: str,
        mf_steps: dict[int, MF],
        component_type: ComponentType,
        ts_type: Optional[TimestepType],
    ) -> None:
        self._series: Optional[pd.Series[Any]] = None
        self._mf_steps = mf_steps
        self._component_type = component_type
        self.name = name
        self._ts_type = ts_type

        if ts_type is None:
            self._ts_name = None
        else:
            self._ts_name = [
                key for key, val in time_steps_map.items() if val == ts_type
            ][0]

    @property
    def component_type(self) -> ComponentType:
        return self._component_type

    @property
    def has_mf(self) -> bool:
        return len(self._mf_steps) > 1

    @property
    def mf_steps(self) -> dict[int, MF]:  # type: ignore
        return self._mf_steps

    def get_levels(self) -> list[str]:
        return []

    @property
    def timestep_type(self) -> Optional[TimestepType]:
        return self._ts_type

    @property
    def timestep_name(self) -> Optional[str]:
        return self._ts_name

    @classmethod
    def from_scenario_folder(cls, folder_path: str) -> dict[str, "AbstractComponent"]:
        folder_content = os.listdir(folder_path)
        has_mf = "MF_1" in folder_content

        if has_mf:
            mf_folders = [
                os.path.join(folder_path, i)
                for i in folder_content
                if i.startswith("MF_")
            ]
        else:
            mf_folders = [folder_path]

        names_mf_map: dict[str, dict[int, MF]] = {}
        names_componenttype_map: dict[str, ComponentType] = {}
        names_tstype_map: dict[str, Optional[TimestepType]] = {}

        for folder in mf_folders:
            h5_files = [i for i in os.listdir(folder) if i in file_names_maps]
            for h5_file in h5_files:
                h5_file_path = os.path.join(folder, h5_file)
                mfs = MF.from_hdf(h5_file_path)

                for mf in mfs:
                    if mf.component_name not in names_mf_map:
                        names_mf_map[mf.component_name] = {}

                    names_mf_map[mf.component_name][mf.step] = mf

        for component_name, mfs_dict in names_mf_map.items():
            first_mf: MF = mfs_dict[0]
            names_componenttype_map[component_name] = first_mf.component_type
            index_names = first_mf.get_index_names()

            time_index = set(index_names).intersection(set(time_steps_map.keys()))

            timestep_name = time_index.pop() if len(time_index) > 0 else ""
            timestep_type = time_steps_map.get(timestep_name, None)
            names_tstype_map[component_name] = timestep_type

        ans: dict[str, "AbstractComponent"] = {
            name: Component(
                name, mf_steps, names_componenttype_map[name], names_tstype_map[name]
            )
            for name, mf_steps in names_mf_map.items()
        }

        return ans


class Scenario(AbstractScenario):
    def __init__(self, path: str) -> None:
        self.path = path
        self._analysis: Analysis = self._read_analysis()
        self._system: System = self._read_system()
        self._components = self._read_components()

    def _read_analysis(self) -> Analysis:
        analysis_path = os.path.join(self.path, "analysis.json")

        if os.path.exists(analysis_path):
            with open(analysis_path, "r") as f:
                return Analysis(**json.load(f))
        return Analysis()

    def _read_system(self) -> System:
        system_path = os.path.join(self.path, "system.json")

        if os.path.exists(system_path):
            with open(system_path, "r") as f:
                return System(**json.load(f))
        return System()

    def _read_components(self) -> dict[str, AbstractComponent]:
        return Component.from_scenario_folder(self.path)

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    @property
    def system(self) -> System:
        return self._system

    @property
    def components(self) -> dict[str, AbstractComponent]:
        return self._components


class MultiHdfLoader(AbstractLoader):
    def __init__(self, path: str) -> None:
        self.path = path
        self._scenarios: dict[str, AbstractScenario] = self._read_scenarios()
        self._time_steps_year2operation_cache: dict[
            str, pd.Series[Any] | pd.DataFrame
        ] = {}

    @property
    def scenarios(self) -> dict[str, AbstractScenario]:
        return self._scenarios

    def _read_scenarios(self) -> dict[str, AbstractScenario]:
        scenarios_json_path = os.path.join(self.path, "scenarios.json")
        ans: dict[str, AbstractScenario] = {}

        with open(scenarios_json_path, "r") as f:
            scenarios_config = json.load(f)

        if len(scenarios_config) == 1:
            scenario_name = "none"
            scenario_path = self.path
            ans[scenario_name] = Scenario(scenario_path)
        else:
            for scenario_id in scenarios_config:
                scenario_name = "scenario_" + scenario_id
                scenario_path = os.path.join(self.path, scenario_name)
                ans[scenario_name] = Scenario(scenario_path)

        return ans

    def get_time_steps_year2operation(self, year: int, is_storage: bool) -> Any:
        cache_key = str(year) + str(is_storage)

        if cache_key in self._time_steps_year2operation_cache:
            return self._time_steps_year2operation_cache[cache_key]

        file_path = os.path.join(self.path, "dict_all_sequence_time_steps.h5")
        h5_file = h5py.File(file_path)

        ts_name = "time_steps_year2operation"
        if ts_name in h5_file:
            # select any element for year2operation
            if is_storage:
                element = list(
                    filter(lambda x: "_storage_level" in x, h5_file[ts_name].keys())
                )[0]
            else:
                element = next(iter(h5_file[ts_name].keys()))

            relevant_group = h5_file[ts_name][element][str(year)]
            self._time_steps_year2operation_cache[cache_key] = pd.read_hdf(
                relevant_group.file.filename, relevant_group.name
            )

        return self._time_steps_year2operation_cache[cache_key]

    @property
    def component_names(self) -> list[str]:
        first_scenario = next(iter(self.scenarios.values()))
        return [component for component in first_scenario.components]
