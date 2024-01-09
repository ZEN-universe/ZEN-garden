from pandas.core.api import Series as Series
from zen_garden.postprocess.new_results.solution_loader import (
    Component as AbstractComponent,
    Scenario as AbstractScenario,
    SolutionLoader as AbstractLoader,
    ComponentType,
    TimestepType,
)

from zen_garden.model.default_config import Analysis, System
import json
import os
import h5py  # type: ignore
from typing import Optional
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


def get_index_names(h5_group: h5py.Group) -> list[str]:
    ans = []

    for key, val in h5_group.items():
        if not key.startswith("axis"):
            continue
        try:
            name = val.attrs["name"].decode()
        except KeyError:
            continue

        if name != "N.":
            ans.append(name)

    return ans


def get_df_form_path(path, component_name) -> pd.Series:
    pd_read = pd.read_hdf(path, component_name + "/dataframe")

    if isinstance(pd_read, pd.DataFrame):
        ans = pd_read.squeeze()
    elif isinstance(pd_read, pd.Series):
        ans = pd_read
    if isinstance(ans, (np.float_, str)):
        ans = pd.Series([ans], index=pd_read.index)
    return ans


class Component(AbstractComponent):
    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        ts_type: Optional[TimestepType],
        file_name: str,
    ) -> None:
        self._component_type = component_type
        self.name = name
        self._ts_type = ts_type
        self.file_name = file_name

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
    def timestep_type(self) -> Optional[TimestepType]:
        return self._ts_type


class Scenario(AbstractScenario):
    def __init__(self, path: str) -> None:
        self.path = path
        self._analysis: Analysis = self._read_analysis()
        self._system: System = self._read_system()

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

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    @property
    def system(self) -> System:
        return self._system

    def get_time_steps_of_year(self, ts_type: TimestepType, year: int):
        tech_proxy = self.system.set_storage_technologies[0]
        time_step_path = os.path.join(self.path, "dict_all_sequence_time_steps.h5")
        time_step_file = h5py.File(time_step_path)

        if ts_type is TimestepType.storage:
            tech_proxy = tech_proxy + "_storage_level"
            time_step_name = "time_steps_year2operation"
        elif ts_type is TimestepType.operational:
            time_step_name = "time_steps_year2operation"

        time_step_yearly = time_step_file[time_step_name]

        if tech_proxy in time_step_yearly:
            time_step_yearly = time_step_yearly[tech_proxy]

        year_series = time_step_yearly[str(year)]

        return pd.read_hdf(time_step_path, year_series.name)


class MultiHdfLoader(AbstractLoader):
    def __init__(self, path: str) -> None:
        self.path = path
        self._scenarios: dict[str, Scenario] = self._read_scenarios()
        self._components: dict[str, AbstractComponent] = self._read_components()
        self._series_cache: dict[str, pd.Series] = {}

    @property
    def scenarios(self) -> dict[str, Scenario]:
        return self._scenarios

    @property
    def components(self) -> list[str]:
        return self._components

    @property
    def has_rh(self):
        first_scenario_name = next(iter(self._scenarios.keys()))
        first_scenario = self._scenarios[first_scenario_name]
        return first_scenario.system.use_rolling_horizon

    def combine_dataseries(
        self, component: Component, scenario: Scenario, pd_dict: dict[str, pd.Series]
    ):
        series_to_concat = []
        n_years = len(pd_dict)

        for year in range(n_years):
            current_mf = pd_dict[f"MF_{year}"]
            if component.timestep_type is TimestepType.yearly:
                year_series = current_mf[
                    current_mf.index.get_level_values("year") == year
                ]
                series_to_concat.append(year_series)
            elif component.timestep_type in [
                TimestepType.operational,
                TimestepType.storage,
            ]:
                time_level_name = (
                    "time_operation"
                    if component.timestep_type is TimestepType.operational
                    else "time_storage_level"
                )
                time_steps = scenario.get_time_steps_of_year(
                    component.timestep_type, year
                )
                time_step_list = {tstep for tstep in time_steps}
                year_series = current_mf[
                    [
                        i in time_step_list
                        for i in current_mf.index.get_level_values(time_level_name)
                    ]
                ]
                series_to_concat.append(year_series)
            else:
                series_to_concat.append(current_mf)

        return pd.concat(series_to_concat)

    def get_component_data(self, scenario: Scenario, component: Component) -> Series:
        if self.has_rh:
            subfolder_names = [i for i in os.listdir(scenario.path) if "MF_" in i]
            pd_series_dict = {}

            for subfolder_name in subfolder_names:
                file_path = os.path.join(
                    scenario.path, subfolder_name, component.file_name
                )
                pd_series_dict[subfolder_name] = get_df_form_path(
                    file_path, component.name
                )

            return self.combine_dataseries(component, scenario, pd_series_dict)
        else:
            file_path = os.path.join(scenario.path, component.file_name)
            return get_df_form_path(file_path, component.name)

    def _read_scenarios(self) -> dict[str, AbstractScenario]:
        scenarios_json_path = os.path.join(self.path, "scenarios.json")
        ans: dict[str, AbstractScenario] = {}

        with open(scenarios_json_path, "r") as f:
            scenario_configs = json.load(f)

        if len(scenario_configs) == 1:
            scenario_name = "none"
            scenario_path = self.path
            ans[scenario_name] = Scenario(scenario_path)
        else:
            for scenario_id, scenario_config in scenario_configs.items():
                scenario_name = f"scenario_{scenario_id}"
                scenario_path = os.path.join(
                    self.path, f"scenario_{scenario_config['base_scenario']}"
                )

                scenario_subfolder = scenario_config["sub_folder"]

                if scenario_subfolder != "":
                    scenario_path = os.path.join(
                        scenario_path, f"scenario_{scenario_subfolder}"
                    )

                ans[scenario_name] = Scenario(scenario_path)

        return ans

    def _read_components(self) -> dict[str, Component]:
        ans: dict[str, Component] = {}
        first_scenario_name = next(iter(self._scenarios.keys()))
        first_scenario = self._scenarios[first_scenario_name]

        if self.has_rh:
            mf_name = [i for i in os.listdir(first_scenario.path) if "MF_" in i][0]
            component_folder = os.path.join(first_scenario.path, mf_name)
        else:
            component_folder = first_scenario.path

        for file_name, component_type in file_names_maps.items():
            file_path = os.path.join(component_folder, file_name)
            h5_file = h5py.File(file_path)
            for component_name in h5_file.keys():
                index_names = get_index_names(h5_file[component_name + "/dataframe"])
                time_index = set(index_names).intersection(set(time_steps_map.keys()))
                timestep_name = time_index.pop() if len(time_index) > 0 else ""
                timestep_type = time_steps_map.get(timestep_name, None)
                ans[component_name] = Component(
                    component_name, component_type, timestep_type, file_name
                )

        return ans
