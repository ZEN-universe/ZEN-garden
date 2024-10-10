"""
This module contains the implementation of a SolutionLoader that reads the solution.
"""
import re

from zen_garden.postprocess.results.solution_loader import (
    Component as AbstractComponent,
    Scenario as AbstractScenario,
    SolutionLoader as AbstractLoader,
    ComponentType,
    TimestepType,
)

from zen_garden.model.default_config import Analysis, System, Solver
import json
import os
import h5py  # type: ignore
from typing import Optional, Any,Literal
import pandas as pd
import numpy as np
from functools import cache

file_names_maps = {
    "param_dict.h5": ComponentType.parameter,
    "var_dict.h5": ComponentType.variable,
    "set_dict.h5": ComponentType.sets,
    "dual_dict.h5": ComponentType.dual,
}

time_steps_map: dict[str | None, TimestepType] = {
    "year": TimestepType.yearly,
    "time_operation": TimestepType.operational,
    "time_storage_level": TimestepType.storage,
}


def get_index_names(h5_group: h5py.Group) -> list[str]:
    """
    Helper-function that returns the pandas dataframe index names of a h5-Group
    """
    ans = []

    for val in h5_group.values():
        name = None
        try:
            name = val.attrs["name"].decode()
        except KeyError:
            continue

        if name != "N.":
            ans.append(name)

    return ans


@cache
def get_df_from_path(path: str, component_name: str, data_type: Literal["dataframe","units"] = "dataframe") -> "pd.Series[Any]":
    """
    Helper-function that returns a Pandas series given the path of a file and the
    component name.
    """
    pd_read = pd.read_hdf(path, component_name + f"/{data_type}")

    if isinstance(pd_read, pd.DataFrame):
        ans = pd_read.squeeze()
    elif isinstance(pd_read, pd.Series):
        ans = pd_read

    if isinstance(ans, (np.float_, str)):
        ans = pd.Series([ans], index=pd_read.index)

    assert type(ans) is pd.Series

    return ans


class Component(AbstractComponent):
    """
    Implementation of the abstract component. It uses the
    globally defined time_steps_map to derive the timestep-name of a component given its
    type.
    """

    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        ts_type: Optional[TimestepType],
        ts_name: Optional[str],
        file_name: str,
        doc: str,
        has_units: bool
    ) -> None:
        self._component_type = component_type
        self._name = name
        self._ts_type = ts_type
        self._file_name = file_name
        self._ts_name = ts_name
        self._doc = doc
        self._has_units = has_units

    @property
    def component_type(self) -> ComponentType:
        return self._component_type

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

class Scenario(AbstractScenario):
    """
    Implementation of the abstract scenario. In this solution version, the analysis and
    system configs are stored as jsons for each of the scenario in the corresponding
    folder.
    """

    def __init__(self, path: str, name: str, base_scenario: str) -> None:
        self._path = path
        self._analysis: Analysis = self._read_analysis()
        self._system: System = self._read_system()
        self._solver: Solver = self._read_solver()
        self.name = name
        self.base_name = base_scenario

    def _read_analysis(self) -> Analysis:
        analysis_path = os.path.join(self.path, "analysis.json")

        with open(analysis_path, "r") as f:
            return Analysis(**json.load(f))

    def _read_system(self) -> System:
        system_path = os.path.join(self.path, "system.json")

        with open(system_path, "r") as f:
            return System(**json.load(f))

    def _read_solver(self) -> Solver:
        solver_path = os.path.join(self.path, "solver.json")

        with open(solver_path, "r") as f:
            return Solver(**json.load(f))

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    @property
    def solver(self) -> Solver:
        return self._solver

    @property
    def path(self) -> str:
        return self._path

    @property
    def system(self) -> System:
        return self._system


class MultiHdfLoader(AbstractLoader):
    """
    Implementation of a SolutionLoader.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        assert len(os.listdir(path)) > 0, f"Path {path} is empty."
        self._scenarios: dict[str, AbstractScenario] = self._read_scenarios()
        self._components: dict[str, AbstractComponent] = self._read_components()
        self._series_cache: dict[str, "pd.Series[Any]"] = {}

    @property
    def scenarios(self) -> dict[str, AbstractScenario]:
        return self._scenarios

    @property
    def components(self) -> dict[str, AbstractComponent]:
        return self._components

    @property
    def name(self) -> str:
        scenario = self.scenarios[next(iter(self.scenarios.keys()))]
        name = scenario.analysis.dataset.split("/")[-1]
        return name

    @property
    def has_rh(self) -> bool:
        first_scenario_name = next(iter(self._scenarios.keys()))
        first_scenario = self._scenarios[first_scenario_name]
        return first_scenario.system.use_rolling_horizon

    @property
    def has_duals(self) -> bool:
        first_scenario_name = next(iter(self._scenarios.keys()))
        first_scenario = self._scenarios[first_scenario_name]
        return first_scenario.system.use_duals

    def _combine_dataseries(
        self,
        component: AbstractComponent,
        scenario: AbstractScenario,
        pd_dict: dict[int, "pd.Series[Any]"],
    ) -> "pd.DataFrame | pd.Series[Any]":
        """
        Method that combines the values when a solution is created without perfect
        foresight given a component, a scenario and a dictionary containing the name of
        the MF-data (Format: "MF_{year}").
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
            if component.timestep_type is TimestepType.yearly:
                year_series = current_mf[
                    current_mf.index.get_level_values("year").isin(decision_horizon)
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

        return pd.concat(series_to_concat)

    def _concatenate_raw_dataseries(
        self,
        pd_dict: dict[int, "pd.Series[Any]"],
    ) -> "pd.DataFrame | pd.Series[Any]":
        """
        Method that concatenates the raw values when a solution is created without perfect
        foresight given a component, a scenario and a dictionary containing the name of
        the MF-data (Format: "MF_{year}"). The raw values are not combined, i.e.,
        the data is kept for all the foresight steps.
        """
        series = pd.concat(pd_dict, keys=pd_dict.keys())
        series = series.sort_index(level=0)
        index_names = pd_dict[list(pd_dict.keys())[0]].index.names
        new_index_names = ["mf"] + index_names
        series.index.names = new_index_names
        return series

    def get_component_data(
        self,
        scenario: AbstractScenario,
        component: AbstractComponent,
        keep_raw: bool = False,
        data_type: Literal["dataframe","units"] = "dataframe"
    ) -> "pd.DataFrame | pd.Series[Any]":
        """
        Implementation of the abstract method. Returns the actual component values given
        a component and a scenario. Already combines the yearly data if the solution does
        not use perfect foresight, unless explicitly desired otherwise (keep_raw = True).
        """
        if self.has_rh:
            # If solution has rolling horizon, load the values for all the foresight
            # steps and combine them.
            pattern = re.compile(r'^MF_\d+$')
            subfolder_names = list(filter(lambda x: pattern.match(x), os.listdir(scenario.path)))
            pd_series_dict = {}

            for subfolder_name in subfolder_names:
                mf_idx = int(subfolder_name.replace("MF_", ""))
                file_path = os.path.join(
                    scenario.path, subfolder_name, component.file_name
                )
                pd_series_dict[mf_idx] = get_df_from_path(
                    file_path, component.name, data_type
                )
            if not keep_raw:
                combined_dataseries = self._combine_dataseries(
                    component, scenario, pd_series_dict
                )
            else:
                combined_dataseries = self._concatenate_raw_dataseries(
                    pd_series_dict
                )
            return combined_dataseries
        else:
            # If solution does not use rolling horizon, simply load the HDF file.
            file_path = os.path.join(scenario.path, component.file_name)
            ans = get_df_from_path(file_path, component.name,data_type)
            return ans

    def _read_scenarios(self) -> dict[str, AbstractScenario]:
        """
        Create the scenario instances. The definitions of the scenarios are stored in the
        scenarios.json files. If the solution does not have multiple scenarios, we store
        the solution as "none".
        """
        scenarios_json_path = os.path.join(self.path, "scenarios.json")
        ans: dict[str, AbstractScenario] = {}

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

                # Some scenarios have additional parameter definitions that are stored in
                # subfolders.
                scenario_subfolder = scenario_config["sub_folder"]

                if scenario_subfolder != "":
                    scenario_path = os.path.join(
                        scenario_path, f"scenario_{scenario_subfolder}"
                    )

                ans[scenario_name] = Scenario(
                    scenario_path, scenario_name, base_scenario
                )

        return ans

    def _read_components(self) -> dict[str, AbstractComponent]:
        """
        Create the component instances.

        The components are stored in three files and the file-names define the types of
        the component. This correspondence is stored in the global variable
        file_names_maps. Furthermore, the timestep name and type are derived by checking
        if any of the defined time steps name is in the index of the dataframe.
        """
        ans: dict[str, AbstractComponent] = {}
        first_scenario_name = next(iter(self._scenarios.keys()))
        first_scenario = self._scenarios[first_scenario_name]

        if self.has_rh:
            mf_name = [i for i in os.listdir(first_scenario.path) if "MF_" in i][0]
            component_folder = os.path.join(first_scenario.path, mf_name)
        else:
            component_folder = first_scenario.path

        for file_name, component_type in file_names_maps.items():
            file_path = os.path.join(component_folder, file_name)

            if not os.path.exists(file_path):
                continue

            h5_file = h5py.File(file_path)
            for component_name in h5_file.keys():
                index_names = get_index_names(h5_file[component_name + "/dataframe"])
                time_index = set(index_names).intersection(set(time_steps_map.keys()))
                timestep_name = time_index.pop() if len(time_index) > 0 else None
                timestep_type = time_steps_map.get(timestep_name, None)

                doc = str(np.char.decode(h5_file[component_name + "/docstring"].attrs.get("value")))
                if ";" in doc and ":" in doc:
                    doc = '\n'.join([f'{v.split(":")[0]}: {v.split(":")[1]}' for v in doc.split(";")])

                has_units = "units" in h5_file[component_name]

                ans[component_name] = Component(
                    component_name,
                    component_type,
                    timestep_type,
                    timestep_name,
                    file_name,
                    doc,
                    has_units
                )

        return ans

    @cache
    def get_timestep_duration(
        self, scenario: AbstractScenario, component: AbstractComponent
    ) -> "pd.Series[Any]":
        """
        The timestep duration is stored as any other component, the only thing is to
        define the correct name depending on the component timestep type.
        """
        if component.timestep_type is TimestepType.operational:
            timestep_duration_name = "time_steps_operation_duration"
        else:
            timestep_duration_name = "time_steps_storage_duration"

        time_step_duration = self.get_component_data(
            scenario, self.components[timestep_duration_name]
        )
        assert type(time_step_duration) is pd.Series

        return time_step_duration

    @cache
    def get_timesteps(
        self, scenario: AbstractScenario, component: AbstractComponent, year: int
    ) -> "pd.Series[Any]":
        """
        THe timesteps are stored in a file HDF-File called dict_all_sequence_time_steps
        saved for each scenario. The name of the dataframe depends on the timestep type.
        """
        time_steps_file_name = [
            i
            for i in os.listdir(scenario.path)
            if "dict_all_sequence_time_steps" in i and ".lock" not in i
        ][0]

        dict_path = os.path.join(
            scenario.path,
            time_steps_file_name,
        )

        timesteps_name = (
            "time_steps_year2operation"
            if component.timestep_type is TimestepType.operational
            else "time_steps_year2storage"
        )

        ans = pd.read_hdf(dict_path, f"{timesteps_name}/{year}")

        assert type(ans) is pd.Series

        return ans

    @cache
    def get_timesteps_of_years(
        self, scenario: Scenario, ts_type: TimestepType, years: tuple
    ) -> "pd.DataFrame | pd.Series[Any]":
        """
        Method that returns the timesteps of the scenario for a given year. These
        timesteps are stored as HDF files that were created by pandas.
        """
        sequence_time_steps_name = [
            i
            for i in os.listdir(scenario.path)
            if "dict_all_sequence_time_steps" in i and ".lock" not in i
        ][0]

        time_step_path = os.path.join(scenario.path, sequence_time_steps_name)

        time_step_file = h5py.File(time_step_path)

        if ts_type is TimestepType.storage:
            time_step_name = "time_steps_year2storage"
        elif ts_type is TimestepType.operational:
            time_step_name = "time_steps_year2operation"

        time_step_yearly = time_step_file[time_step_name]

        time_steps = []
        for year in years:
            year_series = time_step_yearly[str(year)]
            time_steps.append(pd.read_hdf(time_step_path, year_series.name))
        time_steps = pd.concat(time_steps).reset_index(drop=True)
        return time_steps

    def get_sequence_time_steps(
        self, scenario: AbstractScenario, timestep_type: TimestepType
    ) -> "pd.Series[Any]":
        time_steps_file_name = [
            i
            for i in os.listdir(scenario.path)
            if "dict_all_sequence_time_steps" in i and ".lock" not in i
        ][0]

        dict_path = os.path.join(
            scenario.path,
            time_steps_file_name,
        )

        if timestep_type is TimestepType.operational:
            sequence_timesteps_name = "operation"
        elif timestep_type is TimestepType.storage:
            sequence_timesteps_name = "storage"
        else:
            sequence_timesteps_name = "yearly"

        ans = pd.read_hdf(dict_path, sequence_timesteps_name)

        return ans

    def get_optimized_years(
            self,scenario: AbstractScenario
    ) -> list[int]:
        """
        Method that returns the years for which the solution was optimized.
        """
        time_steps_file_name = [
            i
            for i in os.listdir(scenario.path)
            if "dict_all_sequence_time_steps" in i and ".lock" not in i
        ][0]

        dict_path = os.path.join(
            scenario.path,
            time_steps_file_name,
        )
        try:
            ans = pd.read_hdf(dict_path, "optimized_time_steps").tolist()
        # if old version of the solution
        except:
            if self.has_rh:
                pattern = re.compile(r'^MF_\d+$')
                subfolder_names = list(filter(lambda x: pattern.match(x), os.listdir(scenario.path)))
                ans = [int(subfolder_name.replace("MF_", "")) for subfolder_name in subfolder_names]
            else: # if no rolling horizon, single optimized year
                ans = [0]

        return ans

