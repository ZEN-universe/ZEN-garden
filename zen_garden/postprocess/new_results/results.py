import pandas as pd
from typing import Optional, Any
from zen_garden.postprocess.new_results.solution_loader import SolutionLoader, Scenario
from zen_garden.postprocess.new_results.multi_hdf_loader import MultiHdfLoader


class Results:
    def __init__(self, path: str):
        self.solution_loader: SolutionLoader = MultiHdfLoader(path)

    def compare_model_parameters(self, other_results: "Results") -> None:
        pass

    def compare_model_variables(self, other_results: "Results") -> None:
        pass

    def compare_component_values(self, other_results: "Results") -> None:
        pass

    def compare_config(self, other_results: "Results") -> None:
        pass

    def get_component_data(
        self, component_name: str, scenario_name: Optional[str] = None
    ) -> dict[str, "pd.Series[Any]"]:
        component = self.solution_loader.components[component_name]

        scenario_names = (
            self.solution_loader.scenarios.keys()
            if scenario_name is None
            else [scenario_name]
        )

        ans = {}

        for scenario_name in scenario_names:
            scenario = self.solution_loader.scenarios[scenario_name]
            ans[scenario_name] = self.solution_loader.get_component_data(scenario, component)

        return ans

    def get_full_ts(self) -> None:
        pass

    def get_total_from_scenario(
        self,
        scenario: Scenario,
        component_name: str,
        element_name: Optional[str] = None,
        year: Optional[int] = None,
    ) -> pd.DataFrame:
        component = scenario.components[component_name]
        series = component.get_mf_aggregated_series()

        years = (
            list(range(0, scenario.system.optimized_years)) if year is None else [year]
        )

        if component.timestep_type is None:
            return series.to_frame()

        if component.timestep_type.value == "year":
            if isinstance(series.index, pd.MultiIndex):
                ans = series.unstack(component.timestep_name)
                return ans[years]
            else:
                return series

        timestep_duration = scenario.get_timestep_duration(component.timestep_type)

        unstacked_series = series.unstack(component.timestep_name)
        total_value = unstacked_series.multiply(timestep_duration, axis=1)

        ans = pd.DataFrame()

        for year in years:
            timesteps = self.solution_loader.get_time_steps_year2operation(
                year, is_storage=component.timestep_type.value == "time_storage_level"
            )
            ans.insert(year, year, total_value[timesteps].sum(axis=1))

        return ans

    def get_total(
        self,
        component_name: str,
        element_name: Optional[str] = None,
        year: Optional[int] = None,
        scenario_name: str = None,
    ) -> Any:
        if scenario_name is None:
            scenario_names = list(self.solution_loader.scenarios)
        else:
            scenario_names = [scenario_name]

        scenarios_dict = {}
        for scenario_name in scenario_names:
            scenario = self.solution_loader.scenarios[scenario_name]
            scenarios_dict[scenario_name] = self.get_total_from_scenario(
                scenario, component_name, element_name, year
            )

        if len(scenario_names) == 1:
            return scenarios_dict[scenario_names[0]]

        if isinstance(scenarios_dict[scenario_name], pd.Series):
            total_value = pd.concat(
                scenarios_dict, keys=scenarios_dict.keys(), axis=1
            ).T
        else:
            total_value = pd.concat(scenarios_dict, keys=scenarios_dict.keys())
        return total_value
