import pandas as pd
from typing import Optional, Any
from zen_garden.postprocess.new_results.solution_loader import SolutionLoader
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

    def get_aggregated_ts(
        self, component_name: str, scenario_name: Optional[str] = None
    ) -> dict[str, "pd.Series[Any]"]:
        scenarios = self.solution_loader.scenarios

        if scenario_name is None:
            scenario_names_to_include = [i for i in scenarios]
        else:
            assert scenario_name in scenarios
            scenario_names_to_include = [scenario_name]

        ans: dict[str, pd.Series[Any]] = {}

        for scenario_name in scenario_names_to_include:
            scenario = scenarios[scenario_name]
            component = scenario.components[component_name]
            scenario_series = component.get_mf_aggregated_series()
            ans[scenario_name] = scenario_series

        return ans

    def get_full_ts(self) -> None:
        pass

    def get_total(
        self,
        component_name: str,
        element_name: Optional[str] = None,
        year: Optional[int] = None,
    ) -> Any:
        scenario = self.solution_loader.scenarios["none"]
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
                # Weird behaviour
                return series.to_frame()

        unstacked_series = series.unstack(component.timestep_name)

        timestep_duration = scenario.get_timestep_duration(
            component.timestep_type
        ).unstack()

        if isinstance(unstacked_series.index, pd.MultiIndex):
            total_value = unstacked_series.apply(
                lambda row: row * timestep_duration.loc[row.name[0]], axis=1
            )
        else:
            total_value = unstacked_series.apply(
                lambda row: row * timestep_duration.loc[row.name], axis=1
            )

        for year in years:
            timesteps = self.solution_loader.get_time_steps_year2operation(
                year, is_storage=component.timestep_type.value == "time_storage_level"
            )
            total_value_series: pd.Series[Any] = total_value[timesteps].sum(axis=1)

        return total_value_series.to_frame()
