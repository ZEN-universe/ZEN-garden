import pandas as pd
from typing import Optional, Any, Literal
from zen_garden.postprocess.results.solution_loader import (
    SolutionLoader,
    Scenario,
    Component,
    TimestepType,
    ComponentType,
)
from zen_garden.postprocess.results.multi_hdf_loader import MultiHdfLoader
from functools import cache
from zen_garden.model.default_config import Config,Analysis,Solver,System
import importlib
import os
import logging
import json
from pathlib import Path

class Results:
    def __init__(self, path: str):
        self.solution_loader: SolutionLoader = MultiHdfLoader(path)
        self.has_scenarios = len(self.solution_loader.scenarios) > 1
        self.has_rh = self.solution_loader.has_rh
        first_scenario = next(iter(self.solution_loader.scenarios.values()))
        self.name = Path(first_scenario.analysis.dataset).name

    def __str__(self):
        first_scenario = next(iter(self.solution_loader.scenarios.values()))
        return f"Results of '{first_scenario.analysis.dataset}'"

    def get_df(
        self, component_name: str, scenario_name: Optional[str] = None, data_type: Literal["dataframe","units"] = "dataframe"
    ) -> Optional[dict[str, "pd.DataFrame | pd.Series[Any]"]]:
        """
        Transforms a parameter or variable dataframe (compressed) string into an actual pandas dataframe

        :component_name string: The string to decode
        :scenario_name: Which scenario to take. If none is specified, all are returned.
        :return: The corresponding dataframe
        """
        component = self.solution_loader.components[component_name]

        if data_type == "units" and not component.has_units:
            return None

        scenario_names = (
            self.solution_loader.scenarios.keys()
            if scenario_name is None
            else [scenario_name]
        )

        ans = {}

        for scenario_name in scenario_names:
            scenario = self.solution_loader.scenarios[scenario_name]
            ans[scenario_name] = self.solution_loader.get_component_data(
                scenario, component, data_type=data_type
            )

        return ans

    def get_full_ts_per_scenario(
        self,
        scenario: Scenario,
        component: Component,
        year: Optional[int] = None,
        discount_to_first_step: bool = True,
        element_name: Optional[str] = None,
        keep_raw: Optional[bool] = False,
    ) -> "pd.Series[Any]":
        """Calculates the full timeseries for a given element per scenario

        :param scenario: The scenario for with the component should be extracted (only if needed)
        :param component: Component for the Series
        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :param year: year of which full time series is selected
        :param element_name: Filter results by a given element
        :param keep_raw: Keep the raw values of the rolling horizon optimization
        """
        assert component.timestep_type is not None
        series = self.solution_loader.get_component_data(scenario, component, keep_raw=keep_raw)

        if element_name is not None and element_name in series.index.get_level_values(0):
            series = series.loc[element_name]

        if year is None:
            years = [i for i in range(0, scenario.system.optimized_years)]
        else:
            years = [year]

        if isinstance(series.index, pd.MultiIndex):
            series = series.unstack(component.timestep_name)

        if component.timestep_type is TimestepType.yearly:
            if component.component_type not in [
                ComponentType.parameter,
                ComponentType.variable,
            ]:
                annuity = self._get_annuity(scenario, discount_to_first_step)
                ans = series / annuity
            else:
                ans = series

            try:
                ans = ans[years]
            except KeyError:
                pass

            return ans

        sequence_timesteps = self.solution_loader.get_sequence_time_steps(
            scenario, component.timestep_type
        )

        if (
            component.component_type is ComponentType.dual
            and component.timestep_type is not None
        ):
            timestep_duration = self.solution_loader.get_timestep_duration(
                scenario, component
            )

            annuity = self._get_annuity(scenario)
            series = series.div(timestep_duration, axis=1)

            for year_temp in annuity.index:
                time_steps_year = self.solution_loader.get_timesteps_of_year(
                    scenario, component.timestep_type, year_temp
                )
                series[time_steps_year] = series[time_steps_year] / annuity[year_temp]

        try:
            output_df = series[sequence_timesteps]
        except KeyError:
            output_df = series

        output_df = output_df.T.reset_index(drop=True).T

        if year is not None:
            _total_hours_per_year = scenario.system.unaggregated_time_steps_per_year

            hours_of_year = list(
                range(year * _total_hours_per_year, (year + 1) * _total_hours_per_year)
            )

            output_df = output_df[hours_of_year]

        return output_df

    def get_full_ts(
        self,
        component_name: str,
        scenario_name: Optional[str] = None,
        discount_to_first_step: bool = True,
        year: Optional[int] = None,
        element_name: Optional[str] = None,
        keep_raw: Optional[bool] = False,
    ) -> "pd.DataFrame | pd.Series[Any]":
        """Calculates the full timeseries for a given element

        :param component_name: Name of the component
        :param scenario_name: The scenario for with the component should be extracted (only if needed)
        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :param year: year of which full time series is selected
        :param element_name: Filter results by a given element
        :param keep_raw: Keep the raw values of the rolling horizon optimization
        """
        if scenario_name is None:
            scenario_names = list(self.solution_loader.scenarios)
        else:
            scenario_names = [scenario_name]

        component = self.solution_loader.components[component_name]

        scenarios_dict: dict[str, "pd.DataFrame | pd.Series[Any]"] = {}

        for scenario_name in scenario_names:
            scenario = self.solution_loader.scenarios[scenario_name]

            scenarios_dict[scenario_name] = self.get_full_ts_per_scenario(
                scenario,
                component,
                discount_to_first_step=discount_to_first_step,
                year=year,
                element_name=element_name,
                keep_raw=keep_raw,
            )

        return self._concat_scenarios_dict(scenarios_dict)

    @cache
    def get_total_per_scenario(
        self,
        scenario: Scenario,
        component: Component,
        element_name: Optional[str] = None,
        year: Optional[int] = None,
        keep_raw: Optional[bool] = False,
    ) -> "pd.DataFrame | pd.Series[Any]":
        """
        Calculates the total values of a component for a specific scenario.

        :param scenario: Scenario
        :param component: Component
        :param element_name: Filter the results by a given element
        :param year: Filter the results by a given year
        :param keep_raw: Keep the raw values of the rolling horizon optimization
        """
        series = self.solution_loader.get_component_data(scenario, component, keep_raw)

        if year is None:
            years = [i for i in range(0, scenario.system.optimized_years)]
        else:
            years = [year]

        if element_name is not None:
            series = series.loc[element_name]

        if component.timestep_type is None or type(series.index) is not pd.MultiIndex:
            return series

        if component.timestep_type is TimestepType.yearly:
            ans = series.unstack(component.timestep_name)
            return ans[years]

        timestep_duration = self.solution_loader.get_timestep_duration(
            scenario, component
        )

        unstacked_series = series.unstack(component.timestep_name)
        total_value = unstacked_series.multiply(timestep_duration, axis=1)  # type: ignore

        ans = pd.DataFrame(index=unstacked_series.index)

        for y in years:
            timesteps = self.solution_loader.get_timesteps(scenario, component, int(y))
            try:
                ans.insert(len(ans.columns), y, total_value[timesteps].sum(axis=1,skipna=False))  # type: ignore
            except KeyError:
                timestep_list = [i for i in timesteps if i in total_value]
                ans.insert(len(ans.columns), year, total_value[timestep_list].sum(axis=1,skipna=False))  # type: ignore # noqa

        if "mf" in ans.index.names:
            ans = ans.reorder_levels([i for i in ans.index.names if i != "mf"] + ["mf"]).sort_index(axis=0)

        return ans

    @cache
    def get_total(
        self,
        component_name: str,
        element_name: Optional[str] = None,
        year: Optional[int] = None,
        scenario_name: Optional[str] = None,
        keep_raw: Optional[bool] = False,
    ) -> "pd.DataFrame | pd.Series[Any]":
        """
        Calculates the total values of a component for a all scenarios.

        :param component_name: Name of the component
        :param element_name: Filter the results by a given element
        :param year: Filter the results by a given year
        :param scenario_name: Filter the results by a given scenario
        :param keep_raw: Keep the raw values of the rolling horizon optimization
        """
        if scenario_name is None:
            scenario_names = list(self.solution_loader.scenarios)
        else:
            scenario_names = [scenario_name]

        component = self.solution_loader.components[component_name]

        scenarios_dict: dict[str, "pd.DataFrame | pd.Series[Any]"] = {}

        for scenario_name in scenario_names:
            scenario = self.solution_loader.scenarios[scenario_name]
            current_total = self.get_total_per_scenario(
                scenario, component, element_name, year, keep_raw
            )

            if type(current_total) is pd.Series:
                current_total = current_total.rename(component_name)

            scenarios_dict[scenario_name] = current_total

        return self._concat_scenarios_dict(scenarios_dict)

    def _concat_scenarios_dict(
        self, scenarios_dict: dict[str, "pd.DataFrame | pd.Series[Any]"]
    ) -> pd.DataFrame:
        """
        Concatenates a dict of the form str: Data to one dataframe.

        :param scenarios_dict: Dict containing the scenario names as key and the values as values.
        """
        scenario_names = list(scenarios_dict.keys())

        if len(scenario_names) == 1:
            ans = scenarios_dict[scenario_names[0]]
            return ans

        if isinstance(scenarios_dict[scenario_names[0]], pd.Series):
            total_value = pd.concat(
                scenarios_dict, keys=scenarios_dict.keys(), axis=1
            ).T
        else:
            try:
                total_value = pd.concat(scenarios_dict, keys=scenarios_dict.keys())  # type: ignore # noqa
            except Exception:
                total_value = pd.concat(
                    scenarios_dict, keys=scenarios_dict.keys(), axis=1
                ).T
        total_value.index.names = ["scenario"]+list(total_value.index.names[1:])
        return total_value

    def _get_annuity(self, scenario: Scenario, discount_to_first_step: bool = True):
        """discounts the duals

        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :param scenario: scenario name whose results are assessed
        :return: #TODO describe parameter/return
        """
        system = scenario.system
        discount_rate_component = self.solution_loader.components["discount_rate"]
        # calculate annuity
        discount_rate = self.solution_loader.get_component_data(
            scenario, discount_rate_component
        ).squeeze()

        years = list(range(0, system["optimized_years"]))

        annuity = pd.Series(index=years, dtype=float)
        for year in years:
            interval_between_years = system.interval_between_years
            if year == years[-1]:
                interval_between_years_this_year = 1
            else:
                interval_between_years_this_year = system.interval_between_years
            if self.solution_loader.has_rh:
                if discount_to_first_step:
                    annuity[year] = interval_between_years_this_year * (
                        1 / (1 + discount_rate)
                    )
                else:
                    annuity[year] = sum(
                        ((1 / (1 + discount_rate)) ** (_intermediate_time_step))
                        for _intermediate_time_step in range(
                            0, interval_between_years_this_year
                        )
                    )
            else:
                if discount_to_first_step:
                    annuity[year] = interval_between_years_this_year * (
                        (1 / (1 + discount_rate))
                        ** (interval_between_years * (year - years[0]))
                    )
                else:
                    annuity[year] = sum(
                        (
                            (1 / (1 + discount_rate))
                            ** (
                                interval_between_years * (year - years[0])
                                + _intermediate_time_step
                            )
                        )
                        for _intermediate_time_step in range(
                            0, interval_between_years_this_year
                        )
                    )
        return annuity

    def get_dual(
        self,
        constraint: str,
        scenario_name: Optional[str] = None,
        element_name: Optional[str] = None,
        year: Optional[int] = None,
        discount_to_first_step=True,
        keep_raw: Optional[bool] = False,
    ) -> Optional["pd.DataFrame | pd.Series[Any]"]:
        """extracts the dual variables of a constraint

        :param constraint: Name of dal
        :param scenario_name: Scenario Name
        :param element_name: Name of Element
        :param year: Year
        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :param keep_raw: Keep the raw values of the rolling horizon optimization
        """
        if not self.get_solver(scenario_name=scenario_name).add_duals:
            logging.warning("Duals are not calculated. Skip.")
            return None

        component = self.solution_loader.components[constraint]
        assert (
            component.component_type is ComponentType.dual
        ), "Given constraint name is not of type Dual."

        _duals = self.get_full_ts(
            component_name=constraint,
            scenario_name=scenario_name,
            element_name=element_name,
            year=year,
            discount_to_first_step=discount_to_first_step,
            keep_raw=keep_raw,
        )
        return _duals

    def get_unit(self, component_name: str, scenario_name: Optional[str] = None,droplevel:bool=True) -> Optional[dict[str, "pd.DataFrame | pd.Series[Any]"]]:
        """
        Extracts the unit of a given Component. If no scenario is given, a random one is taken.

        :param component_name: Name of the component
        :param scenario_name: Name of the scenario
        :param droplevel: Drop the location and time levels of the multiindex
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        res = self.get_df(component_name, scenario_name=scenario_name, data_type="units")
        if res is None:
            return None
        units = res[scenario_name]
        if droplevel:
            # TODO make more flexible
            loc_idx = ["set_nodes","set_location","set_edges"]
            time_idx = ["set_time_steps_yearly","set_time_steps_operation","set_time_steps_storage"]
            drop_idx = pd.Index(loc_idx+time_idx).intersection(units.index.names)
            units.index = units.index.droplevel(drop_idx.to_list())
            units = units[~units.index.duplicated()]
        return units

    def get_system(self, scenario_name: Optional[str] = None) -> System:
        """
        Extracts the System config of a given Scenario. If no scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        return self.solution_loader.scenarios[scenario_name].system

    def get_analysis(self, scenario_name: Optional[str] = None) -> Analysis:
        """
        Extracts the Analysis config of a given Scenario. If no scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        return self.solution_loader.scenarios[scenario_name].analysis

    def get_solver(self, scenario_name: Optional[str] = None) -> Solver:
        """
        Extracts the Solver config of a given Scenario. If no scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        return self.solution_loader.scenarios[scenario_name].solver

    def get_doc(self, component: str) -> str:
        """
        Extracts the documentation of a given Component.

        :param component: Name of the component
        """
        return self.solution_loader.components[component].doc

    def get_years(self, scenario_name: Optional[str] = None) -> list[int]:
        """
        Extracts the years of a given Scenario. If no scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        system = self.solution_loader.scenarios[scenario_name].system
        years = list(range(0, system.optimized_years))
        return years

    def has_MF(self, scenario_name: Optional[str] = None):
        """
        Extracts the System config of a given Scenario. If no scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        scenario = self.solution_loader.scenarios[scenario_name]
        return scenario.system.use_rolling_horizon


if __name__ == "__main__":
    try:
        spec = importlib.util.spec_from_file_location("module", "config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.config
    except FileNotFoundError:
        with open("config.json") as f:
            config = Config(json.load(f))

    model_name = os.path.basename(config.analysis["dataset"])
    if os.path.exists(
        out_folder := os.path.join(config.analysis["folder_output"], model_name)
    ):
        r = Results(out_folder)
    else:
        logging.critical("No results folder found!")
