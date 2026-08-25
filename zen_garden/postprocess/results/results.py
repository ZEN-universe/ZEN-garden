"""This module contains the Results class, which is used to extract and process
the results of a model run.
"""
import logging
import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import pandas as pd
from pandas import Series
from typing import Literal

from zen_garden.default_config import Analysis, Solver, System
from zen_garden.postprocess.results.solution_loader import (
    Component,
    ComponentType,
    Scenario,
    SolutionLoader,
    TimestepType,
)
from zen_garden.postprocess.results.cost_emission_calculation import (
    CostEmissionCalculation)
from zen_garden.utils import reformat_slicing_index

logger = logging.getLogger(__name__)

NestedTuple = tuple[list[str], ...] | tuple[str, ...]
NestedDict = dict[str, str | list[str]]

# used for the cost and emission calculation to specify the mode of calculation
CostEmissionMode = Literal["final_demand", "total_production", "relative"]

class Results:
    """The Results class is used to extract and process the results of a model run."""

    def __init__(self, path: str | os.PathLike[str], enable_cache: bool = True):
        """Initializes the Results class.

        :param path: Path to the results folder
        """
        assert os.path.exists(
            path
        ), f"The output folder {Path(path).absolute()} does not exist."
        assert (
            len(os.listdir(path)) > 0
        ), f"The output folder {Path(path).absolute()} is empty."
        self.solution_loader = SolutionLoader(path, enable_cache=enable_cache)
        self.has_scenarios = len(self.solution_loader.scenarios) > 1
        first_scenario = next(iter(self.solution_loader.scenarios.values()))
        self.name = Path(first_scenario.analysis.dataset).name
        self.ureg = first_scenario.ureg
        self.cost_emission_calculation = CostEmissionCalculation(self)

    def __str__(self) -> str:
        first_scenario = next(iter(self.solution_loader.scenarios.values()))
        return f"Results of '{first_scenario.analysis.dataset}'"

    def get_df(
        self,
        component_name: str,
        scenario_name: Optional[str] = None,
        data_type: Literal["dataframe", "units"] = "dataframe",
        index: Optional[
            Union[NestedTuple, NestedDict, list[str], str, float, int]
        ] = None,
    ) -> Optional[Union[dict[str, "pd.DataFrame | pd.Series[Any]"], pd.Series]]:
        """Returns the raw results without any further processing.

        Transforms a parameter or variable dataframe string into
        an actual pandas dataframe.

        Args:
            component_name (string): The string to decode
            scenario_name: Which scenario to take. If none is specified, all are
                returned.
            data_type: The type of data to extract. Either 'dataframe' or 'units'
            index: slicing index of the resulting dataframe

        Returns:
            DataFrame: The corresponding dataframe

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_df('<component_name>') # dataframe of "<component_name>"
            >>> r.get_df('<component_name>', '<scenario_name>') # dataframe of
                "<component_name>" in "<scenario_name>"
            >>> r.get_df('<component_name>', index={'<index_name>': '<index_value>'})
                # dataframe of "<component_name>" for a specific index value to slice the
                dataframe

        """
        scenario_names = (
            list(self.solution_loader.scenarios.keys())
            if scenario_name is None
            else [scenario_name]
        )

        if len(scenario_names) == 1:
            scenario_name = scenario_names[0]
            scenario = self.solution_loader.scenarios[scenario_name]
            if component_name not in scenario.components:
                logger.warning(
                    f"Component {component_name} not found. If you expected "
                    "this component to be present, the solution is probably "
                    "empty and therefore skipped."
                )
                return pd.Series()
            component = scenario.get_component(component_name)
            if data_type == "units" and not component.has_units:
                return None
            idx = reformat_slicing_index(index, component)
            ans = self.solution_loader.get_component_data(
                scenario, component, data_type=data_type, index=idx
            )
        else:
            ans = {}
            for scenario_name in scenario_names:
                scenario = self.solution_loader.scenarios[scenario_name]
                if component_name not in scenario.components:
                    continue
                component = scenario.get_component(component_name)
                if data_type == "units" and not component.has_units:
                    return None
                idx = reformat_slicing_index(index, component)
                ans[scenario_name] = self.solution_loader.get_component_data(
                    scenario, component, data_type=data_type, index=idx
                )
            if len(ans) == 0:
                logger.warning(
                    f"Component {component_name} not found. If you expected "
                    "this component to be present, the solution is probably "
                    "empty and therefore skipped."
                )
                return {}
        return ans

    def _get_full_ts_per_scenario(
        self,
        scenario: Scenario,
        component: Component,
        year: Optional[int] = None,
        discount_to_first_step: bool = True,
        keep_raw: bool = False,
        index: tuple[str, ...] | None = None,
    ) -> "pd.DataFrame":
        """Calculates the full timeseries per scenario.

        Args:
            scenario: The scenario for with the component should be extracted
                (only if needed)
            component: Component for the Series
            discount_to_first_step: apply annuity to first year of interval or
                entire interval
            year: year of which full time series is selected
            keep_raw: Keep the raw values of the rolling horizon optimization
            index: slicing index of the resulting dataframe

        Returns:
            Full timeseries
        """
        assert component.timestep_type is not None, "Component has no timestep type."

        if index is None:
            index = tuple()

        sequence_timesteps = self.solution_loader.get_sequence_time_steps(
            scenario, component.timestep_type
        )
        if year is None:
            years = [i for i in range(0, scenario.system.optimized_years)]
        else:
            year = scenario.convert_year2ts(year)
            years = [year]

        # slice index with time steps of year
        select_year_time_steps = False
        if (
            component.timestep_type is TimestepType.operational
            or component.timestep_type is TimestepType.storage
        ):
            if not any(str(component.timestep_type.value) in i for i in index):
                time_steps = self.solution_loader.get_timesteps_of_years(
                    scenario, component.timestep_type, tuple(years)
                ).values
                index = index + (
                    f"{component.timestep_type.value} in "
                    f"[{', '.join(time_steps.astype(str))}]",
                )
                select_year_time_steps = True
        series = self.solution_loader.get_component_data(
            scenario, component, keep_raw=keep_raw, index=index
        )
        if isinstance(series.index, pd.MultiIndex):
            series = series.unstack(component.timestep_name)

        if component.timestep_type is TimestepType.yearly:
            if component.component_type is ComponentType.dual:
                annuity = self._get_annuity(scenario, discount_to_first_step)
                ans = series / annuity
            else:
                ans = series

            try:
                ans = ans[years]
            except KeyError:
                pass
            ans = scenario.convert_ts2year(ans)
            return ans

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
                time_steps_year = self.solution_loader.get_timesteps_of_years(
                    scenario, component.timestep_type, (year_temp,)
                )
                series[time_steps_year] = series[time_steps_year] / annuity[year_temp]
        try:
            if component.timestep_type is TimestepType.operational:
                if select_year_time_steps:
                    sequence_timesteps = sequence_timesteps[
                        sequence_timesteps.isin(time_steps)
                    ]
                output_df = series[sequence_timesteps]
            elif component.timestep_type is TimestepType.storage:
                # for storage components, the last timestep is the final state,
                # linear interpolation is used
                last_occurrences = sequence_timesteps.drop_duplicates(keep="last")
                first_occurrences = sequence_timesteps.drop_duplicates(keep="first")
                last_occurrences = pd.Series(
                    last_occurrences.index, index=last_occurrences.values
                )
                first_occurrences = pd.Series(
                    first_occurrences.index, index=first_occurrences.values
                )
                last_occurrences = last_occurrences[
                    last_occurrences.index.intersection(series.columns)
                ]
                output_df = series[last_occurrences.index].rename(
                    last_occurrences, axis=1
                )
                output_df = output_df.apply(
                    lambda row: np.interp(
                        sequence_timesteps.index,
                        row.index,
                        row.values,
                        left=np.nan,
                        right=np.nan,
                    ),
                    axis=1,
                    result_type="expand",
                )
                # fill missing ts with nan
                time_steps_start_end = (
                    self.solution_loader.get_time_steps_storage_level_startend_year(
                        scenario
                    )
                )
                time_steps_start_end = {
                    k: v
                    for k, v in time_steps_start_end.items()
                    if k in first_occurrences and v in last_occurrences
                }
                for tstart, tend in time_steps_start_end.items():
                    tstart_reconstructed = first_occurrences[tstart]
                    _output_df_recon = output_df.iloc[0][tstart_reconstructed:]
                    first_valid_timestep = _output_df_recon.index[
                        np.isnan(_output_df_recon).argmin()
                    ]
                    df_temp = pd.DataFrame(
                        index=series.index,
                        columns=range(
                            tstart_reconstructed - 1, first_valid_timestep + 1
                        ),
                        dtype=float,
                    )
                    df_temp.loc[:, tstart_reconstructed - 1] = series.loc[:, tend]
                    df_temp.loc[:, first_valid_timestep] = series.loc[
                        :, sequence_timesteps[first_valid_timestep]
                    ]
                    df_temp = df_temp.interpolate(method="linear", axis=1)
                    output_df.loc[
                        :, first_occurrences[tstart] : last_occurrences[tstart]
                    ] = df_temp.loc[:, tstart_reconstructed:first_valid_timestep]
                if select_year_time_steps:
                    sequence_timesteps = sequence_timesteps[
                        sequence_timesteps.isin(time_steps)
                    ]
                output_df = output_df[sequence_timesteps.index]
            else:
                raise ValueError(
                    f"Invalid timestep type {component.timestep_type} for "
                    "component {component}"
                )
        except KeyError:
            output_df = series

        output_df = output_df.T.reset_index(drop=True).T

        return output_df

    def get_full_ts(
        self,
        component_name: str,
        scenario_name: Optional[str] = None,
        discount_to_first_step: bool = True,
        year: Optional[int] = None,
        keep_raw: bool = False,
        index: Optional[
            Union[NestedTuple, NestedDict, list[str], str, float, int]
        ] = None,
    ) -> "pd.DataFrame | pd.Series[Any]":
        """Calculates the full timeseries.

        Args:
            component_name: Name of the component
            scenario_name: The scenario for with the component should be
                extracted (only if needed)
            discount_to_first_step: apply annuity to first year of interval or
                entire interval
            year: year of which full time series is selected
            keep_raw: Keep the raw values of the rolling horizon optimization
            index: slicing index of the resulting dataframe

        Returns:
           Full timeseries
        """
        if scenario_name is None:
            scenario_names = list(self.solution_loader.scenarios)
        else:
            scenario_names = [scenario_name]

        scenarios_dict: dict[str, "pd.DataFrame | pd.Series[Any]"] = {}

        for scenario_name in scenario_names:
            scenario = self.solution_loader.scenarios[scenario_name]
            if component_name not in scenario.components:
                continue
            component = scenario.get_component(component_name)
            idx = reformat_slicing_index(index, component)
            scenarios_dict[scenario_name] = self._get_full_ts_per_scenario(
                scenario,
                component,
                discount_to_first_step=discount_to_first_step,
                year=year,
                keep_raw=keep_raw,
                index=idx,
            )
        if len(scenarios_dict) == 0:
            logger.warning(
                f"Component {component_name} not found. If you expected "
                "this component to be present, the solution is probably empty "
                "and therefore skipped."
            )
            return pd.Series()

        return self._concat_scenarios_dict(scenarios_dict, scenario_names)

    def _get_total_per_scenario(
        self,
        scenario: Scenario,
        component: Component,
        year: Optional[int] = None,
        keep_raw: bool = False,
        index: tuple[str, ...] | None = None,
    ) -> "pd.DataFrame | pd.Series[Any]":
        """Calculates the total values of a component for a specific scenario.

        :param scenario: Scenario
        :param component: Component
        :param year: Filter the results by a given year
        :param keep_raw: Keep the raw values of the rolling horizon optimization
        :param index: slicing index of the resulting dataframe
        :return: Total values of the component
        """
        if index is None:
            index = tuple()
        series = self.solution_loader.get_component_data(
            scenario, component, keep_raw, index=index
        )

        if year is None:
            years = [i for i in range(0, scenario.system.optimized_years)]
        else:
            year = scenario.convert_year2ts(year)
            years = [year]

        if component.timestep_type is None or type(series.index) is not pd.MultiIndex:
            if component.timestep_type is TimestepType.yearly:
                series = scenario.convert_ts2year(series)
            return series

        if component.timestep_type is TimestepType.yearly:
            ans = series.unstack(component.timestep_name)
            ans = ans[years]
            ans = scenario.convert_ts2year(ans)
            return ans

        timestep_duration = self.solution_loader.get_timestep_duration(
            scenario, component
        )

        unstacked_series = series.unstack(component.timestep_name)
        total_value = unstacked_series.multiply(timestep_duration, axis=1)

        ans = pd.DataFrame(index=unstacked_series.index)

        for y in years:
            timesteps = self.solution_loader.get_timesteps(scenario, component, int(y))
            try:
                ans.insert(
                    len(ans.columns),
                    y,
                    total_value[timesteps].sum(axis=1, skipna=False),
                )
            except KeyError:
                timestep_list = [i for i in timesteps if i in total_value]
                ans.insert(
                    len(ans.columns),
                    year,
                    total_value[timestep_list].sum(axis=1, skipna=False),
                )

        if "mf" in ans.index.names:
            ans = ans.reorder_levels(
                [i for i in ans.index.names if i != "mf"] + ["mf"]
            ).sort_index(axis=0)
        ans = scenario.convert_ts2year(ans)
        return ans

    def get_total(
        self,
        component_name: str,
        year: Optional[int] = None,
        scenario_name: Optional[str] = None,
        keep_raw: bool = False,
        index: Optional[
            Union[NestedTuple, NestedDict, list[str], str, float, int]
        ] = None,
    ) -> "pd.DataFrame | pd.Series[Any]":
        """Calculates the total values of a component for a all scenarios.

        Args:
            component_name: Name of the component. Should not be used for dual
                variables!
            year: Filter the results by a given year
            scenario_name: Filter the results by a given scenario
            keep_raw: Keep the raw values of the rolling horizon optimization
            index: slicing index of the resulting dataframe

        Returns:
            DataFrame: Total values of the component

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_total('<component_name>') # total values of "<component_name>"
            >>> r.get_total('<component_name>', '<scenario_name>') # total values of
                "<component_name>" in "<scenario_name>"
            >>> r.get_total('<component_name>', <year>) # total values of
                "<component_name>" for a specific year
            >>> r.get_total('<component_name>', index={'<index_name>': '<index_value>'})
                # total values of "<component_name>" for a specific index value to slice
                the dataframe
            >>> r.get_total('<component_name>', keep_raw=True) # total values for the 
                following years in one rolling horizon step are kept, instead of only 
                the first year of the rolling horizon step
        """
        # Throw error if used for a dual variable
        if component_name in self.get_component_names("dual"):
            raise ValueError(
                "This method does not support the extraction of "
                "dual variables. Please use the methods "
                "`get_dual()` or `get_full_ts()` instead."
            )

        if scenario_name is None:
            scenario_names = list(self.solution_loader.scenarios)
        else:
            scenario_names = [scenario_name]

        scenarios_dict: dict[str, "pd.DataFrame | pd.Series[Any]"] = {}

        for scenario_name in scenario_names:
            scenario = self.solution_loader.scenarios[scenario_name]
            if component_name not in scenario.components:
                continue
            component = scenario.get_component(component_name)
            idx = reformat_slicing_index(index, component)
            current_total = self._get_total_per_scenario(
                scenario, component, year, keep_raw, index=idx
            )

            if type(current_total) is pd.Series:
                current_total = current_total.rename(component_name)

            scenarios_dict[scenario_name] = current_total

        if len(scenarios_dict) == 0:
            logger.warning(
                f"Component {component_name} not found. If you expected this "
                "component to be present, the solution is probably empty and "
                "therefore skipped."
            )
            return pd.Series()

        return self._concat_scenarios_dict(scenarios_dict, scenario_names)

    def _concat_scenarios_dict(
        self,
        scenarios_dict: dict[str, "pd.DataFrame | pd.Series[Any]"],
        scenario_names: list[str],
    ) -> pd.DataFrame:
        """Concatenates a dict of the form str: Data to one dataframe.

        Args:
            scenarios_dict: Dict containing the scenario names as key and the
                values as values.

        Returns:
            Concatenated dataframe
        """
        if len(scenario_names) == 1:
            ans = scenarios_dict[scenario_names[0]]
            return ans
        scenario_names = list(scenarios_dict.keys())
        if isinstance(scenarios_dict[scenario_names[0]], pd.Series):
            total_value = pd.concat(
                scenarios_dict, keys=scenarios_dict.keys(), axis=1
            ).T
        else:
            try:
                # type: ignore # noqa
                total_value = pd.concat(scenarios_dict, keys=scenarios_dict.keys())
            except Exception:
                total_value = pd.concat(
                    scenarios_dict, keys=scenarios_dict.keys(), axis=1
                ).T
        return total_value

    def _get_annuity(
        self, scenario: Scenario, discount_to_first_step: bool = True
    ) -> pd.Series:
        """Discounts the duals.

        Args:
            discount_to_first_step: apply annuity to first year of interval or
                entire interval
            scenario: scenario name whose results are assessed

        Returns:
            annuity of the duals
        """
        system = scenario.system
        discount_rate_component = scenario.get_component("discount_rate")
        # calculate annuity
        discount_rate = self.solution_loader.get_component_data(
            scenario, discount_rate_component
        ).squeeze()

        years = list(range(0, system.optimized_years))
        optimized_years = self.solution_loader.get_optimized_years(scenario)
        annuity = pd.Series(index=years, dtype=float)
        for year in years:
            # closest year in optimized years that is smaller than year
            start_year = [y for y in optimized_years if y <= year][-1]
            interval_between_years = system.interval_between_years
            if year == years[-1]:
                interval_between_years_this_year = 1
            else:
                interval_between_years_this_year = system.interval_between_years
            if discount_to_first_step:
                annuity[year] = interval_between_years_this_year * (
                    (1 / (1 + discount_rate))
                    ** (interval_between_years * (year - start_year))
                )
            else:
                annuity[year] = sum(
                    (
                        (1 / (1 + discount_rate))
                        ** (
                            interval_between_years * (year - start_year)
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
        component_name: str,
        scenario_name: Optional[str] = None,
        year: Optional[int] = None,
        index: Optional[
            Union[NestedTuple, NestedDict, list[str], str, float, int]
        ] = None,
        discount_to_first_step: bool = True,
        keep_raw: bool = False,
    ) -> Optional["pd.DataFrame | pd.Series[Any]"]:
        """Extracts the dual variables of a component.

        Args:
            component_name: Name of dual
            scenario_name: Scenario Name
            year: Year
            index: slicing index of the resulting dataframe
            discount_to_first_step: apply annuity to first year of interval or
                entire interval
            keep_raw: Keep the raw values of the rolling horizon optimization

        Returns:
            DataFrame: Duals of the component

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_dual('<component_name>') # duals of "<component_name>"
            >>> r.get_dual('<component_name>', '<scenario_name>') # duals of 
                "<component_name>" in "<scenario_name>"
            >>> r.get_dual('<component_name>', <year>) # duals of
                "<component_name>" for a specific year
            >>> r.get_dual('<component_name>', index={'<index_name>': '<index_value>'}) 
                # duals of "<component_name>" for a specific index value to slice the
                dataframe
            >>> r.get_dual('<component_name>', discount_to_first_step=False) # duals of
                "<component_name>" without discounting to the first step
            >>> r.get_dual('<component_name>', keep_raw=True) # duals for the following
                years in one rolling horizon step are kept, instead of only the first 
                year of the rolling horizon step
        """
        if not self.get_solver(scenario_name=scenario_name).save_duals:
            logger.warning("Duals are not calculated. Skip.")
            return None

        duals = self.get_full_ts(
            component_name=component_name,
            scenario_name=scenario_name,
            year=year,
            discount_to_first_step=discount_to_first_step,
            keep_raw=keep_raw,
            index=index,
        )
        return duals

    def get_unit(
        self,
        component_name: str,
        scenario_name: Optional[str] = None,
        index: Optional[
            Union[NestedTuple, NestedDict, list[str], str, float, int]
        ] = None,
        droplevel: bool = True,
        convert_to_yearly_unit: bool = False,
    ) -> None | Series | str:
        """Extracts the unit of a given Component. If no scenario is given, a
        random one is taken.

        Args:
            component_name: Name of the component
            scenario_name: Name of the scenario
            index: slicing index of the resulting dataframe
            droplevel: Drop the location and time levels of the multiindex
            convert_to_yearly_unit: If True, the unit is converted to a
                yearly unit, i.e., for components with an operational time step
                type, the unit is multiplied by hours.

        Returns:
            DataFrame: The corresponding unit

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_unit('<component_name>') # unit of "<component_name>"
            >>> r.get_unit('<component_name>', '<scenario_name>') # unit of 
                "<component_name>" in "<scenario_name>"
            >>> r.get_unit('<component_name>', index={'<index_name>': '<index_value>'}) 
                # unit of "<component_name>" for a specific index value to slice the 
                dataframe
            >>> r.get_unit('<component_name>', droplevel=False) # unit of
                "<component_name>" without dropping the location and 
                time levels of the multiindex
            >>> r.get_unit('<component_name>', convert_to_yearly_unit=True) # unit of
                "<component_name>" converted to a yearly unit, i.e., for components with 
                an operational time step type, the unit is multiplied by hours.
             
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        if component_name == "objective":
            if self.get_analysis(scenario_name=scenario_name).objective == "total_cost":
                component_name = "net_present_cost"
            elif (
                self.get_analysis(scenario_name=scenario_name).objective
                == "total_carbon_emissions"
            ):
                component_name = "carbon_emissions_annual"
            else:
                raise ValueError(
                    f"Invalid objective function "
                    f"{self.get_analysis(scenario_name=scenario_name).objective}"
                )
            if component_name not in self.get_component_names("variable"):
                logger.warning(
                    f"Component {component_name} not found in "
                    f"{self.get_analysis(scenario_name=scenario_name)}"
                )
        units = self.get_df(
            component_name,
            scenario_name=scenario_name,
            data_type="units",
            index=index,
        )
        if units is None:
            return None
        if not isinstance(units, pd.Series):
            raise TypeError(f"Invalid units type: {type(units)}")
        if droplevel:
            # TODO make more flexible
            loc_idx = ["node", "location", "edge", "set_location", "set_nodes"]
            time_idx = [
                "year",
                "time_operation",
                "time_storage_level",
                "set_time_steps_operation",
            ]
            drop_idx = pd.Index(loc_idx + time_idx).intersection(units.index.names)
            if len(units.index.names.difference(drop_idx)) == 0:
                units = units.iloc[0]
            else:
                units.index = units.index.droplevel(drop_idx.to_list())
                units = units[~units.index.duplicated()]
        # convert to pint units
        if isinstance(units, pd.Series):
            for i in units.index:
                units[i] = self._convert_to_pint_units(
                    units[i], convert_to_yearly_unit, component_name
                )
        elif isinstance(units, str):
            units = self._convert_to_pint_units(
                units, convert_to_yearly_unit, component_name
            )
        else:
            raise TypeError(f"Invalid units type: {type(units)}")

        return units

    def _convert_to_pint_units(
        self, u: str, convert_to_yearly_unit: bool, component_name: str
    ) -> str:
        """Converts a string to a pint unit."""
        component = None
        for s in self.solution_loader.scenarios:
            if component_name in self.solution_loader.scenarios[s].components:
                component = self.solution_loader.scenarios[s].get_component(
                    component_name
                )
                break
        if component is None:
            return u
        timestep_type = component.timestep_type

        try:
            unit_expression = self.ureg.parse_expression(u)
            if convert_to_yearly_unit and timestep_type is TimestepType.operational:
                unit_expression = unit_expression * self.ureg.h
            u_return = f"{unit_expression.u:~D}"
        # if the unit is not in the pint registry, change the string manually
        # (normally when the unit_definition.txt is not saved)
        except Exception:
            if convert_to_yearly_unit and timestep_type is TimestepType.operational:
                if u.endswith(" / hour"):
                    u_return = u.replace(" / hour", "")
                else:
                    u_return = f"{u} * hour"
            else:
                u_return = u
        return u_return

    def get_system(self, scenario_name: Optional[str] = None) -> System:
        """Extract system configurations from a scenario.

        Extracts system configurations from the results of a scenario. This
        ensures the tractability of model configurations. System configurations
        are those specified in the ``system.json`` file of a given model.

        Args:
            scenario_name (str, optional): The name of the scenario for which
                to extract the system configuration. If no value is given, then
                the first scenario is used. Default value: ``None``.

        Returns:
            System: System configuration.

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_system() # system configurations of first scenario
            >>> r.get_system('<scenario_name>') # system configuration of "scenario_name"

        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        return self.solution_loader.scenarios[scenario_name].system

    def get_analysis(self, scenario_name: Optional[str] = None) -> Analysis:
        """Extract analysis configurations from a scenario.

        Extracts analysis configurations from the results of a scenario. This
        ensures the tractability of model configurations. Analysis
        configurations are those specified under the ``analysis`` object in
        the ``config.json`` file.

        Args:
            scenario_name (str, optional): The name of the scenario for which
                to extract the system configuration. If no value is given, then
                the first scenario is used. Default value: ``None``.

        Returns:
            Analysis: Analysis configuration.

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_analysis() # analysis config of first scenario
            >>> r.get_analysis('<scenario_name>') # analysis config of "scenario_name"

        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        return self.solution_loader.scenarios[scenario_name].analysis

    def get_solver(self, scenario_name: Optional[str] = None) -> Solver:
        """Extract solver configurations from a scenario.

        Extracts solver configurations from the results of a scenario. This
        ensures the tractability of model configurations. Solver configurations
        are those specified under the ``solver`` object in the ``config.json``
        file.

        Args:
            scenario_name (str, optional): The name of the scenario for which
                to extract the system configuration. If no value is given, then
                the first scenario is used. Default value: ``None``.

        Returns:
            Solver: Solver configuration.

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_solver() # solver configurations of first scenario
            >>> r.get_solver('<scenario_name>') # solver configuration of "scenario_name"

        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        return self.solution_loader.scenarios[scenario_name].solver

    def get_doc(self, component_name: str) -> str:
        """Extracts the documentation of a given Component.

        Args:
            component_name (str): Name of the component

        Returns:
            str: The corresponding documentation of the component.

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_doc('<component_name>') # documentation of "<component_name>"
        """
        component = None
        for scenario in self.solution_loader.scenarios.values():
            if component_name in scenario.components:
                component = scenario.get_component(component_name)
                break
        if component is None:
            logger.warning(
                f"Component {component_name} not found and the documentation "
                "cannot be returned."
            )
            return ""
        return component.doc

    def get_index_names(
        self, component_name: str, scenario_name: Optional[str] = None
    ) -> list[str]:
        """Docstring for get_index_names.

        Args:
            component_name (str): The name of the component for which to 
                extract the index names.
            scenario_name (Optional[str]): The name of the scenario for which
                to extract the index names. If no value is given, then the first
                scenario is used. Default value: ``None``.
        Returns:
            list[str]: A list of index names for the specified component.

            
        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_index_names('<component_name>') # index names of "<component_name>"
            >>> r.get_index_names('<component_name>', '<scenario_name>') # index names of "<component_name>" in "<scenario_name>"
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        scenario = self.solution_loader.scenarios[scenario_name]
        if component_name not in scenario.components:
            logger.warning(
                f"Component {component_name} not found and the index names "
                "cannot be returned."
            )
            return []
        component = scenario.get_component(component_name)
        return component.index_names

    def get_years(self, scenario_name: Optional[str] = None) -> list[int]:
        """Extracts the years of a given Scenario. If no scenario is given, the first
        scenario is taken.

        Args:
            scenario_name (str, optional): The name of the scenario for which
                to extract the years. If no value is given, then the first
                scenario is used. Default value: ``None``.

        Returns:
            list[int]: A list of years for the specified scenario.
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        system = self.get_system(scenario_name)
        reference_year = system.reference_year
        interval_between_years = system.interval_between_years
        optimized_years = system.optimized_years
        years = [reference_year + i * interval_between_years 
                 for i in range(optimized_years)]
        return years

    def has_MF(self, scenario_name: Optional[str] = None) -> bool:
        """Extracts the System config of a given Scenario. If no scenario is given,
        a random one is taken.

        Args:
            scenario_name (str, optional): The name of the scenario for which
                to extract the System config. If no value is given, then the first
                scenario is used. Default value: ``None``.

        Returns:
            bool: A boolean indicating whether the scenario uses a rolling horizon.
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        scenario = self.solution_loader.scenarios[scenario_name]
        return scenario.system.use_rolling_horizon

    def get_coords(self, scenario_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Extracts the coordinates of the nodes of a given Scenario. If no
        scenario is given, a random one is taken.

        Args:
            scenario_name (str, optional): The name of the scenario for which
                to extract the coordinates. If no value is given, then the first
                scenario is used. Default value: ``None``.

        Returns:
            pd.DataFrame: The corresponding coordinates.
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        system = self.get_system(scenario_name)
        if hasattr(system, "coords"):
            coords = pd.DataFrame(system.coords).T
            if coords.empty:
                logger.warning(
                    f"Coordinates of nodes are not saved for version "
                    f"{self.get_analysis().zen_garden_version}."
                )
                return None
            return pd.DataFrame(system.coords).T
        else:
            logger.warning(
                f"Coordinates of nodes are not saved for version "
                f"{self.get_analysis().zen_garden_version}."
            )
            return None

    def get_component_names(self, component_type: str) -> list[str]:
        """Returns the names of all components of a given type.
        
        Args:
            component_type (str): Type of the component. Must be one of the
                valid component types defined in ComponentType:
                ["variable", "dual", "parameter", "set", "constraint"].
                Duals are only available if the solver has been configured to save them.

        Returns:
            list[str]: List of component names of the specified type.

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_component_names('variable') # list of variable component names
            >>> r.get_component_names('dual') # list of dual component names
        """
        assert component_type in ComponentType.get_component_type_names(), (
            f"Invalid component type: {component_type}. Valid types are: "
            f"{ComponentType.get_component_type_names()}"
        )
        list_names = []
        for scenario in self.solution_loader.scenarios:
            for cn in self.solution_loader.scenarios[scenario].component_types[
                component_type
            ]:
                if cn not in list_names:
                    list_names.append(cn)
        return list_names

    def get_sectoral_costs(
        self,
        scenario_name: Optional[str] = None,
        carrier: Optional[str] = None,
        spatially_resolved: bool = False,
        mode: CostEmissionMode = "final_demand",
        overwrite: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculates the sectoral costs of a scenario through 
        Leontief Input-Output tables. The capital and operational expenditures of each 
        technology and the fuel cost of each carrier are allocated to the sectors 
        that use them. 

        When specifying a carrier, only the cost of producing that carrier is returned.
        Note that the tables are formulated for all sectors, so returning the costs for
        all sectors does not add any overhead.
        The sectoral costs are either returned aggregated over all locations or
        spatially resolved for each location (`spatially_resolved = True`). 
        The cost of transport technologies are 50/50 allocated to the connecting nodes.
        By default, the costs to produce the final demand of each sector are returned
        (`mode = "final_demand"`), but the costs of the total production of each carrier
        can also be returned (`mode = "total_production"`). Finally, the relative 
        production costs of each sector can be returned (`mode = "relative"`).

        Args:
            scenario_name: The scenario for which the sectoral costs should be
                calculated; if None, the costs of the first scenario are returned.
            carrier: The carrier for which the sectoral costs should be calculated. If
                None, the costs of all carriers are returned.
            spatially_resolved: Whether the sectoral costs should be returned
                spatially resolved for each node or aggregated over all nodes.
            mode: The mode of calculation for the sectoral costs 
                ("final_demand", "total_production", or "relative").
            overwrite: Whether to rebuild the leontief input-output tables even if 
                they have already been built and saved.

        Returns:
            Tuple of two DataFrames:
                - The first DataFrame contains the total upstream/downstream costs 
                  of each sector.
                - The second DataFrame contains the direct costs of each sector. 
                  These are the costs that are directly associated with 
                  the sector itself, without considering the upstream/downstream effects.

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_sectoral_costs('<scenario_name>') 
                # sectoral costs of "<scenario_name>"
            >>> r.get_sectoral_costs('<scenario_name>', carrier='<carrier_name>') 
                # sectoral costs of "<carrier_name>" in "<scenario_name>"
            >>> r.get_sectoral_costs('<scenario_name>', spatially_resolved=True)
                # spatially resolved sectoral costs of "<scenario_name>"
            >>> r.get_sectoral_costs('<scenario_name>', mode='total_production')
                # sectoral costs of the total production of each carrier in 
                "<scenario_name>"
            >>> r.get_sectoral_costs('<scenario_name>', mode='relative')
                # relative production costs of each sector in "<scenario_name>"
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        sectoral_costs, direct_costs = (
            self.cost_emission_calculation.calculate_leontief_data(
                scenario_name=scenario_name,
                carrier=carrier,
                spatially_resolved=spatially_resolved,
                mode=mode,
                overwrite=overwrite,
                is_cost=True
            )
        )
        return sectoral_costs, direct_costs
    
    def get_sectoral_emissions(
        self,
        scenario_name: Optional[str] = None,
        carrier: Optional[str] = None,
        spatially_resolved: bool = False,
        mode: CostEmissionMode = "final_demand",
        overwrite: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Calculates the sectoral emissions of a scenario through 
        Leontief Input-Output tables. The capital and operational expenditures of each 
        technology and the fuel cost of each carrier are allocated to the sectors 
        that use them. 

        When specifying a carrier, only the emissions of producing that carrier is returned.
        Note that the tables are formulated for all sectors, so returning the emissions for
        all sectors does not add any overhead.
        The sectoral emissions are either returned aggregated over all locations or
        spatially resolved for each location (`spatially_resolved = True`). 
        The emissions of transport technologies are 50/50 allocated to the connecting nodes.
        By default, the emissions to produce the final demand of each sector are returned
        (`mode = "final_demand"`), but the emissions of the total production of each carrier
        can also be returned (`mode = "total_production"`). Finally, the relative 
        production emissions of each sector can be returned (`mode = "relative"`).

        Args:
            scenario_name: The scenario for which the sectoral emissions should be
                calculated; if None, the emissions of the first scenario are returned.
            carrier: The carrier for which the sectoral emissions should be calculated. If
                None, the emissions of all carriers are returned.
            spatially_resolved: Whether the sectoral emissions should be returned
                spatially resolved for each node or aggregated over all nodes.
            mode: The mode of calculation for the sectoral emissions
                ("final_demand", "total_production", or "relative").
            overwrite: Whether to rebuild the leontief input-output tables even if 
                they have already been built and saved.

        
        Returns:
            Tuple of two DataFrames:
                - The first DataFrame contains the total upstream/downstream emissions 
                  of each sector.
                - The second DataFrame contains the direct emissions of each sector. 
                  These are the emissions that are directly associated with 
                  the sector itself, without considering the upstream/downstream effects.

        Examples:
            Basic usage example:

            >>> from zen_garden.postprocess.results.results import Results
            >>> r = Results(path='<result_folder>')
            >>> r.get_sectoral_emissions('<scenario_name>') 
                # sectoral emissions of "<scenario_name>"
            >>> r.get_sectoral_emissions('<scenario_name>', carrier='<carrier_name>') 
                # sectoral emissions of "<carrier_name>" in "<scenario_name>"
            >>> r.get_sectoral_emissions('<scenario_name>', spatially_resolved=True)
                # spatially resolved sectoral emissions of "<scenario_name>"
            >>> r.get_sectoral_emissions('<scenario_name>', mode='total_production')
                # sectoral emissions of the total production of each carrier in 
                "<scenario_name>"
            >>> r.get_sectoral_emissions('<scenario_name>', mode='relative')
                # relative production emissions of each sector in "<scenario_name>"
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        sectoral_emissions, direct_emissions = (
            self.cost_emission_calculation.calculate_leontief_data(
                scenario_name=scenario_name,
                carrier=carrier,
                spatially_resolved=spatially_resolved,
                mode=mode,
                overwrite=overwrite,
                is_cost=False
            )
        )
        return sectoral_emissions, direct_emissions
