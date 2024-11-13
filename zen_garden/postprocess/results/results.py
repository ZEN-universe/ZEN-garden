"""
This module contains the Results class, which is used to extract and process the results of a model run.
"""
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
from zen_garden.model.default_config import Config, Analysis, Solver, System
import importlib
import os
import logging
import json
from pathlib import Path


class Results:
    """
    The Results class is used to extract and process the results of a model run.
    """
    def __init__(self, path: str):
        """
        Initializes the Results class.

        :param path: Path to the results folder
        """
        self.solution_loader: SolutionLoader = MultiHdfLoader(path)
        self.has_scenarios = len(self.solution_loader.scenarios) > 1
        self.has_rh = self.solution_loader.has_rh
        first_scenario = next(iter(self.solution_loader.scenarios.values()))
        self.name = Path(first_scenario.analysis.dataset).name
        self.ureg = first_scenario.ureg

    def __str__(self) -> str:
        first_scenario = next(iter(self.solution_loader.scenarios.values()))
        return f"Results of '{first_scenario.analysis.dataset}'"

    def get_df(
        self,
        component_name: str,
        scenario_name: Optional[str] = None,
        data_type: Literal["dataframe", "units"] = "dataframe",
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
        :return: Full timeseries of the element
        """
        assert component.timestep_type is not None, "Component has no timestep type."

        series = self.solution_loader.get_component_data(
            scenario, component, keep_raw=keep_raw
        )
        if element_name is not None and element_name in series.index.get_level_values(
            0
        ):
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
                time_steps_year = self.solution_loader.get_timesteps_of_years(
                    scenario, component.timestep_type, (year_temp,)
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
            if output_df.empty:
                return pd.DataFrame(index=[],columns=hours_of_year)
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
        :return: Full timeseries of the element
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
        :return: Total values of the component
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
                ans.insert(len(ans.columns), y, total_value[timesteps].sum(axis=1, skipna=False))  # type: ignore
            except KeyError:
                timestep_list = [i for i in timesteps if i in total_value]
                ans.insert(len(ans.columns), year, total_value[timestep_list].sum(axis=1, skipna=False))  # type: ignore # noqa

        if "mf" in ans.index.names:
            ans = ans.reorder_levels(
                [i for i in ans.index.names if i != "mf"] + ["mf"]
            ).sort_index(axis=0)

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
        :return: Total values of the component
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
        :return: Concatenated dataframe
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
        return total_value

    def _get_annuity(
        self, scenario: Scenario, discount_to_first_step: bool = True
    ) -> pd.Series:
        """discounts the duals

        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :param scenario: scenario name whose results are assessed
        :return: annuity of the duals
        """
        system = scenario.system
        discount_rate_component = self.solution_loader.components["discount_rate"]
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
            # if self.solution_loader.has_rh:
            #     if discount_to_first_step:
            #         annuity[year] = interval_between_years_this_year * (
            #             1 / (1 + discount_rate)
            #         )
            #     else:
            #         annuity[year] = sum(
            #             ((1 / (1 + discount_rate)) ** (_intermediate_time_step))
            #             for _intermediate_time_step in range(
            #                 0, interval_between_years_this_year
            #             )
            #         )
            # else:
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
        element_name: Optional[str] = None,
        year: Optional[int] = None,
        discount_to_first_step: bool = True,
        keep_raw: Optional[bool] = False,
    ) -> Optional["pd.DataFrame | pd.Series[Any]"]:
        """extracts the dual variables of a component

        :param component: Name of dal
        :param scenario_name: Scenario Name
        :param element_name: Name of Element
        :param year: Year
        :param discount_to_first_step: apply annuity to first year of interval or entire interval
        :param keep_raw: Keep the raw values of the rolling horizon optimization
        :return: Duals of the component
        """
        if not self.get_solver(scenario_name=scenario_name).add_duals:
            logging.warning("Duals are not calculated. Skip.")
            return None

        component = self.solution_loader.components[component_name]
        assert (
            component.component_type is ComponentType.dual
        ), f"Given component {component} is not of type Dual."

        duals = self.get_full_ts(
            component_name=component_name,
            scenario_name=scenario_name,
            element_name=element_name,
            year=year,
            discount_to_first_step=discount_to_first_step,
            keep_raw=keep_raw,
        )
        return duals

    def get_unit(
        self,
        component_name: str,
        scenario_name: Optional[str] = None,
        droplevel: bool = True,
        is_total: bool = True,
    ) -> Optional[dict[str, "pd.DataFrame | pd.Series[Any]"]]:
        """
        Extracts the unit of a given Component. If no scenario is given, a random one is taken.

        :param component_name: Name of the component
        :param scenario_name: Name of the scenario
        :param droplevel: Drop the location and time levels of the multiindex
        :return: The corresponding unit
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        res = self.get_df(
            component_name, scenario_name=scenario_name, data_type="units"
        )
        if res is None:
            return None
        units = res[scenario_name]
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
        for i in units.index:
            try:
                u = units[i]
                u = self.ureg.parse_expression(u)
                if is_total and self.solution_loader.components[component_name].timestep_type is TimestepType.operational:
                    u = u * self.ureg.h
                units[i] = f"{u.u:~D}"
            # if the unit is not in the pint registry, change the string manually (normally, when the unit_definition.txt is not saved)
            except Exception:
                if is_total and self.solution_loader.components[component_name].timestep_type is TimestepType.operational:
                    if units[i].endswith(" / hour"):
                        units[i] = units[i].replace(" / hour", "")
                    else:
                        units[i] = f"{units[i]} * hour"

        return units

    def get_system(self, scenario_name: Optional[str] = None) -> System:
        """
        Extracts the System config of a given Scenario. If no scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        :return: The corresponding System config
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        return self.solution_loader.scenarios[scenario_name].system

    def get_analysis(self, scenario_name: Optional[str] = None) -> Analysis:
        """
        Extracts the Analysis config of a given Scenario. If no scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        :return: The corresponding Analysis config
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        return self.solution_loader.scenarios[scenario_name].analysis

    def get_solver(self, scenario_name: Optional[str] = None) -> Solver:
        """
        Extracts the Solver config of a given Scenario. If no scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        :return: The corresponding Solver config
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        return self.solution_loader.scenarios[scenario_name].solver

    def get_doc(self, component: str) -> str:
        """
        Extracts the documentation of a given Component.

        :param component: Name of the component
        :return: The corresponding documentation
        """
        return self.solution_loader.components[component].doc

    def get_years(self, scenario_name: Optional[str] = None) -> list[int]:
        """
        Extracts the years of a given Scenario. If no scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        :return: List of years
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        system = self.solution_loader.scenarios[scenario_name].system
        years = list(range(0, system.optimized_years))
        return years

    def has_MF(self, scenario_name: Optional[str] = None) -> bool:
        """
        Extracts the System config of a given Scenario. If no scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        :return: The corresponding System config
        """
        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))
        scenario = self.solution_loader.scenarios[scenario_name]
        return scenario.system.use_rolling_horizon

    def calculate_connected_edges(
        self, node: str, direction: str, set_nodes_on_edges: dict[str, str]
    ):
        """calculates connected edges going in (direction = 'in') or going out (direction = 'out')

        :param node: current node, connected by edges
        :param direction: direction of edges, either in or out. In: node = endnode, out: node = startnode
        :param set_nodes_on_edges: set of nodes on edges
        :return set_connected_edges: list of connected edges"""
        if direction == "in":
            # second entry is node into which the flow goes
            set_connected_edges = [
                edge
                for edge in set_nodes_on_edges
                if set_nodes_on_edges[edge][1] == node
            ]
        elif direction == "out":
            # first entry is node out of which the flow starts
            set_connected_edges = [
                edge
                for edge in set_nodes_on_edges
                if set_nodes_on_edges[edge][0] == node
            ]
        else:
            raise KeyError(f"invalid direction '{direction}'")
        return set_connected_edges

    def extract_carrier(
        self, dataframe: pd.DataFrame, carrier: str, scenario_name: str
    ) -> pd.DataFrame:
        """Returns a dataframe that only contains the desired carrier.
        If carrier is not contained in the dataframe, the technologies that have the provided reference carrier are returned.

        :param dataframe: pd.Dataframe containing the base data
        :param carrier: name of the carrier
        :param scenario_name: name of the scenario
        :return: filtered pd.Dataframe containing only the provided carrier
        """

        if "carrier" not in dataframe.index.names:
            reference_carriers = self.get_df(
                "set_reference_carriers", scenario_name=scenario_name
            )[scenario_name]
            data_extracted = pd.DataFrame()
            for tech in dataframe.index.get_level_values("technology"):
                if reference_carriers[tech] == carrier:
                    data_extracted = pd.concat(
                        [data_extracted, dataframe.query(f"technology == '{tech}'")],
                        axis=0,
                    )
            return data_extracted

        # check if desired carrier isn't contained in data (otherwise .loc raises an error)
        if carrier not in dataframe.index.get_level_values("carrier"):
            return None

        return dataframe.query(f"carrier == '{carrier}'")

    def edit_carrier_flows(
        self, data: pd.DataFrame, node: str, direction: str, scenario: str
    ) -> pd.DataFrame:
        """Extracts data of carrier_flow variable as needed for the plot_energy_balance function

        :param data: pd.DataFrame containing data to extract
        :param node: node of interest
        :param carrier: carrier of interest
        :param direction: flow direction with respect to node
        :param scenario: scenario of interest
        :return: pd.DataFrame containing carrier_flow data desired
        """
        set_nodes_on_edges = self.get_df("set_nodes_on_edges", scenario_name=scenario)[
            scenario
        ]
        set_nodes_on_edges = {
            edge: set_nodes_on_edges[edge].split(",")
            for edge in set_nodes_on_edges.index
        }

        data = data.loc[
            (
                slice(None),
                self.calculate_connected_edges(node, direction, set_nodes_on_edges),
            ),
            :,
        ]

        return data

    def get_energy_balance_dataframes(
        self, node: str, carrier: str, year: int, scenario_name: Optional[str] = None
    ) -> dict[str, "pd.Series[Any]"]:
        """Returns a dictionary with all dataframes that are relevant for the energy balance.
        The dataframes "flow_transport_in" and "flow_transport_out" contain the data of "flow_transport", filtered for in / out flow.

        :param node: Node of interest
        :param carrier: Carrier of interest
        :param year: Year of interest
        :param scenario_name: Scenario name of interest
        :return: Dictionary containing the relevant pd.Dataframes
        """
        components = {
            "flow_conversion_output": 1,
            "flow_conversion_input": -1,
            "flow_export": -1,
            "flow_import": 1,
            "flow_storage_charge": -1,
            "flow_storage_discharge": 1,
            "demand": 1,
            "flow_transport_in": 1,
            "flow_transport_out": -1,
            "shed_demand": 1,
        }

        if scenario_name is None:
            scenario_name = next(iter(self.solution_loader.scenarios.keys()))

        ans: dict[str, pd.DataFrame] = {}

        for component, factor in components.items():

            if component == "flow_transport_in":
                full_ts = self.get_full_ts(
                    "flow_transport", scenario_name=scenario_name, year=year
                )
                transport_loss = self.get_full_ts(
                    "flow_transport_loss", scenario_name=scenario_name, year=year
                )
                if full_ts.empty or transport_loss.empty:
                    continue
                full_ts = self.edit_carrier_flows(
                    full_ts - transport_loss, node, "in", scenario_name
                )
            elif component == "flow_transport_out":
                full_ts = self.get_full_ts(
                    "flow_transport", scenario_name=scenario_name, year=year
                )
                if full_ts.empty:
                    continue
                full_ts = self.edit_carrier_flows(full_ts, node, "out", scenario_name)
            else:
                try:
                    full_ts = self.get_full_ts(
                        component, scenario_name=scenario_name, year=year
                    )
                    if full_ts.empty:
                        continue
                except KeyError:
                    continue
            carrier_df = self.extract_carrier(full_ts, carrier, scenario_name)
            if carrier_df is not None:
                if "node" in carrier_df.index.names:
                    carrier_df = carrier_df.query(f"node == '{node}'")
                ans[component] = carrier_df.multiply(factor)

        return ans

    def get_component_names(self, component_type:str) -> list[str]:
        """ Returns the names of all components of a given type

        :param component_type: Type of the component
        :return: List of component names
        """
        assert component_type in ComponentType.get_component_type_names(), f"Invalid component type: {component_type}. Valid types are: {ComponentType.get_component_type_names()}"
        return [component for component in self.solution_loader.components if self.solution_loader.components[component].component_type.name == component_type]

if __name__ == "__main__":
    try:
        spec = importlib.util.spec_from_file_location("module", "config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.config
    except FileNotFoundError:
        with open("config.json") as f:
            config = Config(**json.load(f))

    model_name = os.path.basename(config.analysis.dataset)
    if os.path.exists(
        out_folder := os.path.join(config.analysis.folder_output, model_name)
    ):
        r = Results(out_folder)
    else:
        logging.critical("No results folder found!")
