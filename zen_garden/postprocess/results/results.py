"""This module contains the Results class, which is used to extract and process
the results of a model run.
"""
import logging
from pathlib import Path
from typing import Literal, cast, overload

import pandas as pd
from pandas import Series
from typing_extensions import override

from zen_garden.default_config import Analysis, Solver, System
from zen_garden.postprocess.results.cost_emission_calculation import (
    CostEmissionCalculation,
)
from zen_garden.postprocess.results.scenario import Index, Scenario
from zen_garden.postprocess.results.solution_loader import SolutionLoader

logger = logging.getLogger(__name__)

# used for the cost and emission calculation to specify the mode of calculation
CostEmissionMode = Literal["final_demand", "total_production", "relative"]

class Results:
    """The Results class is used to extract and process the results of a model run."""

    def __init__(self, path: Path | str):
        """Initializes the Results class.

        :param path: Path to the results folder
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"The output folder {path.absolute()} does not exist."
            )
        elif not path.is_dir():
            raise NotADirectoryError(
                f"The output folder {path.absolute()} is not a directory."
            )
        elif len(list(path.glob("*"))) == 0:
            raise ValueError(
                f"The output folder {path.absolute()} does not contain any files."
            )

        self._solution_loader: SolutionLoader = SolutionLoader(path)
        self.name: str = self._solution_loader.name
        self.path: Path = path
        self.has_scenarios: bool = len(self._solution_loader.scenarios) > 1
        self._cost_emission_calculation: CostEmissionCalculation = (
            CostEmissionCalculation(self))

    @override
    def __str__(self) -> str:
        return (
            f"Results of '{self._solution_loader.name}' "
            f"with scenarios: {list(self.scenarios.keys())}"
        )

    @property
    def scenarios(self) -> dict[str, Scenario]:
        """Returns the scenarios of the results.

        :return: Dictionary of scenarios
        """
        return self._solution_loader.scenarios

    @property
    def first_scenario(self) -> Scenario:
        """Returns the first scenario in the loaded results.

        :return: First scenario
        """
        return self._solution_loader.first_scenario

    def __getitem__(self, scenario_name: str) -> Scenario:
        """Returns the scenario with the given scenario_name.

        Args:
            scenario_name (str): The scenario_name of the scenario to retrieve.

        Example:
            This syntax allows for easy access to specific scenarios and their data::

                res = Results("<result_folder>")
                res["scenario_1"].get_total("capacity")

        """
        return self._solution_loader.scenarios[scenario_name]

    def get_total(
        self,
        component_name: str,
        year: int | None = None,
        scenario_name: str | None = None,
        keep_raw: bool = False,
        index: Index | None = None,
    ) -> pd.DataFrame | pd.Series:
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
        if component_name in self.get_component_names("dual"):
            raise ValueError(
                (
                    "This method does not support the extraction of "
                    "dual variables. Please use the methods "
                    "`get_dual()` or `get_full_ts()` instead."
                )
            )

        if scenario_name is not None or len(self.scenarios) == 1:
            scenario = (
                self._solution_loader.find_scenario(scenario_name)
                if scenario_name is not None
                else self.first_scenario
            )
            return scenario.get_total(component_name, year, keep_raw, index)

        df_dict = {
            name: scenario.get_total(component_name, year, keep_raw, index)
            for name, scenario in self.scenarios.items()
        }
        return self._concatenate_scenarios(df_dict)

    def get_full_ts(
        self,
        component_name: str,
        scenario_name: str | None = None,
        discount_to_first_step: bool = True,
        year: int | None = None,
        keep_raw: bool = False,
        index: Index | None = None,
    ) -> pd.DataFrame | pd.Series:
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
        if scenario_name is not None or len(self.scenarios) == 1:
            scenario = (
                self._solution_loader.find_scenario(scenario_name)
                if scenario_name is not None
                else self.first_scenario
            )
            return scenario.get_full_ts(
                component_name,
                discount_to_first_step=discount_to_first_step,
                year=year,
                keep_raw=keep_raw,
                index=index,
            )

        df_dict = {
            name: scenario.get_full_ts(
                component_name,
                discount_to_first_step=discount_to_first_step,
                year=year,
                keep_raw=keep_raw,
                index=index,
            )
            for name, scenario in self.scenarios.items()
        }
        return self._concatenate_scenarios(df_dict)
    
    @overload
    def get_unprocessed_result(
        self,
        component_name: str,
        scenario_name: str,
        index: Index | None = None,
    ) -> pd.Series: ...

    @overload
    def get_unprocessed_result(
        self,
        component_name: str,
        scenario_name: None = None,
        index: Index | None = None,
    ) -> pd.Series | dict[str, pd.Series]: ...

    def get_unprocessed_result(
        self,
        component_name: str,
        scenario_name: str | None = None,
        index: Index | None = None,
    ) -> pd.Series | dict[str, pd.Series]:
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
            >>> r.get_unprocessed_result('<component_name>') # dataframe of "<component_name>"
            >>> r.get_unprocessed_result('<component_name>', '<scenario_name>') # dataframe of
                "<component_name>" in "<scenario_name>"
            >>> r.get_unprocessed_result('<component_name>', index={'<index_name>': '<index_value>'})
                # dataframe of "<component_name>" for a specific index value to slice the
                dataframe

        """
        if scenario_name is not None or len(self.scenarios) == 1:
            scenario = (
                self._solution_loader.find_scenario(scenario_name)
                if scenario_name is not None
                else self.first_scenario
            )
            return scenario.get_values(component_name, index)

        return {
            name: scenario.get_values(component_name, index)
            for name, scenario in self.scenarios.items()
        }

    def get_dual(
        self,
        component_name: str,
        scenario_name: str | None = None,
        year: int | None = None,
        discount_to_first_step: bool = True,
        keep_raw: bool = False,
        index: Index | None = None,
    ) -> pd.DataFrame | pd.Series | None:
        """Extracts the dual variables of a component.

        Args:
            component_name: Name of dual
            scenario_name: Scenario Name
            **kwargs: Additional arguments to pass to the get_full_ts method

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
        if not self.get_solver(scenario_name).save_duals:
            logger.warning(f"Duals are not calculated for `{scenario_name}`. Skip.")
            return None

        return self.get_full_ts(
            component_name,
            scenario_name,
            discount_to_first_step,
            year,
            keep_raw,
            index
        )

    def get_unit(
        self,
        component_name: str,
        scenario_name: str | None = None,
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
        scenario = self._solution_loader.find_scenario(scenario_name)
        return scenario.get_unit(component_name, convert_to_yearly_unit)

    def get_system(self, scenario_name: str | None = None) -> System:
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
        scenario = self._solution_loader.find_scenario(scenario_name)
        return scenario.system

    def get_analysis(self, scenario_name: str | None = None) -> Analysis:
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
        scenario = self._solution_loader.find_scenario(scenario_name)
        return scenario.analysis

    def get_solver(self, scenario_name: str | None = None) -> Solver:
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
        scenario = self._solution_loader.find_scenario(scenario_name)
        return scenario.solver

    def get_doc(
        self, component_name: str, scenario_name: str | None = None
    ) -> str | None:
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
        scenario = self._solution_loader.find_scenario(scenario_name)
        return scenario.get_doc(component_name)

    def get_index_names(
        self, component_name: str, scenario_name: str | None = None
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
        scenario = self._solution_loader.find_scenario(scenario_name)
        return scenario.get_index_names(component_name)

    def get_years(self, scenario_name: str | None = None) -> list[int]:
        """Extracts the years of a given Scenario. If no scenario is given, a
        random one is taken.

        Args:
            scenario_name (str, optional): The name of the scenario for which
                to extract the years. If no value is given, then the first
                scenario is used. Default value: ``None``.

        Returns:
            list[int]: A list of years for the specified scenario.
        """
        scenario = self._solution_loader.find_scenario(scenario_name)
        ref_year = scenario.system.reference_year
        interval_between_years = scenario.system.interval_between_years
        return [
            ref_year + i * interval_between_years 
            for i in range(scenario.system.optimized_years)]

    def has_RH(self, scenario_name: str | None = None) -> bool:
        """Whether the given scenario uses rolling horizon optimization.
        If no scenario is given, the first one is taken.

        :param scenario_name: Name of the scenario.
            Defaults to the first scenario if None.
        :return: Boolean indicating whether the scenario
            uses rolling horizon optimization.
        """
        scenario = self._solution_loader.find_scenario(scenario_name)
        return scenario.has_rh

    def get_component_names(
        self,
        component_type: (
            Literal["sets", "variable", "parameter", "dual", "reduced_cost"] | None
        ),
        scenario_name: str | None = None,
    ) -> list[str]:
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
        scenario = self._solution_loader.find_scenario(scenario_name)
        if component_type is None:
            return scenario.component_map.all_components
        return cast(list[str], getattr(scenario.component_map, component_type))

    def _concatenate_scenarios(
        self, df_dict: dict[str, pd.DataFrame] | dict[str, pd.DataFrame | pd.Series]
    ) -> pd.DataFrame | pd.Series:
        """Concatenates the dataframes or series from different scenarios."""
        if len(df_dict) == 1:
            return next(iter(df_dict.values()))

        if all(isinstance(df, pd.Series) for df in df_dict.values()):
            return pd.concat(df_dict, axis=1)
        elif all(isinstance(df, pd.DataFrame) for df in df_dict.values()):
            return pd.concat(df_dict, keys=df_dict.keys())
        else:
            raise ValueError(
                (
                    "All values in df_dict must be of the same type "
                    "(either all DataFrames or all Series)."
                )
            )

    def get_sectoral_costs(
        self,
        scenario_name: str | None = None,
        carrier: str | None = None,
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
            scenario_name = next(iter(self._solution_loader.scenarios.keys()))
        sectoral_costs, direct_costs = (
            self._cost_emission_calculation.calculate_leontief_data(
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
        scenario_name: str | None = None,
        carrier: str | None = None,
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
            scenario_name = next(iter(self._solution_loader.scenarios.keys()))
        sectoral_emissions, direct_emissions = (
            self._cost_emission_calculation.calculate_leontief_data(
                scenario_name=scenario_name,
                carrier=carrier,
                spatially_resolved=spatially_resolved,
                mode=mode,
                overwrite=overwrite,
                is_cost=False
            )
        )
        return sectoral_emissions, direct_emissions