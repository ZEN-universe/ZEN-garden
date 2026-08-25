"""This module contains the Results class, which is used to extract and process
the results of a model run.
"""

import logging
from pathlib import Path
from typing import Literal, cast

import pandas as pd
from pandas import Series
from pint import UnitRegistry
from typing_extensions import override

from zen_garden.default_config import Analysis, Solver, System
from zen_garden.postprocess.results.scenario import Scenario
from zen_garden.postprocess.results.solution_loader import SolutionLoader

logger = logging.getLogger(__name__)

NestedTuple = tuple[list[str], ...] | tuple[str, ...]
NestedDict = dict[str, str | list[str]]


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

        self.solution_loader: SolutionLoader = SolutionLoader(path)
        self.has_scenarios: bool = len(self.solution_loader.scenarios) > 1
        self.name: str = self.solution_loader.name
        self.ureg: UnitRegistry = self.solution_loader.ureg

    @override
    def __str__(self) -> str:
        return (
            f"Results of '{self.solution_loader.name}' "
            f"with scenarios: {list(self.scenarios.keys())}"
        )

    @property
    def scenarios(self) -> dict[str, Scenario]:
        """Returns the scenarios of the results.

        :return: Dictionary of scenarios
        """
        return self.solution_loader.scenarios

    @property
    def first_scenario(self) -> Scenario:
        """Returns the first scenario in the loaded results.

        :return: First scenario
        """
        return self.solution_loader.first_scenario

    def __getitem__(self, key: str) -> Scenario:
        """Returns the scenario with the given key.

        Example:
            This syntax allows for easy access to specific scenarios and their data::

                res = Results("<result_folder>")
                res["scenario_1"].get_df("capacity")

        """
        return self.solution_loader.scenarios[key]

    def get_df(
        self,
        component_name: str,
        scenario_name: str | None = None,
        index: dict[str, str] | None = None,
    ) -> pd.Series:
        """Returns the raw results without any further processing.

        Transforms a parameter or variable dataframe (compressed) string into
        an actual pandas dataframe.

        Args:
            component_name (string): The string to decode
            scenario_name: Which scenario to take. If none is specified, all are
                returned.
            data_type: The type of data to extract. Either 'dataframe' or 'units'
            index: slicing index of the resulting dataframe

        Returns:
            DataFrame: The corresponding dataframe
        """
        scenario = self.solution_loader.find_scenario(scenario_name)
        return scenario.get_values(component_name, index)

    def get_full_ts(
        self,
        component_name: str,
        scenario_name: str | None = None,
        discount_to_first_step: bool = True,
        year: int | None = None,
        keep_raw: bool = False,
        index: dict[str, str] | None = None,
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
        # TODO: Maybe revert decision to load only first scenario
        #       if scenario_name is None.

        scenario = self.solution_loader.find_scenario(scenario_name)
        full_ts = scenario.get_full_ts(
            component_name,
            discount_to_first_step=discount_to_first_step,
            year=year,
            keep_raw=keep_raw,
            index=index,
        )
        return full_ts

    def get_total(
        self,
        component_name: str,
        year: int | None = None,
        scenario_name: str | None = None,
        keep_raw: bool = False,
        index: dict[str, str] | None = None,
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
        """
        if component_name in self.get_component_names("dual"):
            raise ValueError(
                (
                    "This method does not support the extraction of "
                    "dual variables. Please use the methods "
                    "`get_dual()` or `get_full_ts()` instead."
                )
            )

        scenario = self.solution_loader.find_scenario(scenario_name)
        return scenario.get_total(component_name, year, keep_raw, index)

    def get_dual(
        self,
        component_name: str,
        scenario_name: str | None = None,
        discount_to_first_step: bool = True,
        year: int | None = None,
        keep_raw: bool = False,
        index: dict[str, str] | None = None,
    ) -> pd.DataFrame | pd.Series | None:
        """Extracts the dual variables of a component.

        Args:
            component_name: Name of dual
            scenario_name: Scenario Name
            **kwargs: Additional arguments to pass to the get_full_ts method

        Returns:
            DataFrame: Duals of the component
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
            index,
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
        """
        scenario = self.solution_loader.find_scenario(scenario_name)
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
            >>> r.get_system('scenario_name') # system configuration of "scenario_name"

        """
        scenario = self.solution_loader.find_scenario(scenario_name)
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
            >>> r.get_analysis('scenario_name') # analysis config of "scenario_name"

        """
        scenario = self.solution_loader.find_scenario(scenario_name)
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
            >>> r.get_solver('scenario_name') # solver configuration of "scenario_name"

        """
        scenario = self.solution_loader.find_scenario(scenario_name)
        return scenario.solver

    def get_doc(
        self, component_name: str, scenario_name: str | None = None
    ) -> str | None:
        """Extracts the documentation of a given Component.

        :param component_name: Name of the component
        :return: The corresponding documentation
        """
        scenario = self.solution_loader.find_scenario(scenario_name)
        return scenario.get_doc(component_name)

    def get_index_names(
        self, component_name: str, scenario_name: str | None = None
    ) -> list[str]:
        """Docstring for get_index_names.

        :param self: Description
        :param component_name: Description
        :type component_name: str
        :param scenario_name: Description
        :type scenario_name: Optional[str]
        :return: Description
        :rtype: list[str]
        """
        # TODO: read out the index names from the docstring
        return [
            str(name) for name in self.get_df(component_name, scenario_name).index.names
        ]

    def get_years(self, scenario_name: str | None = None) -> list[int]:
        """Extracts the years of a given Scenario. If no scenario is given, a
        random one is taken.

        :param scenario_name: Name of the scenario
        :return: List of years
        """
        scenario = self.solution_loader.find_scenario(scenario_name)
        return list(range(0, scenario.system.optimized_years))

    def has_MF(self, scenario_name: str | None = None) -> bool:
        """Whether the given scenario uses rolling horizon optimization.
        If no scenario is given, the first one is taken.

        :param scenario_name: Name of the scenario.
            Defaults to the first scenario if None.
        :return: Boolean indicating whether the scenario
            uses rolling horizon optimization.
        """
        scenario = self.solution_loader.find_scenario(scenario_name)
        return scenario.has_rh

    def get_coords(self, scenario_name: str | None = None) -> pd.DataFrame | None:
        """Extracts the coordinates of the nodes of a given Scenario. If no
        scenario is given, a random one is taken.

        :param scenario_name: Name of the scenario
        :return: The corresponding coordinates
        """
        scenario = self.solution_loader.find_scenario(scenario_name)
        coords = pd.DataFrame(scenario.system.coords).T
        if coords.empty:
            logger.warning(
                (
                    f"Coordinates of nodes are not saved for version "
                    f"{scenario.analysis.zen_garden_version}."
                )
            )
            return None
        return pd.DataFrame(scenario.system.coords).T

    def extract_carrier(
        self, series: pd.Series, carrier: str, scenario_name: str
    ) -> pd.Series:
        """Returns a dataframe that only contains the desired carrier.
        If carrier is not contained in the dataframe, the technologies that
        have the provided reference carrier are returned.

        :param dataframe: pd.Dataframe containing the base data
        :param carrier: name of the carrier
        :param scenario_name: name of the scenario
        :return: filtered pd.Dataframe containing only the provided carrier
        """
        if "carrier" in series.index.names:
            return series.xs(carrier, level="carrier", drop_level=False)

        reference_carriers = self.get_df("set_reference_carriers", scenario_name)
        assert isinstance(reference_carriers, pd.Series)

        technologies_with_carrier = reference_carriers[reference_carriers == carrier]
        return series[
            series.index.get_level_values("technology").isin(
                technologies_with_carrier.index
            )
        ]

    def get_component_names(
        self,
        component_type: (
            Literal["sets", "variable", "parameter", "dual", "reduced_cost"] | None
        ),
        scenario_name: str | None = None,
    ) -> list[str]:
        """Returns the names of all components of a given type.

        :param component_type: Type of the component
        :return: List of component names
        """
        scenario = self.solution_loader.find_scenario(scenario_name)
        if component_type is None:
            return scenario.component_map.all_components
        return cast(list[str], getattr(scenario.component_map, component_type))
