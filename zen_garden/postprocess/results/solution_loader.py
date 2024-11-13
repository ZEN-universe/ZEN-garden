"""
This module defines the abstract classes for the components, scenarios and solution loaders.
"""
from abc import ABC, abstractmethod
from enum import Enum
from zen_garden.model.default_config import Analysis, System, Solver
import pandas as pd
import pint
from typing import Optional, Any, Literal


class ComponentType(Enum):
    parameter: str = "parameter"
    variable: str = "variable"
    dual: str = "dual"
    sets: str = "sets"

    @classmethod
    def get_component_type_names(cls):
        return [component_type.value for component_type in cls]

class TimestepType(Enum):
    yearly: str = "year"
    operational: str = "time_operation"
    storage: str = "time_storage_level"


class Component(ABC):
    """
    Abstract Component-Class
    """

    @property
    @abstractmethod
    def component_type(self) -> ComponentType:
        """
        Abstract property that implements the type of a component. The possible component
        types are defined in the ComponentType-Enumy
        """
        pass

    @property
    @abstractmethod
    def timestep_type(self) -> Optional[TimestepType]:
        """
        Abstract property that implements the timestep-type of a component. The possible
        timestep-types are defined in the ComponentType-Enumy
        """
        pass

    @property
    @abstractmethod
    def timestep_name(self) -> Optional[str]:
        """
        Abstract property that implements the name of the timestep-index. Should return
        None if the component does not have a timestep.
        """
        pass

    @property
    @abstractmethod
    def file_name(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def doc(self) -> str:
        pass

    @property
    @abstractmethod
    def has_units(self) -> bool:
        pass

class Scenario(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def analysis(self) -> Analysis:
        """
        Abstract property that should return the Analysis-Config of a sceanrio.
        """
        pass

    @property
    @abstractmethod
    def solver(self) -> Solver:
        """
        Abstract property that should return the Solver-Config of a sceanrio.
        """
        pass

    @property
    @abstractmethod
    def system(self) -> System:
        """
        Abstract property that should return the System-Config of a sceanrio.
        """
        pass

    @property
    @abstractmethod
    def path(self) -> str:
        pass

    @property
    @abstractmethod
    def ureg(self) -> pint.UnitRegistry:
        pass

class SolutionLoader(ABC):
    """
    Abstract Solution-loader. An abstract class defines the structure of how a class
    should look like but does not include actual implementations.
    However, when another piece of code is using a solution loader it can expect that
    every solution loader has the same methods and properties implemented.

    For example we can write an implementation of a solution loader that loads solutions
    that are stored as HDF-Files and another one that loads solutions that are stored as
    JSON-Files. In other parts of the code base we do not have to care about the origin
    of the data, we simply want to be sure that we have an implementation of a
    SolutionLoader which guarantees that we have all the functionalities defined in the
    abstract class available.
    """

    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Abstract property that should define the name of the solution that is loaded.
        """
        pass

    @property
    @abstractmethod
    def scenarios(self) -> dict[str, Scenario]:
        """
        Abstract property should define a dictionary with the names of the scenario as
        keys and the implementations of the Scenarios as values.
        """
        pass

    @property
    @abstractmethod
    def components(self) -> dict[str, Component]:
        """
        Abstract property should define a dictionary with the names of the components as
        keys and the implementations of the Components as values.
        """
        pass

    @property
    @abstractmethod
    def has_rh(self) -> bool:
        """
        Abstract boolean property specifying if a solution has rolling horizon or not.
        """
        pass

    @property
    @abstractmethod
    def has_duals(self) -> bool:
        """
        Abstract boolean property specifying if a solution has duals or not.
        """
        pass

    @abstractmethod
    def get_component_data(
        self,
        scenario: Scenario,
        component: Component,
        keep_raw: bool = False,
        data_type: Literal["dataframe","units"] = "dataframe"
    ) -> "pd.DataFrame | pd.Series[Any]":
        """
        Abstract method that should return the component values of a given scenario and a
        given component.

        If the solution uses rolling horizon, the returned component values should already
        take into account the limited foresight and therefore should not include all the
        data of the different foresight years but combine them to one series, unless explicitly desired (keep_raw = True).
        """
        pass

    @abstractmethod
    def get_timestep_duration(
        self, scenario: Scenario, component: Component
    ) -> "pd.Series[Any]":
        """
        Abstract method that should return the timestep durations given a scenario and a
        component.
        """
        pass

    @abstractmethod
    def get_timesteps(
        self, scenario: Scenario, component: Component, year: int
    ) -> "pd.Series[Any]":
        """
        Abstract method that should return the timesteps given a scenario and a component.
        """
        pass

    @abstractmethod
    def get_sequence_time_steps(
        self, scenario: Scenario, timestep_type: TimestepType
    ) -> "pd.Series[Any]":
        pass

    @abstractmethod
    def get_optimized_years(
            self, scenario: Scenario
    ) -> list[int]:
        pass

    @abstractmethod
    def get_timesteps_of_years(
        self, scenario: Scenario, ts_type: TimestepType, years: tuple
    ) -> "pd.DataFrame | pd.Series[Any]":
        pass

    def has_scenarios(self) -> bool:
        return len(self.scenarios) > 1
