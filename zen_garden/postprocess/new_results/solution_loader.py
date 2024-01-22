from abc import ABC, abstractmethod
from enum import Enum
from zen_garden.model.default_config import Analysis, System
import pandas as pd
from typing import Optional


class ComponentType(Enum):
    parameter: str = "parameter"
    variable: str = "variable"
    dual: str = "dual"
    sets: str = "sets"


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
    def system(self) -> System:
        """
        Abstract property that should return the System-Config of a sceanrio.
        """
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

    @abstractmethod
    def get_component_data(self, scenario: Scenario, component: Component) -> pd.Series:
        """
        Abstract method that should return the component values of a given scenario and a
        given component.

        If the solution uses rolling horizon, the returned component values should already
        take into account the limited foresight and therefore should not include all the
        data of the different foresight years but combine them to one series.
        """
        pass

    @abstractmethod
    def get_timestep_duration(
        self, scenario: Scenario, component: Component
    ) -> pd.Series:
        """
        Abstract method that should return the timestep durations given a scenario and a
        component.
        """
        pass

    @abstractmethod
    def get_timesteps(
        self, scenario: Scenario, component: Component, year: int
    ) -> pd.Series:
        """
        Abstract method that should return the timesteps given a scenario and a component.
        """
        pass
