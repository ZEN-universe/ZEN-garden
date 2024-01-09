from abc import ABC, abstractmethod, abstractproperty
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
    @property
    @abstractmethod
    def component_type(self) -> ComponentType:
        pass

    @property
    @abstractmethod
    def timestep_type(self) -> Optional[TimestepType]:
        pass


class Scenario(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def analysis(self) -> Analysis:
        pass

    @property
    @abstractmethod
    def system(self) -> System:
        pass


class SolutionLoader(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def scenarios(self) -> dict[str, Scenario]:
        pass

    @property
    @abstractmethod
    def components(self) -> dict[str, Component]:
        pass

    @property
    @abstractmethod
    def has_rh(self) -> bool:
        pass

    @abstractmethod
    def get_component_data(scenario: Scenario, component: Component) -> pd.Series:
        pass
