from abc import ABC, abstractmethod, abstractproperty
from enum import Enum
from zen_garden.model.default_config import Analysis, System
import pandas as pd
from typing import Any, Optional


class ComponentType(Enum):
    parameter: str = "parameter"
    variable: str = "variable"
    dual: str = "dual"
    sets: str = "sets"


class TimestepType(Enum):
    yearly: str = "year"
    operational: str = "time_operation"
    storage: str = "time_storage_level"


class MF(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_series(self) -> "pd.Series[Any]":
        pass


class Component(ABC):
    @abstractmethod
    def get_levels(self) -> list[str]:
        pass

    @abstractproperty
    def component_type(self) -> ComponentType:
        pass

    @abstractproperty
    def timestep_type(self) -> Optional[TimestepType]:
        pass

    @abstractproperty
    def mf_steps(self) -> dict[int, MF]:
        pass

    @abstractproperty
    def has_mf(self) -> bool:
        pass

    @abstractproperty
    def timestep_name(self) -> Optional[str]:
        pass

    def get_mf_aggregated_series(self) -> "pd.Series[Any]":
        aggregation_necessary = self.timestep_type is not None

        if aggregation_necessary:
            dataframes = [mf.get_series() for _, mf in self.mf_steps.items()]
            return pd.concat(dataframes)
        else:
            first_mf = self.mf_steps[0]
            return first_mf.get_series()


class Scenario(ABC):
    def __init__(self) -> None:
        pass

    @abstractproperty
    def analysis(self) -> Analysis:
        pass

    @abstractproperty
    def system(self) -> System:
        pass

    @abstractproperty
    def components(self) -> dict[str, Component]:
        pass

    def get_timestep_duration(self, ts_type: TimestepType) -> "pd.Series[Any]":
        if ts_type.value == TimestepType.storage.value:
            relevant_component = self.components["time_steps_storage_level_duration"]
        elif ts_type.value == TimestepType.operational.value:
            relevant_component = self.components["time_steps_operation_duration"]
        else:
            raise ValueError("Timestep duration of yearly component!")

        return relevant_component.get_mf_aggregated_series().unstack().iloc[0]

    def get_mf_aggregated_series(self, component_name: str) -> "pd.Series[Any]":
        return self.components[component_name].get_mf_aggregated_series()


class SolutionLoader(ABC):
    def __init__(self) -> None:
        pass

    @abstractproperty
    def scenarios(self) -> dict[str, Scenario]:
        pass

    def __getitem__(self, scenario_name: str) -> Scenario:
        return self.scenarios[scenario_name]

    @abstractproperty
    def component_names(self) -> list[str]:
        pass

    @abstractmethod
    def get_time_steps_year2operation(self, year: int, is_storage: bool) -> Any:
        pass
