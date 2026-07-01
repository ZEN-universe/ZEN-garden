from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Mapping

from zen_garden.model.utils import freeze

if TYPE_CHECKING:
    from zen_garden.utils.scenario_dict import ScenarioDict


@dataclass(frozen=True, slots=True)
class Context:
    """
    A class to represent the context of the optimization problem.

    Attributes:
        config (dict): The configuration dictionary.
        scenario_dict (dict): The scenario dictionary.
        input_data_checks (dict): The input data checks dictionary.
    """

    paths: Mapping[str, Any]
    element_classes: Mapping[str, type]
    scenario_dict: "ScenarioDict"
    dict_elements: defaultdict[str, list[Any]]
    parameter_change_log: dict[str, Any]
    year_specific_ts: dict[int, dict[tuple[str, str], Any]]

    @classmethod
    def from_setup(
        cls,
        paths: Mapping[str, Any],
        element_classes: Mapping[str, type],
        scenario_dict: "ScenarioDict",
        dict_elements: defaultdict[str, list[Any]],
        parameter_change_log: dict[str, Any],
        year_specific_ts: dict[int, dict[tuple[str, str], Any]],
    ) -> "Context":
        """
        Creates a Context instance from the given paths.

        Args:
            paths (dict): The paths dictionary.

        Returns:
            Context: A new instance of the Context class.
        """

        return cls(
            paths=freeze(dict(paths)),
            element_classes=dict(element_classes),
            scenario_dict=scenario_dict,
            dict_elements=dict_elements,
            parameter_change_log=parameter_change_log,
            year_specific_ts=year_specific_ts,
        )

    def update(self, **kwargs) -> "Context":
        """
        Updates the Context instance with new values.

        Args:
            **kwargs: Key-value pairs to update the Context instance.

        Returns:
            Context: A new instance of the Context class with updated values.
        """

        return replace(self, **kwargs)
