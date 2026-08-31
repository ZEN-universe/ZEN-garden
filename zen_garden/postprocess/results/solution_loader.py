"""Contains the implementation of a SolutionLoader that reads the solution."""

import json
import logging
from pathlib import Path
from typing import Any

from pint import UnitRegistry

from zen_garden.postprocess.results.scenario import Scenario
from zen_garden.postprocess.results.scenario_v1 import ScenarioV1
from zen_garden.postprocess.results.scenario_v2 import ScenarioV2
from zen_garden.postprocess.results.scenario_v3 import ScenarioV3

logger = logging.getLogger(__name__)

CURRENT_OUTPUT_VERSION = 4

# Result folders written by older ZEN-garden versions use a different on-disk
# layout. Each such layout has a dedicated :class:`Scenario` subclass that only
# overrides the file access. This maps the detected output version to the
# reader that understands that layout; versions absent here are read with the
# current :class:`Scenario`.
LEGACY_SCENARIO_CLASSES: dict[int, type[Scenario]] = {
    ScenarioV1.OUTPUT_VERSION: ScenarioV1,
    ScenarioV2.OUTPUT_VERSION: ScenarioV2,
    ScenarioV3.OUTPUT_VERSION: ScenarioV3,
}


class SolutionLoader:
    """Implementation of a SolutionLoader."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path

        self._scenarios: dict[str, Scenario] = self._read_scenarios()
        self._ureg: UnitRegistry = (
            self.first_scenario.ureg
        )  # pyright:ignore[reportUnknownMemberType]
        self._output_version: int = self.first_scenario.output_version

        if (
            self._output_version < CURRENT_OUTPUT_VERSION
            and self._output_version not in LEGACY_SCENARIO_CLASSES
        ):
            raise ValueError(
                (
                    f"Output version {self._output_version} is not supported. "
                    f"Please use an newer version of ZEN-garden."
                )
            )

    def _build_scenario(self, path: Path, name: str, base_scenario: str) -> Scenario:
        """Instantiate the scenario reader matching the folder's output version.

        Construction of a :class:`Scenario` does no I/O, so a lightweight probe
        instance is used to detect the output version before picking the
        (possibly legacy) reader class.

        :param path: The path to the scenario folder.
        :param name: The name of the scenario.
        :param base_scenario: The name of the base scenario.
        :return: A :class:`Scenario` (or subclass) instance for the folder.
        """
        probe = Scenario(path, name, base_scenario)
        scenario_class = LEGACY_SCENARIO_CLASSES.get(probe.output_version)
        if scenario_class is None:
            return probe
        logger.warning(
            "Scenario %s has an outdated output version (%s).\n"
            "Using the legacy reader %s, which will not be maintained in the "
            "future.\nPlease consider re-running the model with the current "
            "ZEN-garden version to update the outputs.",
            name,
            probe.output_version,
            scenario_class.__name__,
        )
        return scenario_class(path, name, base_scenario)

    def _read_scenarios(self) -> dict[str, Scenario]:
        """Create the scenario instances. The definitions of the scenarios are
        stored in the scenarios.json files. If the solution does not have
        multiple scenarios, we store the solution as "none".
        """
        scenarios: dict[str, Scenario] = {}
        with open(self.path / "scenarios.json", "r") as f:
            scenario_configs: dict[str, dict[str, Any]] = json.load(f)

        if len(scenario_configs) == 1:
            return {"none": self._build_scenario(self.path, "none", "")}

        for id, config in scenario_configs.items():
            path = self.path / f"scenario_{config['base_scenario']}"
            # For list-expansion scenarios, we store the results in a subfolder
            # of the base scenario.
            scenario_subfolder = config["sub_folder"]
            if scenario_subfolder != "":
                path = path / f"scenario_{scenario_subfolder}"

            if not (path / "analysis.json").exists():
                logger.warning(f"Scenario `scenario_{id}` does not exist. Skipping it.")
                continue

            name = f"scenario_{id}"
            base_scenario: str = config["base_scenario"]

            scenarios[name] = self._build_scenario(path, name, base_scenario)

        return scenarios

    @property
    def scenarios(self) -> dict[str, Scenario]:
        return self._scenarios

    @property
    def name(self) -> str:
        return Path(self.first_scenario.analysis.dataset).name

    @property
    def ureg(self) -> UnitRegistry:
        return self._ureg

    @property
    def has_duals(self) -> bool:
        return self.first_scenario.solver.save_duals

    @property
    def has_parameters(self) -> bool:
        return (
            not hasattr(self.first_scenario.solver, "save_parameters")
            or self.first_scenario.solver.save_parameters
        )

    def find_scenario(self, scenario_name: str | None) -> Scenario:
        """Find the scenario with the given name or raise exception.

        :param scenario_name: Name of the scenario
        :return: Scenario instance for the given name
        """
        if scenario_name is None:
            return self.first_scenario
        elif scenario_name in self.scenarios:
            return self.scenarios[scenario_name]
        else:
            raise ValueError(f"Scenario `{scenario_name}` not found.")

    #### Helper functions
    @property
    def first_scenario(self) -> Scenario:
        """Returns the first scenario of the dictionary of scenarios.

        :return: The first scenario of the dictionary.
        """
        return self._scenarios[next(iter(self._scenarios.keys()))]
