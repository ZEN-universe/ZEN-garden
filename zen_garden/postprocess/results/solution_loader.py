"""Contains the implementation of a SolutionLoader that reads the solution."""

import json
import logging
from pathlib import Path
from typing import Any

from pint import UnitRegistry

from zen_garden.postprocess.results.scenario import Scenario

logger = logging.getLogger(__name__)

CURRENT_OUTPUT_VERSION = 4


class SolutionLoader:
    """Implementation of a SolutionLoader."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path

        self._scenarios: dict[str, Scenario] = self._read_scenarios()
        self._ureg: UnitRegistry = (
            self.first_scenario.ureg
        )  # pyright:ignore[reportUnknownMemberType]
        self._output_version: int = self.first_scenario.output_version

        if self._output_version < CURRENT_OUTPUT_VERSION:
            raise ValueError(
                (
                    f"Output version {self._output_version} is not supported. "
                    f"Please use an older version of ZEN-garden "
                    f"or migrate your outputs folder."
                )
            )

    def _read_scenarios(self) -> dict[str, Scenario]:
        """Create the scenario instances. The definitions of the scenarios are
        stored in the scenarios.json files. If the solution does not have
        multiple scenarios, we store the solution as "none".
        """
        scenarios: dict[str, Scenario] = {}
        with open(self.path / "scenarios.json", "r") as f:
            scenario_configs: dict[str, dict[str, Any]] = json.load(f)

        if len(scenario_configs) == 1:
            return {"none": Scenario(self.path, "none", "")}

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

            scenarios[name] = Scenario(path, name, base_scenario)

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
