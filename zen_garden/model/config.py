from dataclasses import dataclass

from zen_garden.default_config import Analysis, Solver, System
from zen_garden.model.utils import freeze


@dataclass(frozen=True, slots=True)
class Config:
    """
    A class to represent the configuration of the optimization problem.

    Attributes:
        analysis (dict): The analysis dictionary.
        solver (dict): The solver dictionary.
        plugins (list): The list of plugins.
    """

    analysis: Analysis
    system: System
    solver: Solver

    @classmethod
    def from_setup(cls, analysis: Analysis, system: System, solver: Solver) -> "Config":
        """
        Creates a Config instance from the given analysis, solver, and plugins.

        Args:
            analysis (dict): The analysis dictionary.
            system (dict): The system dictionary.
            solver (dict): The solver dictionary.

        Returns:
            Config: A new instance of the Config class.
        """

        return cls(
            analysis=freeze(analysis), system=freeze(system), solver=freeze(solver)
        )
