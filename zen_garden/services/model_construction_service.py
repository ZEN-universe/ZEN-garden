import functools
import logging
import os
import time
from typing import TYPE_CHECKING, Callable

import psutil

from zen_garden.elements import ELEMENT_CONSTRUCTORS
from zen_garden.model.zen_model import ZenModel

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.model.config import Config
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.element_registry import ElementRegistry

logger = logging.getLogger(__name__)


def measure_run_time(func: Callable) -> Callable:
    """Decorator to measure the run time of a function."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not self.config.solver.run_diagnostics:
            return func(self, *args, **kwargs)

        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.debug(f"Function {func.__name__} took {run_time:.4f} seconds to run.")
        mem_usage = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        logger.debug(f"Memory usage: {mem_usage:0.1f} MB")
        return result

    return wrapper


class ModelConstructionService:
    def __init__(
        self,
        config: "Config",
        energy_system: "EnergySystem",
        element_registry: "ElementRegistry",
        unit_handling: "UnitHandling",
        time_steps: "TimeStepsDicts",
    ):
        self.config = config
        self.energy_system = energy_system
        self.element_registry = element_registry
        self.unit_handling = unit_handling
        self.time_steps = time_steps

    def construct_model(self) -> ZenModel:
        """Logic to construct a model based on the provided name and parameters."""
        self.zen_model = ZenModel(
            self.config, self.energy_system, self.unit_handling, self.element_registry
        )
        self.element_constructors = [
            ElementConstructor(
                self.config,
                self.element_registry,
                self.zen_model,
                self.energy_system,
                self.time_steps,
            )
            for ElementConstructor in ELEMENT_CONSTRUCTORS
        ]

        self._construct_sets()
        self._construct_params()
        self._construct_vars()
        self._construct_constraints()
        self._construct_objective()

        return self.zen_model

    @measure_run_time
    def _construct_sets(self):
        for element_constructor in self.element_constructors:
            if not element_constructor.has_elements():
                continue
            element_constructor.construct_sets()

    @measure_run_time
    def _construct_params(self):
        for element_constructor in self.element_constructors:
            if not element_constructor.has_elements():
                continue
            element_constructor.construct_params()

    @measure_run_time
    def _construct_vars(self):
        for element_constructor in self.element_constructors:
            if not element_constructor.has_elements():
                continue
            element_constructor.construct_vars()

    @measure_run_time
    def _construct_constraints(self):
        for element_constructor in self.element_constructors:
            if not element_constructor.has_elements():
                continue
            element_constructor.construct_constraints()

    def _construct_objective(self):
        for element_constructor in self.element_constructors:
            if not element_constructor.has_elements():
                continue
            element_constructor.construct_objective()
