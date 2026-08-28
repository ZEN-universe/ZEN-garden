import functools
import logging
import os
import time
from typing import TYPE_CHECKING, Callable

import psutil

from zen_garden.di import ServiceContainer
from zen_garden.model.constructor import ModelConstructor

if TYPE_CHECKING:
    from zen_garden.model.optimization_model import OptimizationModel
    from zen_garden.model.schema import ModelSchema

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
        service_container: "ServiceContainer",
        model_schema: "ModelSchema",
        optimization_model: "OptimizationModel",
    ):
        self.service_container = service_container
        self.model_schema = model_schema
        self.optimization_model = optimization_model

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    def construct_model(self):
        """Logic to construct a model based on the provided name and parameters."""
        # model_schema.element_classes is the ordered list of every element type
        # (EnergySystem first, then ELEMENT_TYPE_CLASSES). One generic
        # ModelConstructor is built per type.
        self._model_constructors = [
            # Injected services: service_container, element_registry,
            # optimization_model, model_schema, network_topology, time_steps;
            # explicit argument: element_class.
            self.service_container.build(ModelConstructor, element_class=element_class)
            for element_class in self.model_schema.element_classes
        ]
        # Filter out model constructors that do not have any elements to construct
        self._model_constructors = [
            constructor
            for constructor in self._model_constructors
            if constructor.has_elements()
        ]

        self._construct_sets()
        self._construct_params()
        self._construct_vars()
        self._construct_expressions()
        self._construct_constraints()
        self._construct_objective()

    @measure_run_time
    def _construct_sets(self):
        for model_constructor in self._model_constructors:
            model_constructor.construct_sets()

    @measure_run_time
    def _construct_params(self):
        for model_constructor in self._model_constructors:
            model_constructor.construct_params()

    @measure_run_time
    def _construct_vars(self):
        for model_constructor in self._model_constructors:
            model_constructor.construct_vars()

    @measure_run_time
    def _construct_expressions(self):
        for model_constructor in self._model_constructors:
            model_constructor.construct_expressions()

    @measure_run_time
    def _construct_constraints(self):
        for model_constructor in self._model_constructors:
            model_constructor.construct_constraints()

    def _construct_objective(self):
        """Select the optimization objective from a registered expression.

        Objective candidates are built as energy-system expressions (see
        ``zen_garden.elements.energy_system.expressions``); ``config.analysis``
        picks which one to use and with which sense.
        """
        logger.info("Constructing objective")

        objective_name = self.config.analysis.objective
        if objective_name not in self.optimization_model.expressions:
            raise KeyError(f"Objective type {objective_name} not known")

        sense = self.config.analysis.sense
        assert sense in ("min", "max"), f"Objective sense {sense} not known"

        self.optimization_model.lp_model.add_objective(
            self.optimization_model.expressions[objective_name], sense=sense
        )
