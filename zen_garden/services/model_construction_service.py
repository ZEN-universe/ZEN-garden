import logging
from typing import TYPE_CHECKING

from zen_garden.model import ELEMENT_CONSTRUCTORS
from zen_garden.model.zen_model import ZenModel

if TYPE_CHECKING:
    from zen_garden.model.config import Config
    from zen_garden.model.context import Context
    from zen_garden.model.energy_system import EnergySystem
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.element_registry import ElementRegistry

logger = logging.getLogger(__name__)


class ModelConstructionService:
    def __init__(
        self,
        config: "Config",
        context: "Context",
        energy_system: "EnergySystem",
        element_registry: "ElementRegistry",
        unit_handling: "UnitHandling",
    ):
        self.config = config
        self.context = context
        self.energy_system = energy_system
        self.element_registry = element_registry
        self.unit_handling = unit_handling

        self.element_constructors = [
            ElementConstructor(config, context, element_registry)
            for ElementConstructor in ELEMENT_CONSTRUCTORS
        ]

    def construct_model(self) -> ZenModel:
        """Logic to construct a model based on the provided name and parameters."""
        zen_model = ZenModel(
            self.config, self.context, self.energy_system, self.unit_handling
        )
        self._construct_sets(zen_model)
        self._construct_params(zen_model)
        self._construct_vars(zen_model)
        self._construct_constraints(zen_model)
        self._construct_objective(zen_model)
        return zen_model

    def _construct_sets(self, zen_model: ZenModel):
        logger.info("Constructing sets...")
        self.energy_system.construct_sets(zen_model)
        for element_constructor in self.element_constructors:
            if not element_constructor.has_elements():
                continue
            logger.debug(
                "Constructing sets using: %s", element_constructor.__class__.__name__
            )
            element_constructor.construct_sets(zen_model, self.energy_system)

    def _construct_params(self, zen_model: ZenModel):
        logger.info("Constructing parameters...")
        self.energy_system.construct_params(zen_model)
        for element_constructor in self.element_constructors:
            if not element_constructor.has_elements():
                continue
            logger.debug(
                "Constructing parameters using: %s",
                element_constructor.__class__.__name__,
            )
            element_constructor.construct_params(zen_model, self.energy_system)

    def _construct_vars(self, zen_model: ZenModel):
        logger.info("Constructing variables...")
        self.energy_system.construct_vars(zen_model)
        for element_constructor in self.element_constructors:
            if not element_constructor.has_elements():
                continue
            logger.debug(
                "Constructing variables using: %s",
                element_constructor.__class__.__name__,
            )
            element_constructor.construct_vars(zen_model, self.energy_system)

    def _construct_constraints(self, zen_model: ZenModel):
        logger.info("Constructing constraints...")
        self.energy_system.construct_constraints(zen_model)
        for element_constructor in self.element_constructors:
            if not element_constructor.has_elements():
                continue
            logger.debug(
                "Constructing constraints using: %s",
                element_constructor.__class__.__name__,
            )
            element_constructor.construct_constraints(zen_model, self.energy_system)

    def _construct_objective(self, zen_model: ZenModel):
        logger.info("Constructing objective...")
        self.energy_system.construct_objective(zen_model)
