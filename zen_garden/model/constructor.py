"""Constructor for elements.

One :class:`ModelConstructor` instance is created per element *type*, given the
element class as a constructor argument. The element class is the single source
of truth for the sets, parameters, variables, expressions and constraints of
that type; they are read off it in :meth:`~ModelConstructor.__init__`.
Component-specific behavior is implemented by the
set/parameter/variable/expression/constraint classes themselves, so no
subclassing is needed. The optimization objective is a whole-model concern and
is set by ``ModelConstructionService`` instead.
"""

import logging
from typing import TYPE_CHECKING

from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.component_types.parameter import GenericParameter
from zen_garden.model.component_types.set import GenericSet
from zen_garden.model.component_types.variable import GenericVariable
from zen_garden.model.element import Element
from zen_garden.service_container import ServiceContainer

if TYPE_CHECKING:
    from zen_garden.elements.energy_system import EnergySystem
    from zen_garden.input.network_topology import NetworkTopology
    from zen_garden.model.element_registry import ElementRegistry
    from zen_garden.model.optimization_model import OptimizationModel
    from zen_garden.model.schema import ModelSchema
    from zen_garden.model.time_steps import TimeStepsDicts

logger = logging.getLogger(__name__)


class ModelConstructor:
    """Builds the model components (sets, parameters, variables, constraints).

    There is one constructor instance per element *type*, whereas there is one
    :class:`~zen_garden.model.element.Element` instance per concrete element
    (each carrier, each technology). The element class is passed to
    :meth:`__init__` and is the single source of truth for which parameters,
    variables, sets and constraints belong to the type.
    """

    # Default element class; the real one is passed to __init__. Not a ClassVar
    # so that __init__ can bind it per instance.
    element_class: "type[Element] | type[EnergySystem]" = Element
    constraints: list[type[GenericConstraint]] = []
    expressions: list[type[GenericExpression]] = []
    parameters: list[type[GenericParameter]] = []
    variables: list[type[GenericVariable]] = []
    sets: list[type[GenericSet]] = []
    always_construct: bool = True

    def __init__(
        self,
        service_container: "ServiceContainer",
        element_registry: "ElementRegistry",
        optimization_model: "OptimizationModel",
        model_schema: "ModelSchema",
        network_topology: "NetworkTopology",
        time_steps: "TimeStepsDicts",
        element_class: "type[Element] | type[EnergySystem] | None" = None,
    ):
        self.service_container = service_container
        self.element_registry = element_registry
        self.optimization_model = optimization_model
        self.model_schema = model_schema
        self.network_topology = network_topology
        self.time_steps = time_steps

        if element_class is not None:
            self.element_class = element_class
        element_class = self.element_class
        # The element class is the single source of truth for the components of
        # this type. ``own_*`` are the declarations defined at that class level
        # (not inherited); ``variables``/``constraints`` are per-class lists.
        self.parameters = element_class.__dict__.get("own_parameters", [])
        self.variables = element_class.__dict__.get("variables", [])
        self.sets = element_class.__dict__.get("own_sets", [])
        self.expressions = element_class.__dict__.get("expressions", [])
        self.constraints = element_class.__dict__.get("constraints", [])
        # A type may declare itself optional; then it is only built when at least
        # one element of it is configured (see :meth:`has_elements`).
        self.always_construct = getattr(element_class, "always_construct", True)

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    @property
    def energy_system(self):
        """Return the canonical energy-system element from the schema."""
        return self.model_schema.energy_system

    def has_elements(self) -> bool:
        """Check whether this constructor should run.

        Constructors are skipped entirely when this returns False (see
        :meth:`~zen_garden.model.construction_service.ModelConstructionService.construct_model`).
        Mandatory types (:attr:`always_construct`) always run; optional types
        run only when at least one element of :attr:`element_class` is
        registered.
        """
        if self.always_construct:
            return True
        return bool(self.element_registry.all_names_of_elements(self.element_class))

    def construct_sets(self):
        """Constructs the Sets of this class."""
        logger.info(f"Constructing sets for {self.element_class.__name__}")
        for model_set in self.sets:
            model_set.build(self)

    def construct_params(self):
        logger.info(f"Constructing parameters for {self.element_class.name}")

        # Build in dependency order so that derived parameters (which read other
        # already-registered model parameters in their build()) see their inputs.
        # ignore_missing: cross-type dependencies (e.g. on "distance") are only
        # relevant to the store_input_data pass and may not be in this list.
        ordered = GenericParameter.construction_order(
            self.parameters, ignore_missing=True
        )
        for parameter in ordered:
            parameter.build(self)

    def construct_vars(self):
        """Constructs the Vars of this class."""
        logger.info(f"Constructing variables for {self.element_class.__name__}")

        for variable in self.variables:
            variable.build(self)

    def construct_expressions(self):
        """Construct reusable linear expressions from parameters and variables."""
        logger.info(f"Constructing expressions for {self.element_class.__name__}")
        for expression in self.expressions:
            expression.build(self)

    def construct_constraints(self):
        """Constructs the Constraints of this class."""
        logger.info(f"Constructing constraints for {self.element_class.__name__}")

        for constraint in self.constraints:
            constraint.build(self)

    def create_custom_set(self, list_index: list[str]):
        """Creates custom set for model component. See
        :meth:`zen_garden.model.registries.set_registry.SetRegistry.create_custom_set`.

        :param list_index: list of names of indices
        :return: list_index: list of names of indices
        """
        return self.optimization_model.create_custom_set(list_index)
