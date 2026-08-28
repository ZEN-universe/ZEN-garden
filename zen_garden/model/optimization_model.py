"""The optimization model: the linopy model plus the component registries
(sets, parameters, expressions, variables, constraints) from all elements."""

from typing import TYPE_CHECKING, Any

from linopy import Model as LinopyModel

from zen_garden.model.registries.constraint import ConstraintRegistry
from zen_garden.model.registries.parameter import ParameterRegistry
from zen_garden.model.registries.set_registry import SetRegistry
from zen_garden.model.registries.variable import VariableRegistry

if TYPE_CHECKING:
    from zen_garden.model.schema import ModelSchema
    from zen_garden.service_container import ServiceContainer


class OptimizationModel:
    """Store optimization-model sets, parameters, variables, and constraints."""

    def __init__(
        self,
        service_container: "ServiceContainer",
        model_schema: "ModelSchema",
    ):
        self.model_schema = model_schema
        self.indexing_sets = [key for key in self.config.system.keys() if "set" in key]

        self.lp_model = LinopyModel(solver_dir=self.config.solver.solver_dir)
        # Injected services: model_schema, element_registry; explicit argument:
        # indexing_sets.
        self.sets = service_container.build(
            SetRegistry, indexing_sets=self.indexing_sets
        )
        # Injected services: unit_converter, model_schema, element_registry;
        # explicit arguments: lp_model and sets.
        self.variables = service_container.build(
            VariableRegistry, lp_model=self.lp_model, sets=self.sets
        )
        self.parameters = ParameterRegistry(sets=self.sets)
        # Expressions are model-construction artifacts rather than input data.
        self.expressions: dict[str, Any] = {}
        self.constraints = ConstraintRegistry(lp_model=self.lp_model)

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    def add_set(self, *args, **kwargs):
        """Add sets to the model.
        See :meth:`zen_garden.model.registries.set_registry.SetRegistry.add_set`.
        """
        self.sets.add_set(*args, **kwargs)

    def create_custom_set(self, *args, **kwargs):
        """Create custom sets in the model.
        See
        :meth:`zen_garden.model.registries.set_registry.SetRegistry.create_custom_set`.
        """
        return self.sets.create_custom_set(*args, **kwargs)

    def add_variable(self, *args, **kwargs):
        """Add variables to the model.
        See :meth:`zen_garden.model.registries.variable.VariableRegistry.add_variable`.
        """
        self.variables.add_variable(*args, **kwargs)

    def add_parameter(self, *args, **kwargs):
        """Add parameters to the model.
        See :meth:`~zen_garden.model.registries.parameter.ParameterRegistry`.
        """
        self.parameters.add_parameter(*args, **kwargs)

    def add_expression(self, name: str, expression: Any) -> None:
        """Register a reusable expression created during model construction."""
        if name in self.expressions:
            raise ValueError(f"Expression {name!r} already added")
        self.expressions[name] = expression

    def add_constraint(self, *args, **kwargs):
        """Add constraints to the model.
        See :meth:`~zen_garden.model.registries.constraint.ConstraintRegistry`.
        """
        self.constraints.add_constraint(*args, **kwargs)
