"""ZEN-model to combine sets, parameters, expressions, variables and constraints
from all elements into a single model."""

from typing import TYPE_CHECKING, Any

from linopy import Model as LinopyModel

from zen_garden.model.components.constraint import Constraint
from zen_garden.model.components.parameter import Parameter
from zen_garden.model.components.set_registry import SetRegistry
from zen_garden.model.components.variable import Variable

if TYPE_CHECKING:
    from zen_garden.model.config import Config
    from zen_garden.services.service_container import ServiceContainer


class ZenModel:
    """Store optimization-model sets, parameters, variables, and constraints."""

    def __init__(
        self,
        service_container: "ServiceContainer",
        config: "Config",
    ):
        self.indexing_sets = [key for key in config.system.keys() if "set" in key]

        self.lp_model = LinopyModel(solver_dir=config.solver.solver_dir)
        self.sets = service_container.build(
            SetRegistry, indexing_sets=self.indexing_sets
        )
        self.variables = service_container.build(
            Variable, lp_model=self.lp_model, sets=self.sets
        )
        self.parameters = Parameter(sets=self.sets)
        # Expressions are model-construction artifacts rather than input data.
        self.expressions: dict[str, Any] = {}
        self.constraints = Constraint(lp_model=self.lp_model)

    def add_set(self, *args, **kwargs):
        """Add sets to the model.
        See :meth:`zen_garden.model.components.set_registry.SetRegistry.add_set`.
        """
        self.sets.add_set(*args, **kwargs)

    def create_custom_set(self, *args, **kwargs):
        """Create custom sets in the model.
        See
        :meth:`zen_garden.model.components.set_registry.SetRegistry.create_custom_set`.
        """
        return self.sets.create_custom_set(*args, **kwargs)

    def add_variable(self, *args, **kwargs):
        """Add variables to the model.
        See :meth:`zen_garden.model.components.variable.Variable.add_variable`.
        """
        self.variables.add_variable(*args, **kwargs)

    def add_parameter(self, *args, **kwargs):
        """Add parameters to the model.
        See :meth:`zen_garden.model.components.parameter.Parameter.add_parameter`.
        """
        self.parameters.add_parameter(*args, **kwargs)

    def add_expression(self, name: str, expression: Any) -> None:
        """Register a reusable expression created during model construction."""
        if name in self.expressions:
            raise ValueError(f"Expression {name!r} already added")
        self.expressions[name] = expression

    def add_constraint(self, *args, **kwargs):
        """Add constraints to the model.
        See :meth:`zen_garden.model.components.constraint.Constraint.add_constraint`.
        """
        self.constraints.add_constraint(*args, **kwargs)
