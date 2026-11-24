import functools
import logging
import os
import time
from typing import TYPE_CHECKING, Callable, Literal, TypeVar

import psutil

from zen_garden.model.component_types.component import GenericComponent
from zen_garden.model.component_types.constraint import GenericConstraint
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.component_types.parameter import GenericParameter
from zen_garden.model.component_types.set import (
    GenericDomainSet,
    GenericRelationSet,
    SetDeclaration,
)
from zen_garden.model.component_types.variable import GenericVariable
from zen_garden.model.constructor import ModelConstructor
from zen_garden.service_container import ServiceContainer

if TYPE_CHECKING:
    from zen_garden.model.optimization_model import OptimizationModel
    from zen_garden.model.schema import ModelSchema

logger = logging.getLogger(__name__)

ComponentSpec = TypeVar("ComponentSpec", bound=GenericComponent)
ComponentCollection = Literal[
    "sets", "parameters", "variables", "expressions", "constraints"
]


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
        # model_schema.element_classes_ordered is the ordered list of every element type
        # (EnergySystem first, then ELEMENT_TYPE_CLASSES). One generic
        # ModelConstructor is built per type.
        self._model_constructors = [
            # Injected services: service_container, element_registry,
            # optimization_model, model_schema, network_topology, time_steps;
            # explicit argument: element_class.
            self.service_container.build(ModelConstructor, element_class=element_class)
            for element_class in self.model_schema.element_classes_ordered
        ]
        # Filter out model constructors that do not have any elements to construct
        self._model_constructors = [
            constructor
            for constructor in self._model_constructors
            if constructor.has_elements()
        ]

        self._construct_sets()
        self._construct_params()
        self._construct_masks()
        self._construct_vars()
        self._construct_expressions()
        self._construct_constraints()
        self._construct_objective()

    @measure_run_time
    def _construct_sets(self):
        declarations = self._set_declarations()
        domains = [
            declaration
            for declaration in declarations
            if issubclass(declaration, GenericDomainSet)
        ]
        relations = [
            declaration
            for declaration in declarations
            if issubclass(declaration, GenericRelationSet)
        ]
        for domain in GenericDomainSet.construction_order(domains, ignore_missing=True):
            domain.build(declarations[domain])
        self.optimization_model.sets.validate_subsets()
        for relation in GenericRelationSet.construction_order(
            relations, ignore_missing=True
        ):
            relation.build(declarations[relation])

    def _set_declarations(self) -> dict[SetDeclaration, ModelConstructor]:
        """Collect mixed domain/relation declarations and their owners."""
        declarations: dict[SetDeclaration, ModelConstructor] = {}
        names: dict[str, SetDeclaration] = {}
        for model_constructor in self._model_constructors:
            for declaration in model_constructor.sets:
                if not issubclass(declaration, (GenericDomainSet, GenericRelationSet)):
                    raise TypeError(
                        f"{declaration.__name__} in 'sets' is not a domain or "
                        "relation declaration"
                    )
                existing = names.get(declaration.name)
                if existing is not None and existing is not declaration:
                    raise ValueError(
                        "Multiple set specifications define " f"{declaration.name!r}"
                    )
                names[declaration.name] = declaration
                declarations[declaration] = model_constructor
        return declarations

    @measure_run_time
    def _construct_params(self):
        for parameter, owner in self._ordered_components(
            "parameters", GenericParameter
        ):
            parameter.build(owner)

    @measure_run_time
    def _construct_masks(self):
        """Construct all masks referenced by variables, expressions, and constraints."""
        mask_owners = {}
        for model_constructor in self._model_constructors:
            for variable in model_constructor.variables:
                for mask in variable.masks:
                    mask_owners.setdefault(mask, model_constructor)
            for expression in model_constructor.expressions:
                for mask in expression.masks:
                    mask_owners.setdefault(mask, model_constructor)
            for constraint in model_constructor.constraints:
                for mask in constraint.masks:
                    mask_owners.setdefault(mask, model_constructor)
        self.optimization_model.masks.build_all(mask_owners)

    @measure_run_time
    def _construct_vars(self):
        for variable, owner in self._ordered_components("variables", GenericVariable):
            variable.build(owner)

    @measure_run_time
    def _construct_expressions(self):
        for expression, owner in self._ordered_components(
            "expressions", GenericExpression
        ):
            expression.build(owner)

    @measure_run_time
    def _construct_constraints(self):
        for constraint, owner in self._ordered_components(
            "constraints", GenericConstraint
        ):
            constraint.build(owner)

    def _ordered_components(
        self,
        collection: ComponentCollection,
        component_type: type[ComponentSpec],
    ) -> list[tuple[type[ComponentSpec], ModelConstructor]]:
        """Collect one component kind and return it in global dependency order."""
        owners: dict[str, ModelConstructor] = {}
        components: list[type[ComponentSpec]] = []
        for model_constructor in self._model_constructors:
            for component in getattr(model_constructor, collection):
                if not issubclass(component, component_type):
                    raise TypeError(
                        f"{component.__name__} in {collection!r} is not a "
                        f"{component_type.__name__}"
                    )
                if component.name in owners:
                    raise ValueError(
                        f"Multiple {collection} specifications define "
                        f"{component.name!r}"
                    )
                owners[component.name] = model_constructor
                components.append(component)

        ordered = component_type.construction_order(components)
        return [(component, owners[component.name]) for component in ordered]

    def _construct_objective(self):
        """Select the optimization objective from a registered expression.

        Objective candidates are built as energy-system expressions (see
        ``zen_garden.elements.energy_system.expressions``); ``config.analysis``
        picks which one to use and with which sense.
        """
        logger.info("Constructing objective")

        objective_name = self.config.analysis.objective
        if objective_name not in self.optimization_model.expressions:
            available = sorted(self.optimization_model.expressions)
            raise KeyError(
                f"Objective expression {objective_name!r} is not registered. "
                f"Available expressions: {available}"
            )

        specification = self.optimization_model.expressions.specifications[
            objective_name
        ]
        if specification.indices:
            raise ValueError(
                f"Objective expression {objective_name!r} declares dimensions "
                f"{specification.indices}. Define a scalar expression with an "
                "explicit sum or selection and use that expression as the objective."
            )
        objective = self.optimization_model.expressions[objective_name]
        public_dimensions = tuple(
            dimension for dimension in objective.dims if dimension != "_term"
        )
        if public_dimensions:
            raise ValueError(
                f"Objective expression {objective_name!r} has dimensions "
                f"{public_dimensions} despite declaring indices=(). Define a scalar "
                "expression with an explicit sum or selection."
            )

        sense = self.config.analysis.sense
        if sense not in ("min", "max"):
            raise ValueError(
                f"Objective sense {sense!r} is invalid; expected 'min' or 'max'."
            )

        objective_constant = float(objective.const.item())
        self.optimization_model.objective_constant = objective_constant
        self.optimization_model.lp_model.add_objective(
            objective - objective_constant, sense=sense
        )
