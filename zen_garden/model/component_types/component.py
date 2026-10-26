"""Metadata shared by declarative optimization-model components."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, ClassVar, Sequence, TypeVar

if TYPE_CHECKING:
    from zen_garden.model.constructor import ModelConstructor

ComponentType = TypeVar("ComponentType", bound="GenericComponent")


class GenericComponent(ABC):
    """Base metadata shared by model-component specifications."""

    name: ClassVar[str]
    doc: ClassVar[str]
    dependencies: ClassVar[tuple[type["GenericComponent"], ...]] = ()
    _abstract_component: ClassVar[bool] = True

    @staticmethod
    def _indices_are_nonempty(
        model_constructor: "ModelConstructor", indices: Sequence[str]
    ) -> bool:
        """Return whether every registered index contains at least one member."""
        sets = model_constructor.optimization_model.sets
        return all(index not in sets or bool(sets[index]) for index in indices)

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get("_abstract_component", False):
            return
        name = cls.__dict__.get("name")
        if not isinstance(name, str) or not name.strip():
            raise TypeError(f"{cls.__name__}.name must be a non-empty string")
        doc = cls.__dict__.get("doc", cls.__doc__)
        if not isinstance(doc, str) or not doc.strip():
            raise TypeError(f"{cls.__name__}.doc must be a non-empty string")
        cls.doc = doc.strip()
        if not isinstance(cls.dependencies, tuple):
            raise TypeError(f"{cls.__name__}.dependencies must be a tuple")
        if not all(
            isinstance(dependency, type) and issubclass(dependency, GenericComponent)
            for dependency in cls.dependencies
        ):
            raise TypeError(
                f"{cls.__name__}.dependencies must contain component classes"
            )

    @classmethod
    def construction_order(
        cls,
        components: list[type[ComponentType]],
        *,
        ignore_missing: bool = False,
    ) -> list[type[ComponentType]]:
        """Return component specifications in dependency order."""
        components_by_name: dict[str, type[ComponentType]] = {}
        for component in components:
            existing = components_by_name.get(component.name)
            if existing is not None and existing is not component:
                raise ValueError(
                    f"Multiple component specifications define {component.name!r}: "
                    f"{existing.__name__} and {component.__name__}"
                )
            components_by_name[component.name] = component

        available = set(components_by_name)
        if not ignore_missing:
            for component in components_by_name.values():
                missing = {
                    dependency.name for dependency in component.dependencies
                }.difference(available)
                if missing:
                    names = ", ".join(sorted(missing))
                    raise ValueError(
                        f"Component {component.name!r} has unknown dependencies: "
                        f"{names}"
                    )

        remaining = list(components_by_name.values())
        completed: set[str] = set()
        ordered: list[type[ComponentType]] = []
        component_kind = cls.__name__.removeprefix("Generic").lower()
        while remaining:
            ready = [
                component
                for component in remaining
                if {
                    dependency.name
                    for dependency in component.dependencies
                    if dependency.name in available
                }.issubset(completed)
            ]
            if not ready:
                cycle = ", ".join(component.name for component in remaining)
                raise ValueError(f"Cyclic {component_kind} dependencies: {cycle}")
            for component in ready:
                remaining.remove(component)
                ordered.append(component)
                completed.add(component.name)
        return ordered
