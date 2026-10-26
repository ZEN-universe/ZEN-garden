"""Sphinx directive for tables of declarative model components."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import ExtensionMetadata

from zen_garden.elements import ELEMENT_TYPE_CLASSES, EnergySystem
from zen_garden.model.component_types.index import STRUCTURAL_INDICES, GenericIndex
from zen_garden.model.component_types.parameter import GenericParameter
from zen_garden.model.component_types.set import GenericSet
from zen_garden.model.component_types.variable import GenericVariable

ModelComponent = (
    type[GenericIndex]
    | type[GenericSet]
    | type[GenericParameter]
    | type[GenericVariable]
)


def _registered_components(attribute: str) -> list[ModelComponent]:
    """Return unique registered components for an element attribute."""
    element_classes = [EnergySystem, *ELEMENT_TYPE_CLASSES.values()]
    components: dict[str, ModelComponent] = {}
    for element_class in element_classes:
        for component in getattr(element_class, attribute, ()):
            existing = components.get(component.name)
            if existing is not None and existing is not component:
                raise ValueError(
                    f"Conflicting component classes define {component.name!r}: "
                    f"{existing.__name__} and {component.__name__}"
                )
            components[component.name] = component
    return sorted(components.values(), key=lambda item: (item.symbol, item.name))


def _unit_category(component: ModelComponent) -> str:
    """Format a component's unit dimensions for display."""
    category = getattr(component, "unit_category", None)
    if not category:
        return "dimensionless"
    return ", ".join(
        name if exponent == 1 else f"{name}^{exponent}"
        for name, exponent in sorted(category.items())
    )


def _entry(content: Iterable[nodes.Node | str]) -> nodes.entry:
    """Create one table entry containing a paragraph."""
    entry = nodes.entry()
    paragraph = nodes.paragraph()
    for item in content:
        paragraph += nodes.Text(item) if isinstance(item, str) else item
    entry += paragraph
    return entry


class ModelComponentTable(SphinxDirective):
    """Render registered sets, parameters, or variables as a notation table."""

    required_arguments = 1
    has_content = False

    component_attributes: ClassVar[dict[str, str]] = {
        "sets": "sets",
        "parameters": "parameters",
        "variables": "variables",
    }

    def run(self) -> list[nodes.Node]:
        """Build the requested model-component table."""
        kind = self.arguments[0].strip().lower()
        try:
            attribute = self.component_attributes[kind]
        except KeyError as error:
            valid = ", ".join(self.component_attributes)
            raise self.error(
                f"Unknown model component kind {kind!r}; expected one of {valid}"
            ) from error

        components = _registered_components(attribute)
        headings: tuple[str, ...]
        if kind == "sets":
            registered = {component.name: component for component in components}
            for index in STRUCTURAL_INDICES:
                if index.name in registered:
                    raise ValueError(
                        f"Structural index {index.name!r} conflicts with registered "
                        f"set {registered[index.name].__name__}"
                    )
            components.extend(STRUCTURAL_INDICES)
            components.sort(key=lambda item: (item.symbol, item.name))
        if kind == "sets":
            headings = ("Name", "Symbol", "Indexed by", "Description")
        else:
            headings = ("Name", "Symbol", "Indices", "Description", "Unit category")

        table = nodes.table(classes=["model-component-table"])
        table += nodes.title(text=kind.capitalize())
        table_group = nodes.tgroup(cols=len(headings))
        table += table_group
        for _ in headings:
            table_group += nodes.colspec(colwidth=1)

        table_head = nodes.thead()
        header_row = nodes.row()
        for heading in headings:
            header_row += _entry([heading])
        table_head += header_row
        table_group += table_head

        table_body = nodes.tbody()
        for component in components:
            row = nodes.row()
            row += _entry([nodes.literal(text=component.name)])
            row += _entry([nodes.math(text=component.symbol)])
            if kind == "sets":
                index_set = getattr(component, "index_set", None)
                index_set_name = (
                    index_set
                    if isinstance(index_set, str)
                    else getattr(index_set, "name", None)
                )
                row += _entry(
                    [nodes.literal(text=index_set_name)] if index_set_name else ["—"]
                )
            else:
                indices = getattr(component, "indices", ())
                row += _entry(
                    [nodes.literal(text=", ".join(indices))] if indices else ["scalar"]
                )
            row += _entry([component.doc])
            if kind != "sets":
                row += _entry([_unit_category(component)])
            table_body += row
        table_group += table_body
        return [table]


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the model-component table directive."""
    app.add_directive("model-component-table", ModelComponentTable)
    return {"version": "1.0", "parallel_read_safe": True}
