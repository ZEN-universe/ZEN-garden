from __future__ import annotations

import importlib
import inspect
import re
from typing import Any

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import ExtensionMetadata

"""
TO DEBUG:

import pdb;
pdb.set_trace()

The latter line can be placed anywhere in the code and will set a breakpoint
at the location. When calling "make html" to build the documentation, the
breakpoint will be activated. Type variable names to view variables values
"next" to go to the next line, "step" to step into a function, and "exit" to
leave the debugger.
"""


GOOGLE_SECTION_HEADERS = (
    "Args:",
    "Arguments:",
    "Parameters:",
    "Returns:",
    "Yields:",
    "Raises:",
    "Examples:",
    "Attributes:",
)

FIELD_LIST_PREFIXES = (":param", ":return", ":rtype", ":raises")


class DocstringDirective(SphinxDirective):
    """Base directive for rendering selected portions of a docstring.

    Structured docstrings use ``Summary:``, ``Formulation:``, and
    ``Notation:`` headers. The optional ``sections`` argument accepts a
    comma-separated subset of those names, in the order in which they should
    be rendered.
    """

    required_arguments = 1
    number_equations = False
    option_spec = {
        "include-summary": directives.flag,
        "sections": directives.unchanged_required,
    }

    section_header = re.compile(r"^(Summary|Formulation|Notation):$", re.I)
    math_directive = re.compile(r"^(?P<indent>\s*)\.\. math::\s*$")

    @staticmethod
    def make_label_component(value: str) -> str:
        """Convert a document or Python object name into a label component."""
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

    @staticmethod
    def math_directive_has_label(lines: list[str], directive_index: int) -> bool:
        """Return whether a math directive already has an explicit label."""
        directive_indent = len(lines[directive_index]) - len(
            lines[directive_index].lstrip()
        )

        for line in lines[directive_index + 1 :]:
            if not line.strip():
                break

            indentation = len(line) - len(line.lstrip())
            if indentation <= directive_indent:
                break

            stripped = line.strip()
            if stripped.startswith(":label:"):
                return True
            if not stripped.startswith(":"):
                break

        return False

    def number_math_directives(
        self, lines: list[str], full_name: str
    ) -> list[tuple[str, int]]:
        """Add page-qualified labels to otherwise-unlabeled math directives."""
        document = self.make_label_component(self.env.docname)
        python_object = self.make_label_component(full_name)
        numbered_lines: list[tuple[str, int]] = []
        equation_index = 0

        for offset, line in enumerate(lines):
            numbered_lines.append((line, offset))
            match = self.math_directive.match(line)
            if match is None:
                continue

            equation_index += 1
            if self.math_directive_has_label(lines, offset):
                continue

            label = f"docstring-equation-{document}-{python_object}-{equation_index}"
            option_indent = f"{match.group('indent')}   "
            numbered_lines.append((f"{option_indent}:label: {label}", offset))

            next_line_is_content = offset + 1 < len(lines) and bool(
                lines[offset + 1].strip()
            )
            if next_line_is_content:
                numbered_lines.append(("", offset))

        return numbered_lines

    def select_sections(self, lines: list[str]) -> list[str] | None:
        """Select named sections from a structured constraint docstring."""
        sections: dict[str, list[str]] = {}
        current_section: str | None = None

        for line in lines:
            match = self.section_header.match(line.strip())
            if match:
                current_section = match.group(1).lower()
                if current_section in sections:
                    raise ValueError(
                        f"docstring contains duplicate '{current_section}' sections"
                    )
                sections[current_section] = []
            elif current_section is not None:
                sections[current_section].append(line)

        required_sections = {"summary", "formulation", "notation"}
        if not required_sections.issubset(sections):
            return None

        requested_option = self.options.get("sections")
        requested = (
            [part.strip().lower() for part in requested_option.split(",")]
            if requested_option
            else list(sections)
        )
        invalid = [part for part in requested if part not in sections]
        if invalid:
            available = ", ".join(sections)
            raise ValueError(
                f"unknown or unavailable docstring sections {invalid}; "
                f"available sections: {available}"
            )

        selected: list[str] = []
        for part in requested:
            content = sections[part]
            while content and not content[0].strip():
                content = content[1:]
            while content and not content[-1].strip():
                content = content[:-1]
            if selected and content:
                selected.append("")
            selected.extend(content)
        return selected

    def render_docstring(self, obj: Any, full_name: str) -> list[nodes.Node]:
        """Render a cleaned docstring and register its source as a dependency."""
        doc = inspect.getdoc(obj)
        if not doc:
            raise ValueError("the selected object has no docstring")

        lines = doc.splitlines()
        selected = self.select_sections(lines)
        if selected is not None:
            filtered = selected
        else:
            if "sections" in self.options:
                raise ValueError(
                    "the :sections: option requires Summary:, Formulation:, "
                    "and Notation: docstring sections"
                )
            if "include-summary" not in self.options and lines:
                lines.pop(0)
                while lines and not lines[0].strip():
                    lines.pop(0)

            filtered = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(FIELD_LIST_PREFIXES) or any(
                    stripped.startswith(header) for header in GOOGLE_SECTION_HEADERS
                ):
                    break
                filtered.append(line)

        source = inspect.getsourcefile(obj)
        if source is not None:
            self.env.note_dependency(source)
        else:
            source = f"{full_name} docstring"

        try:
            _, source_line = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            source_line = 0

        rendered_lines = (
            self.number_math_directives(filtered, full_name)
            if self.number_equations
            else [(line, offset) for offset, line in enumerate(filtered)]
        )

        content = StringList()
        for line, offset in rendered_lines:
            content.append(line, source, source_line + offset)

        node = nodes.section()
        node.document = self.state.document
        nested_parse_with_titles(self.state, content, node)
        return node.children


class DocstringMethod(DocstringDirective):
    """A directive insert the trimmed docstring of a class method.

    The can be used to insert the docstring of a class method directly into
    the documentation text. It removes the function header, summary line,
    and the information about parameters, arguments, and return types. The
    only text that is inserted into the documentation is therefore the main
    body of the original docstring.

    The directive is useful for minimizing the amount of redundant information
    that is required when writting both docstrings and documentation. For
    example, the mathematical formulation of constraints can now be written
    only once (in the docstring) and copied directly over to the documentation.

    To use this directive, use the following command in a reStructuredText
    file:

    .. docstring_method:: <module.class.method_name>

    Example:

    .. docstring_method::
       zen_garden.elements.conversion_technology.constraints.
       CapacityFactorConversionConstraint.build

    To include only selected structured sections:

    .. docstring_method:: package.module.ConstraintClass.build
       :sections: summary, formulation


    """

    number_equations = True

    def run(self):
        full_name = self.arguments[0]

        try:
            parts = full_name.split(".")
            module_name = ".".join(parts[:-2] if len(parts) > 2 else parts[:-1])
            obj_path = parts[-2:] if len(parts) > 2 else parts[-1:]

            module = importlib.import_module(module_name)
            obj = module
            for attr in obj_path:
                obj = getattr(obj, attr)

            return self.render_docstring(obj, full_name)

        except Exception as e:
            error = self.state_machine.reporter.error(
                f"Failed to extract docstring for '{full_name}': {e}", line=self.lineno
            )
            return [error]


class DocstringClass(DocstringDirective):
    """A directive insert the trimmed docstring of a class.

    The can be used to insert the docstring of a class directly into
    the documentation text. It removes the function header, summary line,
    and the information about parameters, arguments, and return types. The
    only text that is inserted into the documentation is therefore the main
    body of the original docstring.

    The directive is useful for minimizing the amount of redundant information
    that is required when writing both docstrings and documentation. For
    example, the docstring for conversion technologies can now contain all
    required information and pasted directly into the documentation.

    To use this directive, use the following command in a reStructuredText
    file:

    .. docstring_class:: <module.class_name>

    Example:

    .. docstring_class:: zen_garden.elements.carrier.Carrier
    """

    def run(self):
        full_name = self.arguments[0]
        module_name, _, obj_name = full_name.rpartition(".")

        try:
            module = importlib.import_module(module_name)
            obj = getattr(module, obj_name)

            return self.render_docstring(obj, full_name)

        except Exception as e:
            error = self.state_machine.reporter.error(
                f"Failed to extract docstring for '{full_name}': {e}", line=self.lineno
            )
            return [error]


def setup(app: Sphinx) -> ExtensionMetadata:
    """Setup directives.

    This function is required in order to register the directives with
    Sphynx. The name of the directives, as seen by the user, is set here.
    """
    app.add_directive("docstring_method", DocstringMethod)
    app.add_directive("docstring_class", DocstringClass)

    return {
        "version": "0.2",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
