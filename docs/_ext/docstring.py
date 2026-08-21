from __future__ import annotations

import importlib
import inspect
import re

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import ViewList
from sphinx.application import Sphinx
from sphinx.util.docstrings import prepare_docstring
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import ExtensionMetadata

"""
TO DUBUG:

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
    "Notation:",
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
    option_spec = {
        "include-summary": directives.flag,
        "sections": directives.unchanged_required,
    }

    section_header = re.compile(r"^(Summary|Formulation|Notation):$", re.I)

    def preprocess_math_content(self,lines: list[str]) -> list[str]:
        """Fix double backslashes and join wrapped math lines before reST parsing."""
        processed: list[str] = []
        in_math_block = False
        math_indent = 0
        current_math_lines: list[str] = []

        for line in lines:
            # 1. Un-escape double backslashes (\\command -> \command)
            line = line.replace("\\\\", "\\")

            # 2. Track .. math:: directive blocks
            stripped = line.strip()
            current_indent = len(line) - len(line.lstrip())

            if stripped.startswith(".. math::"):
                in_math_block = True
                math_indent = current_indent
                processed.append(line)
                continue

            if in_math_block:
                # Empty lines or non-indented lines mark the end of the .. math:: block
                if stripped and current_indent <= math_indent:
                    # Flush collected math lines into a single continuous equation string
                    if current_math_lines:
                        combined_math = " ".join(
                            line.strip() for line in current_math_lines
                        )
                        # Indent the combined block relative to the directive
                        indent_str = " " * (math_indent + 4)
                        processed.append(f"{indent_str}{combined_math}")
                        current_math_lines = []
                    in_math_block = False
                else:
                    if stripped:
                        current_math_lines.append(stripped)
                    continue

            # 3. Fix inline :math: roles split across lines
            processed.append(line)

        # Flush if docstring ends with a math block
        if in_math_block and current_math_lines:
            combined_math = " ".join(line.strip() for line in current_math_lines)
            indent_str = " " * (math_indent + 4)
            processed.append(f"{indent_str}{combined_math}")

        # 4. Join multi-line :math:`...` roles back onto single lines
        full_text = "\n".join(processed)
        full_text = re.sub(
            r"(:math:`[^`]+`)",
            lambda m: m.group(1).replace("\n", " "),
            full_text,
            flags=re.DOTALL,
        )

        return full_text.splitlines()

    def select_sections(self, lines: list[str]) -> list[str] | None:
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

        selected: list[str] = []
        for part in requested:
            content = sections[part]
            if not content:
                continue

            # Dedent content so nested parser receives base column 0
            dedented_block = textwrap.dedent("\n".join(content)).splitlines()

            while dedented_block and not dedented_block[0].strip():
                dedented_block.pop(0)
            while dedented_block and not dedented_block[-1].strip():
                dedented_block.pop()

            if selected and dedented_block:
                selected.append("")
            selected.extend(dedented_block)

        return selected

    def render_docstring(self, obj: object, full_name: str) -> list[nodes.Node]:
        raw_doc = getattr(obj, "__doc__", "") or ""
        if not raw_doc.strip():
            raise ValueError("the selected object has no docstring")

        lines = prepare_docstring(raw_doc)

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

        # Preprocess lines to fix math line-wrapping and double backslashes automatically
        filtered = self.preprocess_math_content(filtered)

        source = inspect.getsourcefile(obj)
        if source is not None:
            self.env.note_dependency(source)
        else:
            source = f"{full_name} docstring"

        try:
            _, source_line = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            source_line = 0

        content = ViewList()
        for offset, line in enumerate(filtered):
            content.append(line, source, source_line + offset)

        container = nodes.container()
        container.document = self.state.document
        self.state.nested_parse(content, 0, container)
        return container.children

    # def render_docstring(self, obj: object, full_name: str) -> list[nodes.Node]:
    #     """Render a cleaned docstring and register its source as a dependency."""
    #     doc = inspect.getdoc(obj)
    #     if not doc:
    #         raise ValueError("the selected object has no docstring")

    #     lines = doc.splitlines()
    #     selected = self.select_sections(lines)
    #     if selected is not None:
    #         filtered = selected
    #     else:
    #         if "sections" in self.options:
    #             raise ValueError(
    #                 "the :sections: option requires Summary:, Formulation:, "
    #                 "and Notation: docstring sections"
    #             )
    #         if "include-summary" not in self.options and lines:
    #             lines.pop(0)
    #             while lines and not lines[0].strip():
    #                 lines.pop(0)

    #         filtered = []
    #         for line in lines:
    #             stripped = line.strip()
    #             if stripped.startswith(FIELD_LIST_PREFIXES) or any(
    #                 stripped.startswith(header) for header in GOOGLE_SECTION_HEADERS
    #             ):
    #                 break
    #             filtered.append(line)

    #     source = inspect.getsourcefile(obj)
    #     if source is not None:
    #         self.env.note_dependency(source)
    #     else:
    #         source = f"{full_name} docstring"

    #     try:
    #         _, source_line = inspect.getsourcelines(obj)
    #     except (OSError, TypeError):
    #         source_line = 0

    #     content = ViewList()
    #     for offset, line in enumerate(filtered):
    #         content.append(line, source, source_line + offset)

    #     node = nodes.section()
    #     node.document = self.state.document
    #     nested_parse_with_titles(self.state, content, node)
    #     return node.children


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
       zen_garden.constraints.conversion_technology.
       CapacityFactorConversionConstraint.build

    To include only selected structured sections:

    .. docstring_method:: package.module.ConstraintClass.build
       :sections: summary, formulation


    """

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
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
