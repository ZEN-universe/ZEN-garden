"""Generates the high-level workflow diagram shown in the API reference.

The diagram is built from the ``@workflow_step`` markers on the functions
that actually run a ZEN-garden optimization (see
:mod:`zen_garden.workflow_step`), so it is regenerated from the code at
every documentation build and cannot silently drift from it the way a
hand-drawn image can.

To add a new top-level stage to the diagram, decorate the function or
method that performs it with ``@workflow_step`` (see that module for the
parameters) and add its module to ``_STEP_MODULES`` below if it is not
already imported by one of the other listed modules.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata

    from zen_garden.workflow_step import WorkflowStep

# Modules that carry at least one `@workflow_step` marker. Listed explicitly
# (rather than just `import zen_garden`) so that adding a step in a new
# module is a visible, one-line addition here.
_STEP_MODULES = [
    "zen_garden.config",
    "zen_garden.model.schema",
    "zen_garden.plugin_system.loader",
    "zen_garden.input.scenario_utils",
    "zen_garden.model.element_factory",
    "zen_garden.input.data_loading_service",
    "zen_garden.workflow.optimization_workflow",
    "zen_garden.workflow.optimization_step",
]

# A loop's label, and, if the loop is nested inside another loop, its parent.
_LOOP_LABELS = {
    "scenario": "for each scenario",
    "rolling_horizon": (
        "for each rolling-horizon step<br/>"
        "(more than one only under myopic foresight)"
    ),
}
_LOOP_PARENT = {
    "rolling_horizon": "scenario",
}

_OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "files"
    / "references"
    / "_generated"
    / "workflow_diagram.rst"
)


def _loop_path(loop: str | None) -> list[str]:
    """Return the loop nesting from outermost to innermost for ``loop``."""
    path: list[str] = []
    while loop is not None:
        path.insert(0, loop)
        loop = _LOOP_PARENT.get(loop)
    return path


def _short_qualname(qualname: str) -> str:
    """Drop the module prefix, keeping ``Class.method`` or a bare function name."""
    parts = qualname.split(".")
    for i, part in enumerate(parts):
        if part[:1].isupper():
            return ".".join(parts[i:])
    return parts[-1]


def _build_mermaid_source(steps: list["WorkflowStep"]) -> str:
    """Render the workflow steps as a Mermaid ``flowchart`` definition.

    Consecutive steps that share a ``loop`` are nested inside a Mermaid
    ``subgraph`` for that loop (nested per ``_LOOP_PARENT``), with a dashed
    "repeat" edge from the loop's last step back to its first.
    """
    node_ids = [f"n{i}" for i in range(1, len(steps) + 1)]
    paths = [_loop_path(step.loop) for step in steps]

    loop_bounds: dict[str, tuple[int, int]] = {}
    for i, path in enumerate(paths):
        for loop_name in path:
            first_i, last_i = loop_bounds.get(loop_name, (i, i))
            loop_bounds[loop_name] = (min(first_i, i), max(last_i, i))

    lines = ["flowchart TD"]
    indent = "    "
    open_stack: list[str] = []

    def close_to(depth: int) -> None:
        while len(open_stack) > depth:
            loop_name = open_stack.pop()
            level = len(open_stack)
            lines.append(f"{indent * (level + 1)}end")
            first_i, last_i = loop_bounds[loop_name]
            if first_i != last_i:
                lines.append(
                    f"{indent * (level + 1)}{node_ids[last_i]} "
                    f"-.->|repeat| {node_ids[first_i]}"
                )

    prev_node_id: str | None = None
    for i, (step, path) in enumerate(zip(steps, paths, strict=True)):
        common = 0
        for a, b in zip(open_stack, path, strict=False):
            if a != b:
                break
            common += 1
        close_to(common)
        for level in range(common, len(path)):
            loop_name = path[level]
            lines.append(
                f"{indent * (level + 1)}subgraph {loop_name}_loop"
                f'["{_LOOP_LABELS[loop_name]}"]'
            )
            open_stack.append(loop_name)

        level = len(open_stack)
        node_id = node_ids[i]
        label = step.label.replace('"', "&quot;")
        short = _short_qualname(step.qualname)
        lines.append(
            f'{indent * (level + 1)}{node_id}["{label}<br/><small><i>{short}'
            f'</i></small>"]'
        )
        if prev_node_id is not None:
            lines.append(f"{indent * (level + 1)}{prev_node_id} --> {node_id}")
        prev_node_id = node_id

    close_to(0)
    return "\n".join(lines)


def generate_workflow_diagram(app: "Sphinx") -> None:
    """Regenerate the workflow-diagram include file before the build reads it."""
    for module_name in _STEP_MODULES:
        importlib.import_module(module_name)

    from zen_garden.workflow_step import get_workflow_steps

    steps = get_workflow_steps()
    mermaid_source = _build_mermaid_source(steps)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Indented by 6: 3 for the `container` directive's content, 3 more for the
    # nested `mermaid` directive's content.
    indented = "\n".join(
        f"      {line}" if line else "" for line in mermaid_source.splitlines()
    )
    _OUTPUT_PATH.write_text(
        ".. This file is generated by docs/_ext/workflow_diagram.py. Do not edit.\n"
        "   Add or move a `@workflow_step` marker in the code instead.\n\n"
        ".. container:: workflow-diagram-large\n\n"
        "   .. mermaid::\n"
        "      :zoom:\n\n"
        f"{indented}\n"
    )


def setup(app: "Sphinx") -> "ExtensionMetadata":
    """Register the generator to run once, before Sphinx reads any source file."""
    app.connect("builder-inited", generate_workflow_diagram)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
