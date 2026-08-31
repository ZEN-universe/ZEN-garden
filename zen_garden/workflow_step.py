"""Marks the high-level stages of a ZEN-garden run for the workflow diagram.

Each ``@workflow_step`` decorator records where a stage of the run sequence
lives in the code, without changing the decorated function's behavior.
``docs/_ext/workflow_diagram.py`` reads this registry at documentation-build
time to draw the workflow diagram in the API reference. The diagram is
therefore generated from the code it describes and cannot silently drift
from it the way a hand-drawn image can.

This module is deliberately a top-level, dependency-free module (rather than
living under ``zen_garden.utils``) so that decorating early-loaded modules
such as ``zen_garden.config`` with ``@workflow_step`` cannot introduce an
import cycle.

To add a new top-level stage to the diagram, decorate the function or method
that performs it with ``@workflow_step`` and add its module to
``_STEP_MODULES`` in ``docs/_ext/workflow_diagram.py`` if it is not already
covered.
"""

from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)

_WORKFLOW_STEPS: list["WorkflowStep"] = []


class WorkflowStep:
    """One node in the high-level workflow diagram."""

    def __init__(
        self,
        order: int,
        phase: str,
        label: str,
        qualname: str,
        loop: str | None = None,
    ) -> None:
        """Store the diagram metadata for one workflow step.

        :param order: Position in the overall run sequence. Must be unique.
        :param phase: Name of the diagram section this step belongs to, e.g.
            "Setup" or "Construct & solve".
        :param label: Short, human-readable description shown on the diagram
            node.
        :param qualname: Dotted ``module.Class.method`` (or
            ``module.function``) path of the decorated callable.
        :param loop: Name of the loop this step runs inside, e.g.
            "scenario" or "rolling_horizon". ``None`` if the step runs only
            once per model run.
        """
        self.order = order
        self.phase = phase
        self.label = label
        self.qualname = qualname
        self.loop = loop


def workflow_step(
    order: int, phase: str, label: str, loop: str | None = None
) -> Callable[[F], F]:
    """Register the decorated callable as workflow step ``order``.

    :param order: Position in the overall run sequence. Must be unique.
    :param phase: Name of the diagram section this step belongs to, e.g.
        "Setup" or "Construct & solve".
    :param label: Short, human-readable description shown on the diagram
        node.
    :param loop: Name of the loop this step runs inside, e.g. "scenario" or
        "rolling_horizon". Consecutive steps sharing a loop name are drawn
        inside the same diagram subgraph. Leave as ``None`` if the step runs
        only once per model run.
    """

    def decorator(func: F) -> F:
        qualname = f"{func.__module__}.{func.__qualname__}"
        _WORKFLOW_STEPS.append(
            WorkflowStep(
                order=order, phase=phase, label=label, qualname=qualname, loop=loop
            )
        )
        return func

    return decorator


def get_workflow_steps() -> list[WorkflowStep]:
    """Return the registered workflow steps, sorted by ``order``.

    :raises ValueError: If two steps were registered with the same ``order``.
    """
    steps = sorted(_WORKFLOW_STEPS, key=lambda step: step.order)
    orders = [step.order for step in steps]
    if len(orders) != len(set(orders)):
        raise ValueError(f"Duplicate workflow_step order values: {orders}")
    return steps
