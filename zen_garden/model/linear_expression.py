"""Utilities for aligning indexed Linopy expressions."""

from collections.abc import Sequence
from typing import Any

import pandas as pd
import xarray as xr
from linopy import Variable
from linopy.constraints import Constraint
from linopy.expressions import LinearExpression


def validate_component_indices(
    name: str, expected: tuple[str, ...], component: Any
) -> None:
    """Validate a model component's public dimensions against its declaration.

    Linopy's private ``_term`` dimension stores the terms of a linear
    expression and is not part of the mathematical index domain.

    Args:
        name: Registered expression or constraint name.
        expected: Declared public indices.
        component: Constructed Linopy or xarray object exposing ``dims``.

    Raises:
        ValueError: If the constructed public dimensions differ from the
            declaration.
    """
    dimensions = (
        component.vars.dims if isinstance(component, Constraint) else component.dims
    )
    actual = tuple(index for index in dimensions if index != "_term")
    if actual != expected:
        raise ValueError(
            f"{name!r} declares indices {expected!r}, but constructs {actual!r}"
        )


def map_and_expand(array, mapping: pd.Series):
    """Select and expand ``array`` using a named pandas mapping."""
    if isinstance(mapping.index, pd.MultiIndex):
        indexer = mapping.to_xarray()
        valid = indexer.notnull()
        safe_indexer = indexer.fillna(0).astype(int)
        return array.sel({mapping.name: safe_indexer}).where(valid)
    mapped = array.sel({mapping.name: mapping.values})
    mapped = mapped.rename({mapping.name: mapping.index.name})
    return mapped.assign_coords({mapping.index.name: mapping.index})


def align_and_mask(expression, mask):
    """Align a Boolean mask to an expression and apply it."""
    if isinstance(expression, xr.DataArray):
        aligner = expression
    elif isinstance(expression, Variable):
        aligner = expression.lower
    else:
        aligner = expression.const
    aligned_mask = xr.align(mask, aligner, join="right")[0]
    return expression.where(aligned_mask)


def transpose_linear_expression(
    expression: LinearExpression, dimensions: Sequence[str]
) -> LinearExpression:
    """Return a linear expression with a deterministic dimension order."""
    data = xr.Dataset(
        {
            name: array.transpose(
                *[dimension for dimension in dimensions if dimension in array.dims],
                *[dimension for dimension in array.dims if dimension not in dimensions],
            )
            for name, array in expression.data.data_vars.items()
        }
    )
    transposed = LinearExpression(data, expression.model)
    # Linopy's constructor broadcasts the Dataset and restores its global
    # dimension order. The individual arrays are already valid and aligned, so
    # retain their requested public order after constructor validation.
    transposed._data = data
    return transposed


def transpose_constraint(
    constraint: Constraint, dimensions: Sequence[str]
) -> Constraint:
    """Return a constraint with a deterministic public dimension order."""
    data = xr.Dataset(
        {
            name: array.transpose(
                *[dimension for dimension in dimensions if dimension in array.dims],
                *[dimension for dimension in array.dims if dimension not in dimensions],
            )
            for name, array in constraint.data.data_vars.items()
        }
    )
    constraint._data = data
    return constraint
