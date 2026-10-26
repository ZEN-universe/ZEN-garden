"""The VariableRegistry, which collects the model's decision variables."""

import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.model.component_types.variable import GenericVariable, VariableType
from zen_garden.model.registries.base import Registry, component_name
from zen_garden.model.registries.set import DomainSet

if TYPE_CHECKING:
    from linopy import Model as LinopyModel

    from zen_garden.input.unit_converter import UnitConverter
    from zen_garden.model.element_registry import ElementRegistry
    from zen_garden.model.registries.set_registry import SetRegistry
    from zen_garden.model.schema import ModelSchema

logger = logging.getLogger(__name__)


def align_mask(mask: xr.DataArray, target: xr.DataArray) -> xr.DataArray:
    """Align a canonical mask to backend dimensions, including empty renames."""
    if tuple(mask.dims) != tuple(target.dims):
        if mask.ndim != target.ndim or mask.shape != target.shape:
            raise ValueError(
                f"Cannot align mask dimensions {tuple(mask.dims)!r} with backend "
                f"dimensions {tuple(target.dims)!r}"
            )
        invalid_renames = [
            (source, destination)
            for source, destination in zip(mask.dims, target.dims, strict=True)
            if source != destination
            and mask.sizes[source] != 0
            and target.sizes[destination] != 0
        ]
        if invalid_renames:
            raise ValueError(
                "Positional mask-dimension renaming is allowed only for empty "
                f"backend dimensions, not {invalid_renames!r}"
            )
        rename = {
            source: destination
            for source, destination in zip(mask.dims, target.dims, strict=True)
            if source != destination
        }
        mask = mask.rename(rename)
    return mask.reindex_like(target, fill_value=False).astype(bool)


class VariableRegistry(Registry):
    def __init__(
        self,
        unit_converter: "UnitConverter",
        sets: "SetRegistry",
        lp_model: "LinopyModel",
        model_schema: "ModelSchema",
        element_registry: "ElementRegistry",
    ):
        """Initialization of a variable.

        :param unit_converter: UnitConverter object
        :param sets: SetRegistry object
        :param lp_model: LinopyModel object
        :param model_schema: global model schema
        :param element_registry: ElementRegistry object
        """
        super().__init__()

        self.unit_converter = unit_converter
        self.sets = sets
        self.lp_model = lp_model
        self.model_schema = model_schema
        self.element_registry = element_registry

        self.units: dict[str, pd.Series | None] = {}
        self.structural_masks: dict[str, xr.DataArray] = {}
        self.performance_masks: dict[str, xr.DataArray] = {}
        self.mask_default_values: dict[str, float] = {}
        self.applied_relations: dict[str, tuple[str, ...]] = {}
        self.index_diagnostics: dict[str, dict[str, Any]] = {}

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    def add_variable(
        self,
        name: str,
        # TODO: Find smaller set of types for index_sets
        index_sets: (
            DomainSet | tuple[list | pd.Series, list[str]] | list[list] | xr.DataArray
        ),
        unit_category,
        variable_type: VariableType = "continuous",
        # TODO: Create Type alias for bounds
        bounds: (
            tuple[xr.DataArray, xr.DataArray]
            | tuple[float, float]
            | tuple[int, int]
            | np.ndarray
            | Callable
            | None
        ) = None,
        doc: str = "",
        mask: xr.DataArray | None = None,
        structural_mask: xr.DataArray | None = None,
        mask_default_value: float | None = None,
        applied_relations: tuple[str, ...] = (),
    ):
        """Initialization of a variable.

        :param model: parent block component of variable, must be linopy model
        :param name: name of variable
        :param index_sets: Tuple of index values and index names
        :param unit_category: dict defining the dimensionality of the variable's unit
        :param variable_type: Domain of the variable
        :param bounds:  bounds of variable
        :param doc: docstring of variable
        :param mask: mask of variable
        """
        if name in self.docs.keys():
            raise ValueError(f"Variable {name!r} already added. Can only be added once")
        if variable_type not in ("continuous", "binary", "integer"):
            raise ValueError(f"Unknown variable type {variable_type!r}")
        if mask_default_value not in (None, 0.0):
            raise ValueError("Performance-mask defaults currently support only 0.0")
        if (mask is None) != (mask_default_value is None):
            raise ValueError(
                "A performance mask and mask_default_value must be supplied together"
            )

        index_values, index_list = self.get_index_names_data(index_sets)
        mask_index, lower, upper = self.sets.indices_to_mask(
            index_values, index_list, bounds, self.lp_model
        )
        if mask_index.size == 0:
            lower, upper = -np.inf, np.inf
        canonical_structural = (
            structural_mask.copy() if structural_mask is not None else mask_index.copy()
        )
        solver_structural = align_mask(canonical_structural, mask_index)
        performance_mask = xr.full_like(canonical_structural, True, dtype=bool)
        if mask is not None:
            performance_mask = mask.astype(bool)
            solver_performance = align_mask(performance_mask, mask_index)
            mask_index = solver_structural & solver_performance
        else:
            mask_index = solver_structural
        try:
            self.lp_model.add_variables(
                lower=lower,
                upper=upper,
                integer=variable_type == "integer",
                binary=variable_type == "binary",
                name=name,
                mask=mask_index,
                coords=mask_index.coords,
            )
        except ValueError as error:
            raise ValueError(f"Failed to register variable {name!r}") from error

        # save variable doc
        if variable_type == "integer":
            domain = "Integers"
        elif variable_type == "binary":
            domain = "Binary"
        elif isinstance(bounds, tuple) and isinstance(bounds[0], xr.DataArray):
            domain = "BoundedReals"
        elif isinstance(bounds, tuple) and bounds[0] == 0:
            domain = "NonNegativeReals"
        elif callable(bounds) or isinstance(bounds, np.ndarray):
            domain = "BoundedReals"
        else:
            domain = "Reals"

        self.docs[name] = self.compile_doc_string(doc, index_list, name, domain)
        self.units[name] = self._get_var_units(
            unit_category, index_values, index_list, mask_index
        )
        self.structural_masks[name] = canonical_structural
        self.performance_masks[name] = performance_mask
        if mask_default_value is not None:
            self.mask_default_values[name] = mask_default_value
        self.applied_relations[name] = applied_relations
        self._record_index_diagnostics(
            name,
            canonical_structural,
            performance_mask,
            applied_relations,
        )

    def _record_index_diagnostics(
        self,
        name: str,
        structural_mask: xr.DataArray,
        performance_mask: xr.DataArray,
        applied_relations: tuple[str, ...],
    ) -> None:
        """Record structural/performance density diagnostics for a variable."""
        dense_cells = structural_mask.size
        structurally_valid = int(structural_mask.sum().item())
        performance_active = int(performance_mask.sum().item())
        solver_entries = int((structural_mask & performance_mask).sum().item())
        if solver_entries:
            dense_active_ratio = dense_cells / solver_entries
        elif dense_cells:
            dense_active_ratio = float("inf")
        else:
            dense_active_ratio = 1.0
        self.index_diagnostics[name] = {
            "requested_dimensions": tuple(structural_mask.dims),
            "applied_relations": applied_relations,
            "dense_cells": dense_cells,
            "structurally_valid": structurally_valid,
            "performance_active": performance_active,
            "solver_entries": solver_entries,
            "dense_active_ratio": dense_active_ratio,
        }
        if dense_active_ratio >= 5:
            logger.warning(
                "Variable %s has dense/active index ratio %.3g (%d/%d)",
                name,
                dense_active_ratio,
                dense_cells,
                solver_entries,
            )

    def _get_var_units(self, unit_category, var_index_values, index_list, mask):
        """Creates series of units with identical multi-index as variable has.

        :param unit_category: dict defining the dimensionality of the variable's unit
        :param var_index_values: list of variable index values
        :param index_list: list of index names
        :param mask: mask of variable
        :return: series of variable units
        """
        # if not check_unit_consistency
        if not self.config.solver.check_unit_consistency:
            return None
        # binary variables
        if not unit_category:
            return None
        if all(isinstance(item, tuple) for item in var_index_values):
            index = pd.MultiIndex.from_tuples(var_index_values, names=index_list)
        else:
            index = pd.Index(var_index_values)
        unit = self.unit_converter.ureg("dimensionless")
        distinct_dims = {
            "money": "[currency]",
            "distance": "[length]",
            "time": "[time]",
            "emissions": "[mass]",
        }
        for dim, dim_name in distinct_dims.items():
            if dim not in unit_category:
                continue
            dim_unit = [
                key
                for key, value in (self.unit_converter.base_units.items())
                if value == dim_name
            ][0]
            unit = unit * self.unit_converter.ureg(dim_unit) ** unit_category[dim]
        var_units = pd.Series(index=index, dtype=str)

        if "energy_quantity" not in unit_category:
            # variable has constant unit
            var_units[:] = str(unit.units)
            return var_units[mask.to_series()]

        # variable can have different units
        # energy_quantity depends on carrier index level (e.g. flow_import)
        if any(
            carrier_name is not None and "carrier" in str(carrier_name)
            for carrier_name in var_units.index.names
        ):
            carrier_level = [
                str(level)
                for level in var_units.index.names
                if level and "carrier" in str(level)
            ][0]
            energy_quantities = self.unit_converter.carrier_energy_quantities
            # map the units per carrier at once instead of masking per carrier
            unit_by_carrier = {
                carrier: str(
                    (unit * energy_quantity ** unit_category["energy_quantity"]).units
                )
                for carrier, energy_quantity in energy_quantities.items()
            }
            var_units[:] = np.asarray(
                var_units.index.get_level_values(carrier_level).map(unit_by_carrier),
                dtype=object,
            )
        # energy_quantity depends on technology index level (e.g. capacity)
        else:
            tech_level = [
                str(level)
                for level in var_units.index.names
                if level and "technologies" in str(level)
            ][0]
            energy_quantities = self.unit_converter.carrier_energy_quantities
            # map the units per technology at once instead of masking per technology
            unit_by_tech = {}
            for technology in self.element_registry.all_elements_by_type_name(
                "Technology"
            ):
                energy_quantity = energy_quantities[technology.reference_carrier[0]]
                unit_by_tech[technology.name] = str(
                    (unit * energy_quantity ** unit_category["energy_quantity"]).units
                )
            var_units[:] = np.asarray(
                var_units.index.get_level_values(tech_level).map(unit_by_tech),
                dtype=object,
            )
        return var_units[mask.to_series()]

    def __getitem__(self, key: str | type[GenericVariable]):
        """Get variable values by name or declarative variable class.

        :param key: name of variable
        :return: variable values
        """
        return self.lp_model.variables[component_name(key, GenericVariable)]
