import logging
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import pandas as pd
import xarray as xr

from zen_garden.elements.technology import Technology
from zen_garden.model.components.component import Component
from zen_garden.model.components.zen_set import ZenSet

if TYPE_CHECKING:
    from linopy import Model as LinopyModel

    from zen_garden.model.components.index_set import IndexSet
    from zen_garden.model.config import Config
    from zen_garden.optimization_setup import UnitHandling
    from zen_garden.services.element_registry import ElementRegistry

logger = logging.getLogger(__name__)


class Variable(Component):
    def __init__(
        self,
        unit_handling: "UnitHandling",
        sets: "IndexSet",
        lp_model: "LinopyModel",
        config: "Config",
        element_registry: "ElementRegistry",
    ):
        """Initialization of a variable.

        :param optimization_setup: OptimizationSetup object
        """
        super().__init__()

        self.unit_handling = unit_handling
        self.sets = sets
        self.lp_model = lp_model
        self.config = config
        self.element_registry = element_registry

        self.units = {}

    def add_variable(
        self,
        name: str,
        # TODO: Find smaller set of types for index_sets
        index_sets: (
            ZenSet | tuple[list | pd.Series, list[str]] | list[list] | xr.DataArray
        ),
        unit_category,
        integer: bool = False,
        binary: bool = False,
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
    ):
        """Initialization of a variable.

        :param model: parent block component of variable, must be linopy model
        :param name: name of variable
        :param index_sets: Tuple of index values and index names
        :param unit_category: dict defining the dimensionality of the variable's unit
        :param integer: If it is an integer variable
        :param binary: If it is a binary variable
        :param bounds:  bounds of variable
        :param doc: docstring of variable
        :param mask: mask of variable
        """
        if name in self.docs.keys():
            logger.warning(f"Variable {name} already added. Can only be added once")
            return

        index_values, index_list = self.get_index_names_data(index_sets)
        mask_index, lower, upper = self.sets.indices_to_mask(
            index_values, index_list, bounds, self.lp_model
        )
        if mask is not None:
            mask = mask.reindex_like(mask_index, fill_value=cast(Any, False))
            mask_index = mask_index & mask
        self.lp_model.add_variables(
            lower=lower,
            upper=upper,
            integer=integer,
            binary=binary,
            name=name,
            mask=mask_index,
            coords=mask_index.coords,
        )

        # save variable doc
        if integer:
            domain = "Integers"
        elif binary:
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
        self.units[name] = self.get_var_units(
            unit_category, index_values, index_list, mask_index
        )

    def get_var_units(self, unit_category, var_index_values, index_list, mask):
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
        unit = self.unit_handling.ureg("dimensionless")
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
                for key, value in (self.unit_handling.base_units.items())
                if value == dim_name
            ][0]
            unit = unit * self.unit_handling.ureg(dim_unit) ** unit_category[dim]
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
            energy_quantities = self.unit_handling.carrier_energy_quantities
            for (
                carrier,
                energy_quantity,
            ) in energy_quantities.items():
                carrier_idx = var_units.index.get_level_values(carrier_level) == carrier
                var_units[carrier_idx] = str(
                    (unit * energy_quantity ** unit_category["energy_quantity"]).units
                )
        # energy_quantity depends on technology index level (e.g. capacity)
        else:
            tech_level = [
                str(level)
                for level in var_units.index.names
                if level and "technologies" in str(level)
            ][0]
            for technology in self.element_registry.all_elements_of_type(Technology):
                reference_carrier = technology.reference_carrier[0]
                energy_quantities = self.unit_handling.carrier_energy_quantities
                energy_quantity = [
                    energy_quantity
                    for carrier, energy_quantity in energy_quantities.items()
                    if carrier == reference_carrier
                ][0]
                tech_idx = (
                    var_units.index.get_level_values(tech_level) == technology.name
                )
                var_units[tech_idx] = str(
                    (unit * energy_quantity ** unit_category["energy_quantity"]).units
                )
            if "set_capacity_types" in var_units.index.names:
                energy_idx = (
                    var_units.index.get_level_values("set_capacity_types") == "energy"
                )
                var_units[energy_idx] = var_units[energy_idx].apply(
                    lambda u: str(self.unit_handling.ureg(u + "*hour").units)
                )

        return var_units[mask.to_series()]
