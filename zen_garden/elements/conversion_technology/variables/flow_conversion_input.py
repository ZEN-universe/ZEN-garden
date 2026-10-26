import numpy as np
import xarray as xr

from zen_garden.elements.conversion_technology.parameters.conversion_factor import (
    ConversionFactor,
)
from zen_garden.elements.technology.masks import PowerOperationMask
from zen_garden.elements.technology.variables.capacity import Capacity
from zen_garden.model.component_types.variable import GenericVariable


class FlowConversionInput(GenericVariable):
    """Variable for carrier input of conversion technologies."""

    name = "flow_conversion_input"
    symbol = r"F^{\mathrm{conv,in}}_{h,c^{\mathrm{in}},n,t}"
    indices = [
        "set_conversion_technologies",
        "set_input_carriers",
        "set_nodes",
        "set_years",
        "set_representative_hours",
    ]
    doc = "Carrier input of conversion technologies."
    unit_category = {"energy_quantity": 1, "time": -1}
    masks = [PowerOperationMask]
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        optimization_model = model_constructor.optimization_model
        index_values, index_names = index_sets
        sets = optimization_model.sets
        index_arrs = sets.tuple_to_arr(index_values, index_names)
        coords = {
            name: sets.get_coord(data, name).data
            for data, name in zip(index_arrs, index_names, strict=False)
        }
        lower = xr.DataArray(0.0, coords=coords, dims=index_names)
        upper = xr.DataArray(np.inf, coords=coords, dims=index_names)
        technology_set, _, node_set, year_set, _ = [sets[name] for name in index_names]
        carrier_dimension = index_names[1]

        for tech in technology_set:
            carriers = sets.get_related(
                "set_conversion_technologies", tech, carrier_dimension
            )
            reference_carrier = sets.get_related(
                "set_technologies", tech, "set_reference_carriers"
            )[0]
            for carrier in carriers:
                if carrier == reference_carrier:
                    conversion_factor_lower = conversion_factor_upper = 1
                else:
                    conversion_factor = optimization_model.parameters[
                        ConversionFactor
                    ].loc[tech, carrier, node_set]
                    conversion_factor_lower = conversion_factor.min().data
                    conversion_factor_upper = conversion_factor.max().data
                    if 0 in conversion_factor_upper:
                        rounding = (
                            model_constructor.config.solver.rounding_decimal_points_tsa
                        )
                        raise ValueError(
                            f"Maximum conversion factor of {tech} for carrier "
                            f"{carrier} is 0.\n Potentially, the conversion factor "
                            f"is too small (1e-{rounding}), so that it is rounded "
                            "to 0 after the time series aggregation."
                        )

                capacity = optimization_model.variables[Capacity]
                lower.loc[tech, carrier, ...] = (
                    capacity.lower.loc[tech, node_set, year_set].data[..., None]
                    * conversion_factor_lower
                )
                upper.loc[tech, carrier, ...] = (
                    capacity.upper.loc[tech, node_set, year_set].data[..., None]
                    * conversion_factor_upper
                )
        return lower, upper
