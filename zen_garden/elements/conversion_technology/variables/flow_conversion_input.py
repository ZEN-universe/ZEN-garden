import numpy as np
import xarray as xr

from zen_garden.topology.generic_variable import GenericVariable


class FlowConversionInput(GenericVariable):
    """Variable for carrier input of conversion technologies."""

    name = "flow_conversion_input"
    indices = [
        "set_conversion_technologies",
        "set_input_carriers",
        "set_nodes",
        "set_time_steps_operation",
    ]
    doc = "Variable for carrier input of conversion technologies"
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        index_values, index_names = index_sets
        sets = model_constructor.zen_model.sets
        index_arrs = sets.tuple_to_arr(index_values, index_names)
        coords = [
            sets.get_coord(data, name)
            for data, name in zip(index_arrs, index_names, strict=False)
        ]
        lower = xr.DataArray(0.0, coords=coords)
        upper = xr.DataArray(np.inf, coords=coords)
        technology_set, carrier_set, node_set, timestep_set = [
            sets[name] for name in index_names
        ]

        for tech in technology_set:
            for carrier in carrier_set[tech]:
                time_step_year = [
                    model_constructor.time_steps.convert_time_step_operation2year(t)
                    for t in timestep_set
                ]
                if carrier == sets["set_reference_carriers"][tech][0]:
                    conversion_factor_lower = conversion_factor_upper = 1
                else:
                    conversion_factor = (
                        model_constructor.zen_model.parameters.conversion_factor.loc[
                            tech, carrier, node_set
                        ]
                    )
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

                capacity = model_constructor.zen_model.variables["capacity"]
                lower.loc[tech, carrier, ...] = (
                    capacity.lower.loc[tech, "power", node_set, time_step_year].data
                    * conversion_factor_lower
                )
                upper.loc[tech, carrier, ...] = (
                    capacity.upper.loc[tech, "power", node_set, time_step_year].data
                    * conversion_factor_upper
                )
        return lower, upper
