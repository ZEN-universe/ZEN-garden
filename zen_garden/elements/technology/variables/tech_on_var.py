import xarray as xr

from zen_garden.model.component_types.variable import GenericVariable


class TechOnVar(GenericVariable):
    """Variable for technology on/off binary."""

    name = "tech_on_var"
    indices = ["set_technologies", "set_location", "set_time_steps_operation"]
    doc = (
        "Binary variable indicating when technology is switched on at location l and "
        "time t"
    )
    unit_category = {}
    binary = True

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        return None

    @classmethod
    def get_mask(cls, model_constructor, index_sets):
        techs_on_off, index_names = model_constructor.create_custom_set(
            [
                "set_technologies",
                "set_on_off",
                "set_location",
                "set_time_steps_operation",
            ]
        )
        index_names.pop(1)
        mask = model_constructor.optimization_model.sets.indices_to_mask(
            techs_on_off, index_names, (0, 0)
        )[0]
        times = model_constructor.optimization_model.sets["set_time_steps_operation"]
        time_step_year = xr.DataArray(
            [
                model_constructor.time_steps.convert_time_step_operation2year(time)
                for time in times.data
            ],
            coords=[times],
            dims=["set_time_steps_operation"],
        )
        nonzero_capacity = (
            model_constructor.optimization_model.parameters.capacity_limit.sel(
                {"set_capacity_types": "power", "set_years": time_step_year}
            )
            != 0
        )
        return mask & nonzero_capacity.drop_vars("set_capacity_types")
