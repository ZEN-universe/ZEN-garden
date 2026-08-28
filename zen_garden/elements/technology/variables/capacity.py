import numpy as np

from zen_garden.model.component_types.variable import GenericVariable


class Capacity(GenericVariable):
    """Variable for installed technology capacity."""

    name = "capacity"
    indices = ["set_technologies", "set_capacity_types", "set_location", "set_years"]
    doc = "Variable for size of installed technology at location l and time t"
    unit_category = {"energy_quantity": 1, "time": -1}

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        techs_on_off = model_constructor.create_custom_set(
            ["set_technologies", "set_on_off"]
        )[0]

        def capacity_bounds(tech, capacity_type, loc, time):
            if tech not in techs_on_off:
                return 0, np.inf

            sets = model_constructor.optimization_model.sets
            params = model_constructor.optimization_model.parameters.dict_parameters
            capacities_existing = 0
            for existing_id in sets["set_technologies_existing"][tech]:
                lifetime_existing = params.lifetime_existing[tech, loc, existing_id]
                if lifetime_existing > params.lifetime[tech]:
                    is_available = time > lifetime_existing - params.lifetime[tech]
                else:
                    is_available = time <= lifetime_existing + 1
                if is_available:
                    capacities_existing += params.capacity_existing[
                        tech, capacity_type, loc, existing_id
                    ]

            addition_max = (
                len(sets["set_years"])
                * params.capacity_addition_max[tech, capacity_type]
            )
            capacity_limit = params.capacity_limit[tech, capacity_type, loc, time]
            return 0, min(
                addition_max + capacities_existing,
                capacity_limit + capacities_existing,
            )

        return capacity_bounds
