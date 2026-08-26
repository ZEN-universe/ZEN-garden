import numpy as np

# TODO: This could be vectorized
def capacity_bounds(tech, capacity_type, loc, time):
    """Return bounds of capacity for bigM expression.

    :param tech: tech index
    :param capacity_type: either power or energy
    :param loc: location of capacity
    :param time: investment time step
    :return: bounds: bounds of capacity
    """
    # bounds only needed for Big-M formulation,
    #   thus if any technology is modeled with on-off behavior
    if tech in techs_on_off:
        params = self.zen_model.parameters.dict_parameters
        capacity_existing = params.capacity_existing
        capacity_addition_max = params.capacity_addition_max
        capacity_limit = params.capacity_limit
        capacities_existing = 0
        for id_technology_existing in self.zen_model.sets[
            "set_technologies_existing"
        ][tech]:
            if (
                params.lifetime_existing[tech, loc, id_technology_existing]
                > params.lifetime[tech]
            ):
                if (
                    time
                    > params.lifetime_existing[
                        tech, loc, id_technology_existing
                    ]
                    - params.lifetime[tech]
                ):
                    capacities_existing += capacity_existing[
                        tech, capacity_type, loc, id_technology_existing
                    ]
            elif (
                time
                <= params.lifetime_existing[tech, loc, id_technology_existing]
                + 1
            ):
                capacities_existing += capacity_existing[
                    tech, capacity_type, loc, id_technology_existing
                ]

        capacity_addition_max = (
            len(self.zen_model.sets["set_years"])
            * capacity_addition_max[tech, capacity_type]
        )
        max_capacity_limit = capacity_limit[tech, capacity_type, loc, time]
        bound_capacity = min(
            capacity_addition_max + capacities_existing,
            max_capacity_limit + capacities_existing,
        )
        return 0, bound_capacity
    else:
        return 0, np.inf