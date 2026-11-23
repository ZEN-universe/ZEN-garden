import numpy as np

from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.technology.masks import CapacityAvailableMask
from zen_garden.elements.technology.parameters.capacity_addition_max import (
    CapacityAdditionMax,
)
from zen_garden.elements.technology.parameters.capacity_existing import CapacityExisting
from zen_garden.elements.technology.parameters.capacity_limit import CapacityLimit
from zen_garden.elements.technology.parameters.lifetime import Lifetime
from zen_garden.elements.technology.parameters.lifetime_existing import LifetimeExisting
from zen_garden.elements.technology.sets.relation_technology_existing_capacity import (
    RelationTechnologyExistingCapacity,
)
from zen_garden.elements.technology.sets.set_existing_capacity_ids import (
    SetExistingCapacityIds,
)
from zen_garden.model.component_types.variable import GenericVariable


class Capacity(GenericVariable):
    """Variable for installed technology capacity."""

    name = "capacity"
    symbol = r"K_{h,p,y}"
    indices = ("set_technologies", "set_location", "set_years")
    doc = "Size of installed technology h at location p and period y."
    unit_category = {"energy_quantity": 1, "time": -1}
    masks = (CapacityAvailableMask,)
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        techs_on_off = model_constructor.optimization_model.sets.on_off_technologies()

        def capacity_bounds(tech, loc, time):
            if tech not in techs_on_off:
                return 0, np.inf

            sets = model_constructor.optimization_model.sets
            params = model_constructor.optimization_model.parameters.dict_parameters
            capacities_existing = 0
            for existing_id in sets.project_relation(
                RelationTechnologyExistingCapacity,
                SetExistingCapacityIds,
                set_technologies=tech,
            ):
                lifetime_existing = params[LifetimeExisting][tech, loc, existing_id]
                if lifetime_existing > params[Lifetime][tech]:
                    is_available = time > lifetime_existing - params[Lifetime][tech]
                else:
                    is_available = time <= lifetime_existing + 1
                if is_available:
                    capacities_existing += params[CapacityExisting][
                        tech, loc, existing_id
                    ]

            addition_max = len(sets[SetYears]) * params[CapacityAdditionMax][tech]
            capacity_limit = params[CapacityLimit][tech, loc, time]
            return 0, min(
                addition_max + capacities_existing,
                capacity_limit + capacities_existing,
            )

        return capacity_bounds
