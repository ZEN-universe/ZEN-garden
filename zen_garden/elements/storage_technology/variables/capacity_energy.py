import numpy as np

from zen_garden.elements.energy_system.sets.set_technologies import SetTechnologies
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.storage_technology.masks.energy_capacity_available_mask import (  # noqa: E501
    EnergyCapacityAvailableMask,
)
from zen_garden.elements.storage_technology.parameters.capacity_addition_max_energy import (  # noqa: E501
    CapacityAdditionMaxEnergy,
)
from zen_garden.elements.storage_technology.parameters.capacity_existing_energy import (
    CapacityExistingEnergy,
)
from zen_garden.elements.storage_technology.parameters.capacity_limit_energy import (
    CapacityLimitEnergy,
)
from zen_garden.elements.technology.parameters.lifetime import Lifetime
from zen_garden.elements.technology.parameters.lifetime_existing import LifetimeExisting
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.elements.technology.sets.set_technologies_existing import (
    SetTechnologiesExisting,
)
from zen_garden.model.component_types.variable import GenericVariable


class CapacityEnergy(GenericVariable):
    """Installed energy capacity of a storage technology."""

    name = "capacity_energy"
    symbol = r"E_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Installed energy capacity of storage technology h at node n in year y."
    unit_category = {"energy_quantity": 1}
    masks = (EnergyCapacityAvailableMask,)
    mask_default_value = 0.0

    @classmethod
    def get_bounds(cls, model_constructor, index_sets):
        """Return bounds that keep on/off storage formulations finite."""

        technologies_on_off = (
            model_constructor.optimization_model.sets.on_off_technologies(
                SetStorageTechnologies
            )
        )

        def capacity_bounds(technology, node, year):
            if technology not in technologies_on_off:
                return 0, np.inf

            model = model_constructor.optimization_model
            parameters = model.parameters.dict_parameters
            existing_capacity = 0
            for existing_id in model.sets.get_related(
                SetTechnologies, technology, SetTechnologiesExisting
            ):
                existing_lifetime = parameters[LifetimeExisting][
                    technology, node, existing_id
                ]
                if existing_lifetime > parameters[Lifetime][technology]:
                    is_available = year > (
                        existing_lifetime - parameters[Lifetime][technology]
                    )
                else:
                    is_available = year <= existing_lifetime + 1
                if is_available:
                    existing_capacity += parameters[CapacityExistingEnergy][
                        technology, node, existing_id
                    ]

            maximum_addition = (
                len(model.sets[SetYears])
                * parameters[CapacityAdditionMaxEnergy][technology]
            )
            capacity_limit = parameters[CapacityLimitEnergy][technology, node, year]
            return 0, min(
                maximum_addition + existing_capacity,
                capacity_limit + existing_capacity,
            )

        return capacity_bounds
