import numpy as np
import xarray as xr

from zen_garden.elements.storage_technology.masks._utils import canonical_storage_array
from zen_garden.elements.storage_technology.parameters.capacity_addition_max_energy import (  # noqa: E501
    CapacityAdditionMaxEnergy,
)
from zen_garden.elements.storage_technology.parameters.capacity_addition_min_energy import (  # noqa: E501
    CapacityAdditionMinEnergy,
)
from zen_garden.elements.storage_technology.parameters.capacity_limit_energy import (
    CapacityLimitEnergy,
)
from zen_garden.model.component_types.mask import GenericMask


class EnergyInstallationRelevantMask(GenericMask):
    """Storage energy installation binaries with a nonzero model role."""

    name = "energy_installation_relevant"
    indices = ("set_technologies", "set_location", "set_years")

    @classmethod
    def get_mask(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        limit = canonical_storage_array(model.parameters[CapacityLimitEnergy])
        minimum = canonical_storage_array(model.parameters[CapacityAdditionMinEnergy])
        maximum = canonical_storage_array(model.parameters[CapacityAdditionMaxEnergy])
        mask = xr.zeros_like(limit, dtype=bool)
        mask |= minimum.notnull() & (minimum != 0)
        mask |= maximum.notnull() & np.isfinite(maximum) & (maximum != 0)
        return mask
