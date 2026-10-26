from zen_garden.elements.storage_technology.parameters.capacity_existing_energy import (
    CapacityExistingEnergy,
)
from zen_garden.elements.technology.parameters._existing_quantity import (
    compute_existing_quantity,
)
from zen_garden.elements.technology.parameters.lifetime import Lifetime
from zen_garden.elements.technology.parameters.lifetime_existing import LifetimeExisting
from zen_garden.model.component_types.parameter import GenericParameter, ParameterData


class ExistingCapacitiesEnergy(GenericParameter):
    """Existing storage energy capacity available in each modeled year."""

    name = "existing_capacities_energy"
    symbol = r"e^{\mathrm{existing}}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Existing storage energy capacity remaining in each modeled year."
    unit_category = {"energy_quantity": 1}
    dependencies = (CapacityExistingEnergy, LifetimeExisting, Lifetime)

    @classmethod
    def get_input_data(cls, element):
        """Return no input because the parameter is derived."""
        return None

    @classmethod
    def get_parameter_value(cls, model_constructor):
        """Aggregate active existing storage-energy vintages."""
        return ParameterData(
            compute_existing_quantity(
                model_constructor,
                "capacity_existing_energy",
                storage_energy=True,
            ),
            cls.indices,
        )
