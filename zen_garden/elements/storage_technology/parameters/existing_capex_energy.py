from zen_garden.elements.storage_technology.parameters.capex_capacity_existing_energy import (  # noqa: E501
    CapexCapacityExistingEnergy,
)
from zen_garden.elements.technology.parameters._existing_quantity import (
    compute_existing_quantity,
)
from zen_garden.elements.technology.parameters.lifetime import Lifetime
from zen_garden.elements.technology.parameters.lifetime_existing import LifetimeExisting
from zen_garden.model.component_types.parameter import GenericParameter, ParameterData


class ExistingCapexEnergy(GenericParameter):
    """Outstanding existing storage-energy CAPEX in each modeled year."""

    name = "existing_capex_energy"
    symbol = r"c^{\mathrm{existing,E}}_{h,n,y}"
    indices = ("set_storage_technologies", "set_nodes", "set_years")
    doc = "Outstanding CAPEX of existing storage energy capacity by year."
    unit_category = {"money": 1}
    dependencies = (CapexCapacityExistingEnergy, LifetimeExisting, Lifetime)

    @classmethod
    def get_input_data(cls, element):
        """Return no input because the parameter is derived."""
        return None

    @classmethod
    def get_parameter_value(cls, model_constructor):
        """Aggregate active existing storage-energy CAPEX vintages."""
        return ParameterData(
            compute_existing_quantity(
                model_constructor,
                "capex_capacity_existing_energy",
                storage_energy=True,
            ),
            cls.indices,
        )
