"""Annualized CAPEX from existing storage energy capacity."""

from zen_garden.elements.energy_system.parameters.discount_rate import DiscountRate
from zen_garden.elements.storage_technology.parameters.existing_capex_energy import (
    ExistingCapexEnergy,
)
from zen_garden.elements.technology.parameters.depreciation_time import DepreciationTime
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class AnnualizedCapexFromExistingCapacityEnergy(GenericExpression):
    """Annualize remaining CAPEX of existing storage energy capacity."""

    name = "annualized_capex_from_existing_capacity_energy"
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def get_expression(cls, model_constructor):
        r"""Construct annualized existing storage energy CAPEX.

        Summary:
            Apply the annuity factor to remaining existing energy-capacity CAPEX.

        Formulation:

        .. math::

            a(r,d_h)c^{existing,E}_{h,n,y}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Annualized existing storage energy-capacity CAPEX.
        """
        model = model_constructor.optimization_model
        discount_rate = model.parameters[DiscountRate]
        depreciation_time = model.parameters[DepreciationTime]
        annuity = (
            ((1 + discount_rate) ** depreciation_time * discount_rate)
            / ((1 + discount_rate) ** depreciation_time - 1)
            if discount_rate != 0
            else 1 / depreciation_time
        )
        storage_technologies = model.sets[SetStorageTechnologies]
        return (
            annuity.sel(set_technologies=storage_technologies).rename(
                set_technologies="set_storage_technologies"
            )
            * model.parameters[ExistingCapexEnergy]
        )
