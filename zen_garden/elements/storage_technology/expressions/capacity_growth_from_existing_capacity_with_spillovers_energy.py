from zen_garden.elements.energy_system.parameters.interval_between_years import (
    IntervalBetweenYears,
)
from zen_garden.elements.energy_system.parameters.knowledge_depreciation_rate import (
    KnowledgeDepreciationRate,
)
from zen_garden.elements.energy_system.parameters.knowledge_spillover_rate import (
    KnowledgeSpilloverRate,
)
from zen_garden.elements.energy_system.sets.set_nodes import SetNodes
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.storage_technology.parameters.capacity_existing_energy import (
    CapacityExistingEnergy,
)
from zen_garden.elements.storage_technology.variables.capacity_addition_energy import (
    CapacityAdditionEnergy,
)
from zen_garden.elements.technology.diffusion import storage_energy_diffusion_rate
from zen_garden.elements.technology.parameters.lifetime import Lifetime
from zen_garden.elements.technology.parameters.lifetime_existing import LifetimeExisting
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression


class CapacityGrowthFromExistingCapacityWithSpilloversEnergy(GenericExpression):
    """Storage energy growth from existing capacity at all nodes."""

    name = "capacity_growth_from_existing_capacity_with_spillovers_energy"
    indices = ("set_storage_technologies", "set_nodes", "set_years")

    @classmethod
    def should_construct(cls, model_constructor):
        return model_constructor.optimization_model.parameters[
            KnowledgeSpilloverRate
        ] != float("inf")

    @classmethod
    def get_expression(cls, model_constructor):
        r"""Construct existing-capacity spillovers for storage energy.

        Summary:
            Include depreciated existing capacity at the same and other nodes.

        Formulation:

        .. math::

            d_{h,n,y}\sum_v\left[e^{existing}_{h,n,v}
            +s_h\sum_{n'\ne n}e^{existing}_{h,n',v}\right]
            (1-\delta_h)^{\Delta y(y-y_0)-1+\tau_h-\tau^{existing}_{h,n,v}}

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Storage energy growth from existing-capacity spillovers.
        """
        model = model_constructor.optimization_model
        technologies = model.sets[SetStorageTechnologies]
        interval = model.parameters[IntervalBetweenYears]
        addition = model.variables[CapacityAdditionEnergy]
        years_from_start = interval * (
            addition.coords["set_years"] - model.sets[SetYears][0]
        )
        existing_lifetime = (
            model.parameters[LifetimeExisting]
            .sel(
                set_technologies=technologies,
                set_location=model.sets[SetNodes],
            )
            .rename(
                set_technologies="set_storage_technologies",
                set_location="set_nodes",
            )
        )
        lifetime = (
            model.parameters[Lifetime]
            .sel(set_technologies=technologies)
            .rename(set_technologies="set_storage_technologies")
        )
        depreciation = model.parameters[KnowledgeDepreciationRate]
        factor = (1 - depreciation) ** (
            years_from_start - 1 + lifetime - existing_lifetime
        )
        factor = factor.where(lifetime >= existing_lifetime - years_from_start, 0)
        existing = model.parameters[CapacityExistingEnergy]
        existing = existing + model.parameters[KnowledgeSpilloverRate] * (
            existing.sum("set_nodes") - existing
        )
        return storage_energy_diffusion_rate(model_constructor) * (
            existing * factor
        ).sum("set_existing_capacity_ids")
