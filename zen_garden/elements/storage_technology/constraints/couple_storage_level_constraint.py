from zen_garden.elements.storage_technology.expressions.storage_level_change_after_self_discharge import (  # noqa: E501
    StorageLevelChangeAfterSelfDischarge,
)
from zen_garden.elements.storage_technology.expressions.storage_net_energy_inflow import (  # noqa: E501
    StorageNetEnergyInflow,
)
from zen_garden.elements.storage_technology.masks import StorageStateMask
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.constraint import GenericConstraint


class CoupleStorageLevelConstraint(GenericConstraint):
    """Couple Storage Level."""

    name = "constraint_couple_storage_level"
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours_storage",
    )
    masks = (StorageStateMask,)

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_constraint(cls, model_constructor):
        """Summary:
        Couple subsequent storage levels (time coupling constraints).

        Formulation:

        .. math::
            \\begin{aligned}
            S^{\\mathrm{level}}_{h,n,\\tilde{t}}
            ={}&S^{\\mathrm{level}}_{h,n,\\tilde{t}-1}
            (1-\\lambda^{\\mathrm{self}}_{h,n})^{\\Delta \\tilde{t}_{\\tilde{t}}} \\\\
            &+\\left(\\eta^{\\mathrm{ch}}_{h,n,y}
            F^{\\mathrm{ch}}_{h,n,\\sigma(\\tilde{t})}
            -\\frac{F^{\\mathrm{dis}}_{h,n,\\sigma(\\tilde{t})}}
            {\\eta^{\\mathrm{dis}}_{h,n,y}}
            +q^{\\mathrm{in}}_{h,n,\\sigma(\\tilde{t})}
            -F^{\\mathrm{spill}}_{h,n,\\sigma(\\tilde{t})}\\right)
            \\sum_{\\tilde{t}'=0}^{\\Delta \\tilde{t}_{\\tilde{t}}-1}
            (1-\\lambda^{\\mathrm{self}}_{h,n})^{\\tilde{t}'}.
            \\end{aligned}

        Notation:

        :math:`S^{\\mathrm{level}}_{h,n,\\tilde{t}}`: storage level of storage
        technology :math:`h` at node :math:`n` in storage time step :math:`\\tilde{t}`
        of year :math:`y`
        :math:`\\lambda^{\\mathrm{self}}_{h,n}`: self-discharge rate of storage
        technology :math:`h` at node :math:`n`
        :math:`\\Delta \\tilde{t}_{\\tilde{t}}`: duration of storage time step of
        technology :math:`h`
        :math:`\\eta^{\\mathrm{ch}}_{h,n,y}`: efficiency during charging of storage
        technology :math:`h` at node :math:`n` in year :math:`y`
        :math:`\\eta^{\\mathrm{dis}}_{h,n,y}`: efficiency during discharging of storage
        technology :math:`h` at node :math:`n` in year :math:`y`
        :math:`F^{\\mathrm{ch}}_{h,n,\\sigma(\\tilde{t})}`: charge flow into storage
        technology :math:`h` at node :math:`n` and time
        :math:`\\sigma(\\tilde{t})` in year :math:`y`
        :math:`F^{\\mathrm{dis}}_{h,n,\\sigma(\\tilde{t})}`: discharge flow out of
        storage technology :math:`h` at node :math:`n` and time
        :math:`\\sigma(\\tilde{t})` in
        year :math:`y`
        :math:`q^{\\mathrm{in}}_{h,n,\\sigma(\\tilde{t})}`: exogenous inflow into
        storage
        :math:`F^{\\mathrm{spill}}_{h,n,\\sigma(\\tilde{t})}`: storage spillage

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Constraint over its declared model indices.
"""
        model = model_constructor.optimization_model
        return (
            model.expressions[StorageLevelChangeAfterSelfDischarge]
            == model.expressions[StorageNetEnergyInflow]
        )
