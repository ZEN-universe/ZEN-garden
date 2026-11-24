"""Net energy entering storage during a storage time step."""

import pandas as pd

from zen_garden.elements.energy_system.parameters.duration_representative_hours_storage import (  # noqa: E501
    DurationRepresentativeHoursStorage,
)
from zen_garden.elements.energy_system.sets.set_years import SetYears
from zen_garden.elements.storage_technology.parameters.efficiency_charge import (
    EfficiencyCharge,
)
from zen_garden.elements.storage_technology.parameters.efficiency_discharge import (
    EfficiencyDischarge,
)
from zen_garden.elements.storage_technology.parameters.flow_storage_inflow import (
    FlowStorageInflow,
)
from zen_garden.elements.storage_technology.parameters.self_discharge import (
    SelfDischarge,
)
from zen_garden.elements.storage_technology.storage_time import (
    get_previous_storage_time_step_arrays,
)
from zen_garden.elements.storage_technology.variables.flow_storage_charge import (
    FlowStorageCharge,
)
from zen_garden.elements.storage_technology.variables.flow_storage_discharge import (
    FlowStorageDischarge,
)
from zen_garden.elements.storage_technology.variables.flow_storage_spillage import (
    FlowStorageSpillage,
)
from zen_garden.elements.technology.sets.set_storage_technologies import (
    SetStorageTechnologies,
)
from zen_garden.model.component_types.expression import GenericExpression
from zen_garden.model.linear_expression import map_and_expand


class StorageNetEnergyInflow(GenericExpression):
    """Combine charge, discharge, inflow, spillage, and standing-loss duration."""

    name = "storage_net_energy_inflow"
    unit_category = {"energy_quantity": 1, "time": -1}
    indices = (
        "set_storage_technologies",
        "set_nodes",
        "set_years",
        "set_representative_hours_storage",
    )

    @staticmethod
    def _get_storage_to_operation_mapping(model_constructor) -> pd.Series:
        """Return the operational representative selected by each storage hour."""
        mapping = model_constructor.time_steps.storage_to_operation
        assert mapping is not None
        years = list(model_constructor.optimization_model.sets[SetYears])
        return mapping.loc[mapping.index.get_level_values("set_years").isin(years)]

    @classmethod
    def should_construct(cls, model_constructor):
        return bool(model_constructor.optimization_model.sets[SetStorageTechnologies])

    @classmethod
    def get_expression(cls, model_constructor):
        """Construct the model component.

        Summary:
            Combine charge, discharge, inflow, spillage, and standing-loss duration.

        Formulation:

        .. math::

            \\tau_r(\\eta^{ch}F^{ch}_{h,n,y,r}-F^{dis}_{h,n,y,r}/\\eta^{dis}
            +f^{inflow}_{h,n,y,r}-F^{spill}_{h,n,y,r})

        The formulation applies over valid combinations of the declared indices.
        Empty or masked combinations are not constructed.

        Notation:
            Symbols and indices follow ``sets_params_constraints.rst``.

        Args:
            model_constructor: Object containing the optimization model.

        Returns:
            Expression over its declared model indices.
        """
        optimization_model = model_constructor.optimization_model
        self_discharge = optimization_model.parameters[SelfDischarge]
        duration = optimization_model.parameters[DurationRepresentativeHoursStorage]
        with_discharge = (1 - (1 - self_discharge) ** duration) / self_discharge
        multiplier = with_discharge.where(self_discharge != 0, 0.0) + duration.where(
            self_discharge == 0, 0.0
        )
        net_flow = (
            optimization_model.variables[FlowStorageCharge]
            * optimization_model.parameters[EfficiencyCharge]
            - optimization_model.variables[FlowStorageDischarge].to_linexpr()
            / optimization_model.parameters[EfficiencyDischarge]
            + optimization_model.parameters[FlowStorageInflow]
            - optimization_model.variables[FlowStorageSpillage]
        )
        mapped = map_and_expand(
            net_flow, cls._get_storage_to_operation_mapping(model_constructor)
        )
        _, _, coupling_mask = get_previous_storage_time_step_arrays(model_constructor)
        return (mapped * multiplier).where(coupling_mask, 0.0)
