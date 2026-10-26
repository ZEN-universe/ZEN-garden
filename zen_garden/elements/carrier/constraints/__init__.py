"""Carrier constraints."""

from zen_garden.model.component_types.constraint import GenericConstraint

from .availability_export_constraint import AvailabilityExportConstraint
from .availability_export_yearly_constraint import AvailabilityExportYearlyConstraint
from .availability_import_constraint import AvailabilityImportConstraint
from .availability_import_yearly_constraint import AvailabilityImportYearlyConstraint
from .carbon_emission_carrier_constraint import (
    CarbonEmissionsCarrierConstraint,
)
from .carbon_emissions_carrier_total_constraint import (
    CarbonEmissionsCarrierTotalConstraint,
)
from .cost_carrier_constraint import (
    CostCarrierConstraint,
)
from .cost_carrier_total_constraint import (
    CostCarrierTotalConstraint,
)
from .cost_shed_demand_constraint import CostShedDemandConstraint
from .limit_shed_demand_constraint import LimitShedDemandConstraint
from .nodal_energy_balance_constraint import (
    NodalEnergyBalanceConstraint,
)

CARRIER_CONSTRAINTS: list[type[GenericConstraint]] = [
    CostCarrierTotalConstraint,
    CarbonEmissionsCarrierTotalConstraint,
    AvailabilityImportConstraint,
    AvailabilityExportConstraint,
    AvailabilityImportYearlyConstraint,
    AvailabilityExportYearlyConstraint,
    CostCarrierConstraint,
    CostShedDemandConstraint,
    LimitShedDemandConstraint,
    CarbonEmissionsCarrierConstraint,
    NodalEnergyBalanceConstraint,
]

__all__ = [
    "AvailabilityExportConstraint",
    "AvailabilityExportYearlyConstraint",
    "AvailabilityImportConstraint",
    "AvailabilityImportYearlyConstraint",
    "CarbonEmissionsCarrierConstraint",
    "CarbonEmissionsCarrierTotalConstraint",
    "CostCarrierConstraint",
    "CostCarrierTotalConstraint",
    "CostShedDemandConstraint",
    "LimitShedDemandConstraint",
    "NodalEnergyBalanceConstraint",
]
