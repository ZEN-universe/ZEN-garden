"""Carrier constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint

from .availability_import_export_constraint import AvailabilityImportExportConstraint
from .availability_import_export_yearly_constraint import (
    AvailabilityImportExportYearlyConstraint,
)
from .carbon_emission_carrier_constraint import CarbonEmissionsCarrierConstraint
from .carbon_emissions_carrier_total_constraint import (
    CarbonEmissionsCarrierTotalConstraint,
)
from .cost_carrier_constraint import CostCarrierConstraint
from .cost_carrier_total_constraint import CostCarrierTotalConstraint
from .cost_limit_shed_demand_constraint import CostLimitShedDemandConstraint
from .nodal_energy_balance_constraint import NodalEnergyBalanceConstraint

CARRIER_CONSTRAINTS: list[type[GenericConstraint]] = [
    CostCarrierTotalConstraint,
    CarbonEmissionsCarrierTotalConstraint,
    AvailabilityImportExportConstraint,
    AvailabilityImportExportYearlyConstraint,
    CostCarrierConstraint,
    CostLimitShedDemandConstraint,
    CarbonEmissionsCarrierConstraint,
    NodalEnergyBalanceConstraint,
]

__all__ = [
    "CostCarrierTotalConstraint",
    "CarbonEmissionsCarrierTotalConstraint",
    "AvailabilityImportExportConstraint",
    "AvailabilityImportExportYearlyConstraint",
    "CostCarrierConstraint",
    "CostLimitShedDemandConstraint",
    "CarbonEmissionsCarrierConstraint",
    "NodalEnergyBalanceConstraint",
]
