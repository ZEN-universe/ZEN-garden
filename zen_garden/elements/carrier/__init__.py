"""Carrier constraints."""

from zen_garden.constraints.generic_constraint import GenericConstraint
from zen_garden.elements.carrier.carrier import Carrier

from zen_garden.elements.carrier.constraints.availability_import_export_constraint import AvailabilityImportExportConstraint
from zen_garden.elements.carrier.constraints.availability_import_export_yearly_constraint import (
    AvailabilityImportExportYearlyConstraint,
)
from zen_garden.elements.carrier.constraints.carbon_emission_carrier_constraint import CarbonEmissionsCarrierConstraint
from zen_garden.elements.carrier.constraints.carbon_emissions_carrier_total_constraint import (
    CarbonEmissionsCarrierTotalConstraint,
)
from zen_garden.elements.carrier.constraints.cost_carrier_constraint import CostCarrierConstraint
from zen_garden.elements.carrier.constraints.cost_carrier_total_constraint import CostCarrierTotalConstraint
from zen_garden.elements.carrier.constraints.cost_limit_shed_demand_constraint import CostLimitShedDemandConstraint
from zen_garden.elements.carrier.constraints.nodal_energy_balance_constraint import NodalEnergyBalanceConstraint

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
    "Carrier",
    "CostCarrierTotalConstraint",
    "CarbonEmissionsCarrierTotalConstraint",
    "AvailabilityImportExportConstraint",
    "AvailabilityImportExportYearlyConstraint",
    "CostCarrierConstraint",
    "CostLimitShedDemandConstraint",
    "CarbonEmissionsCarrierConstraint",
    "NodalEnergyBalanceConstraint",
]
