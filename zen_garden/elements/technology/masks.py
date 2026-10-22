"""Structural masks shared by technology variables and constraints."""

from __future__ import annotations

import numpy as np
import xarray as xr

from zen_garden.model.component_types.mask import GenericMask


def _year_offset(model_constructor, technology: str, parameter: str, *, ceil=False):
    value = getattr(model_constructor.optimization_model.parameters, parameter).sel(
        set_technologies=technology
    )
    interval = model_constructor.config.system.interval_between_years
    ratio = float(value.item()) / interval
    return int(np.ceil(ratio) if ceil else np.floor(ratio))


def _active_previous_years(
    model_constructor, technology: str, year, parameter: str
) -> list:
    """Return modeled earlier years still active under a duration parameter."""
    years = list(model_constructor.optimization_model.sets["set_years"])
    duration = _year_offset(model_constructor, technology, parameter)
    first_year = year - (duration - 1)
    return [previous for previous in years if first_year <= previous <= year]


class ExistingCapacityMask(GenericMask):
    """Capacity committed outside the current optimization decisions."""

    name = "existing_capacity"

    @classmethod
    def build(cls, model_constructor, registry):
        existing = model_constructor.optimization_model.parameters.existing_capacities
        return existing.fillna(0) > 0


class CapacityAdditionEligibleMask(GenericMask):
    """Locations and completion years where modeled additions may be nonzero."""

    name = "capacity_addition_eligible"

    @classmethod
    def build(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        existing = model.parameters.existing_capacities.fillna(0)
        capacity_limit = model.parameters.capacity_limit
        eligible = capacity_limit.notnull() & (capacity_limit > existing)
        if not model_constructor.config.system.allow_investment:
            return xr.zeros_like(eligible, dtype=bool)

        years = list(model.sets["set_years"])
        for technology in model.sets["set_technologies"]:
            delay = _year_offset(
                model_constructor, technology, "construction_time", ceil=True
            )
            for completion_year in years:
                investment_year = completion_year - delay
                if investment_year in years:
                    continue
                existing_investment = model.parameters.capacity_investment_existing
                existing_years = existing_investment.coords.get(
                    "set_years_entire_horizon"
                )
                if existing_years is None or investment_year not in existing_years:
                    eligible.loc[technology, :, :, completion_year] = False
                    continue
                committed = (
                    existing_investment.sel(
                        set_technologies=technology,
                        set_years_entire_horizon=investment_year,
                    ).fillna(0)
                    != 0
                )
                eligible.loc[technology, :, :, completion_year] &= committed
        return eligible.fillna(False)


class InvestmentEligibleMask(GenericMask):
    """Investment years that map to an eligible completion inside the horizon."""

    name = "investment_eligible"
    dependencies = [CapacityAdditionEligibleMask]

    @classmethod
    def build(cls, model_constructor, registry):
        additions = registry[CapacityAdditionEligibleMask]
        investments = xr.zeros_like(additions, dtype=bool)
        years = list(model_constructor.optimization_model.sets["set_years"])
        technologies = model_constructor.optimization_model.sets["set_technologies"]
        for technology in technologies:
            delay = _year_offset(
                model_constructor, technology, "construction_time", ceil=True
            )
            for completion_year in years:
                investment_year = completion_year - delay
                if investment_year in years:
                    investments.loc[technology, :, :, investment_year] |= additions.loc[
                        technology, :, :, completion_year
                    ]
        return investments


class PowerCapacityAdditionMask(GenericMask):
    """Eligible power-capacity additions without the capacity-type dimension."""

    name = "power_capacity_addition"
    dependencies = [CapacityAdditionEligibleMask]

    @classmethod
    def build(cls, model_constructor, registry):
        mask = registry[CapacityAdditionEligibleMask].sel(set_capacity_types="power")
        return mask.drop_vars("set_capacity_types", errors="ignore")


class CapacityAdditionAnyTypeMask(GenericMask):
    """Addition eligibility reduced across power and energy capacity types."""

    name = "capacity_addition_any_type"
    dependencies = [CapacityAdditionEligibleMask]

    @classmethod
    def build(cls, model_constructor, registry):
        return registry[CapacityAdditionEligibleMask].any("set_capacity_types")


class PotentialNewCapacityMask(GenericMask):
    """Capacity that may be supplied by a surviving modeled addition."""

    name = "potential_new_capacity"
    dependencies = [CapacityAdditionEligibleMask]

    @classmethod
    def build(cls, model_constructor, registry):
        additions = registry[CapacityAdditionEligibleMask]
        capacity = xr.zeros_like(additions, dtype=bool)
        years = list(model_constructor.optimization_model.sets["set_years"])
        technologies = model_constructor.optimization_model.sets["set_technologies"]
        for technology in technologies:
            for year in years:
                previous_years = _active_previous_years(
                    model_constructor, technology, year, "lifetime"
                )
                capacity.loc[technology, :, :, year] = additions.sel(
                    set_technologies=technology,
                    set_years=previous_years,
                ).any("set_years")
        return capacity


class CapacityAvailableMask(GenericMask):
    """Existing or potentially constructed capacity available by model year."""

    name = "capacity_available"
    dependencies = [ExistingCapacityMask, PotentialNewCapacityMask]

    @classmethod
    def build(cls, model_constructor, registry):
        return registry[ExistingCapacityMask] | registry[PotentialNewCapacityMask]


class CapacityAnyTypeMask(GenericMask):
    """Capacity availability reduced across power and energy capacity types."""

    name = "capacity_any_type"
    dependencies = [CapacityAvailableMask]

    @classmethod
    def build(cls, model_constructor, registry):
        return registry[CapacityAvailableMask].any("set_capacity_types")


class PowerCapacityYearMask(GenericMask):
    """Power-capacity availability without the capacity-type dimension."""

    name = "power_capacity_year"
    dependencies = [CapacityAvailableMask]

    @classmethod
    def build(cls, model_constructor, registry):
        mask = registry[CapacityAvailableMask].sel(set_capacity_types="power")
        return mask.drop_vars("set_capacity_types", errors="ignore")


class ExistingCapexMask(GenericMask):
    """Outstanding CAPEX from committed capacity, including future vintages."""

    name = "existing_capex"

    @classmethod
    def build(cls, model_constructor, registry):
        existing = model_constructor.optimization_model.parameters.existing_capex
        return existing.fillna(0) != 0


class ModeledCapexMask(GenericMask):
    """Modeled overnight CAPEX still inside its depreciation period."""

    name = "modeled_capex"
    dependencies = [CapacityAdditionEligibleMask]

    @classmethod
    def build(cls, model_constructor, registry):
        additions = registry[CapacityAdditionEligibleMask]
        capex = xr.zeros_like(additions, dtype=bool)
        years = list(model_constructor.optimization_model.sets["set_years"])
        technologies = model_constructor.optimization_model.sets["set_technologies"]
        for technology in technologies:
            for year in years:
                previous_years = _active_previous_years(
                    model_constructor, technology, year, "depreciation_time"
                )
                capex.loc[technology, :, :, year] = additions.sel(
                    set_technologies=technology,
                    set_years=previous_years,
                ).any("set_years")
        return capex


class CapexActiveMask(GenericMask):
    """Existing or modeled annual CAPEX active in a model year."""

    name = "capex_active"
    dependencies = [ExistingCapexMask, ModeledCapexMask]

    @classmethod
    def build(cls, model_constructor, registry):
        return registry[ExistingCapexMask] | registry[ModeledCapexMask]


class PowerOperationMask(GenericMask):
    """Technology power capacity available in each operational time step."""

    name = "power_operation"
    dependencies = [CapacityAvailableMask]

    @classmethod
    def build(cls, model_constructor, registry):
        capacity = registry[CapacityAvailableMask].sel(set_capacity_types="power")
        capacity = capacity.drop_vars("set_capacity_types", errors="ignore")
        times = model_constructor.optimization_model.sets["set_time_steps_operation"]
        years = xr.DataArray(
            [
                model_constructor.time_steps.convert_time_step_operation2year(time)
                for time in times
            ],
            coords={"set_time_steps_operation": times},
            dims="set_time_steps_operation",
        )
        return capacity.sel(set_years=years).drop_vars("set_years", errors="ignore")


class StorageInflowMask(GenericMask):
    """Operational positions with nonzero exogenous storage inflow."""

    name = "storage_inflow"

    @classmethod
    def build(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        technologies = list(model.sets["set_storage_technologies"])
        if not technologies:
            return xr.DataArray(
                False,
                coords={
                    "set_technologies": technologies,
                    "set_location": list(model.sets["set_nodes"]),
                    "set_time_steps_operation": list(
                        model.sets["set_time_steps_operation"]
                    ),
                },
                dims=(
                    "set_technologies",
                    "set_location",
                    "set_time_steps_operation",
                ),
            )
        inflow = model.parameters.flow_storage_inflow
        return (inflow.fillna(0) != 0).rename(
            {
                "set_storage_technologies": "set_technologies",
                "set_nodes": "set_location",
            }
        )


class StorageOperationMask(GenericMask):
    """Storage operation required by capacity or exogenous inflow."""

    name = "storage_operation"
    dependencies = [PowerOperationMask, StorageInflowMask]

    @classmethod
    def build(cls, model_constructor, registry):
        return registry[PowerOperationMask] | registry[StorageInflowMask]


class StorageStateMask(GenericMask):
    """Storage states required for a year containing capacity or inflow."""

    name = "storage_state"
    dependencies = [CapacityAnyTypeMask, StorageInflowMask]

    @classmethod
    def build(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        available = registry[CapacityAnyTypeMask]
        inflow = registry[StorageInflowMask]
        storage_technologies = list(model.sets["set_storage_technologies"])
        nodes = list(model.sets["set_nodes"])
        storage_times = list(model.sets["set_time_steps_storage"])
        mask = xr.DataArray(
            False,
            coords={
                "set_technologies": storage_technologies,
                "set_location": nodes,
                "set_time_steps_storage": storage_times,
            },
            dims=(
                "set_technologies",
                "set_location",
                "set_time_steps_storage",
            ),
        )
        for year in model.sets["set_years"]:
            operation_times = (
                model_constructor.time_steps.get_time_steps_year2operation(year)
            )
            year_active = available.sel(
                set_technologies=storage_technologies,
                set_location=nodes,
                set_years=year,
            ) | inflow.sel(
                set_technologies=storage_technologies,
                set_location=nodes,
                set_time_steps_operation=operation_times,
            ).any(
                "set_time_steps_operation"
            )
            mask.loc[
                :, :, model_constructor.time_steps.get_time_steps_year2storage(year)
            ] = year_active
        return mask


class OnOffTechnologyMask(GenericMask):
    """Technology-location-time tuples configured for on/off operation."""

    name = "on_off_technology"

    @classmethod
    def build(cls, model_constructor, registry):
        index_values, index_names = model_constructor.create_custom_set(
            [
                "set_technologies",
                "set_on_off",
                "set_location",
                "set_time_steps_operation",
            ]
        )
        index_names.pop(1)
        return model_constructor.optimization_model.sets.indices_to_mask(
            index_values, index_names, (0, 0)
        )[0]


class InstallationRelevantMask(GenericMask):
    """Positions where the installation binary has a nonzero model role."""

    name = "installation_relevant"

    @classmethod
    def build(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        mask = xr.zeros_like(model.parameters.capacity_limit, dtype=bool)
        minimum = model.parameters.capacity_addition_min
        maximum = model.parameters.capacity_addition_max
        mask |= minimum.notnull() & (minimum != 0)
        mask |= maximum.notnull() & np.isfinite(maximum) & (maximum != 0)

        technologies = list(model.sets["set_transport_technologies"])
        if technologies:
            edges = list(model.sets["set_edges"])
            distance_cost = (
                model.parameters.distance.loc[technologies, edges]
                * model.parameters.capex_per_distance_transport.loc[technologies, edges]
                != 0
            ).rename(
                {
                    "set_transport_technologies": "set_technologies",
                    "set_edges": "set_location",
                }
            )
            mask.loc[technologies, :, edges, :] |= distance_cost
        return mask


class FiniteTransportRouteMask(GenericMask):
    """Transport routes with finite distance."""

    name = "finite_transport_route"

    @classmethod
    def build(cls, model_constructor, registry):
        model = model_constructor.optimization_model
        technologies = list(model.sets["set_transport_technologies"])
        edges = list(model.sets["set_edges"])
        mask = xr.DataArray(
            False,
            coords={"set_technologies": technologies, "set_location": edges},
            dims=("set_technologies", "set_location"),
        )
        if technologies:
            mask.loc[technologies, edges] = np.isfinite(
                model.parameters.distance.loc[technologies, edges]
            ).data
        return mask
