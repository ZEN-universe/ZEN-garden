"""Defines the parameters, variables and constraints that hold for all technologies.
The class takes the abstract optimization model as an input, and returns the parameters,
variables and constraints that hold for all technologies.
"""

import logging
from typing import cast

import numpy as np
import pandas as pd

from zen_garden.elements.element import Element

logger = logging.getLogger(__name__)


class Technology(Element):
    """Defines parameters, variables and constraints holding for all technologies."""

    # set label
    label = "set_technologies"
    location_type = None

    def initialize_reference_carrier(self):
        """Retrieves and stores information on reference."""
        self.reference_carrier = cast(
            list[str],
            self.data_input.extract_carriers(carrier_type="reference_carrier"),
        )
        self.energy_system.set_technology_of_carrier(self.name, self.reference_carrier)

    def store_input_data(self):
        """Retrieves and stores input data for element as attributes.

        Each Child class overwrites method to store different attributes.
        """
        # set attributes of technology
        set_location = self.location_type
        self.capacity_addition_min = self.data_input.extract_input_data(
            "capacity_addition_min",
            index_sets=[],
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.capacity_addition_max = self.data_input.extract_input_data(
            "capacity_addition_max",
            index_sets=[],
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.capacity_addition_unbounded = self.data_input.extract_input_data(
            "capacity_addition_unbounded",
            index_sets=[],
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.lifetime = self.data_input.extract_input_data(
            "lifetime", index_sets=[], unit_category={}
        )
        if "depreciation_time" in self.data_input.attribute_dict:
            self.depreciation_time = self.data_input.extract_input_data(
                "depreciation_time", index_sets=[], unit_category={}
            )
            self.depreciation_time[0] = np.max(
                (
                    self.config.system.interval_between_years,
                    self.depreciation_time[0],
                )
            )
        else:
            self.depreciation_time = self.lifetime.copy()
        self.construction_time = self.data_input.extract_input_data(
            "construction_time", index_sets=[], unit_category={}
        )
        # maximum diffusion rate
        self.max_diffusion_rate = self.data_input.extract_input_data(
            "max_diffusion_rate",
            index_sets=["set_years"],
            unit_category={},
        )

        # add all raw time series to dict
        self.raw_time_series["min_load"] = self.data_input.extract_input_data(
            "min_load",
            index_sets=[set_location, "set_hours"],
            unit_category={},
        )
        self.raw_time_series["max_load"] = self.data_input.extract_input_data(
            "max_load",
            index_sets=[set_location, "set_hours"],
            unit_category={},
        )
        self.raw_time_series["opex_specific_variable"] = (
            self.data_input.extract_input_data(
                "opex_specific_variable",
                index_sets=[set_location, "set_hours"],
                unit_category={"money": 1, "energy_quantity": -1},
            )
        )
        # non-time series input data
        self.capacity_limit = self.data_input.extract_input_data(
            "capacity_limit",
            index_sets=[set_location, "set_years"],
            unit_category={"energy_quantity": 1, "time": -1},
        )

        # lower capacity limit
        self.capacity_lower_limit = self.data_input.extract_input_data(
            "capacity_lower_limit",
            index_sets=[set_location, "set_years"],
            unit_category={"energy_quantity": 1, "time": -1},
        )

        self.carbon_intensity_technology = self.data_input.extract_input_data(
            "carbon_intensity_technology",
            index_sets=[set_location],
            unit_category={"emissions": 1, "energy_quantity": -1},
        )
        # extract existing capacity
        self.set_technologies_existing = (
            self.data_input.extract_set_technologies_existing()
        )
        self.capacity_existing = self.data_input.extract_input_data(
            "capacity_existing",
            index_sets=[set_location, "set_technologies_existing"],
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.capacity_investment_existing = self.data_input.extract_input_data(
            "capacity_investment_existing",
            index_sets=[set_location, "set_years"],
            unit_category={"energy_quantity": 1, "time": -1},
        )
        self.lifetime_existing = self.data_input.extract_lifetime_existing(
            "capacity_existing", index_sets=[set_location, "set_technologies_existing"]
        )

    def calculate_capex_of_capacities_existing(self):
        """This method calculates the annualized capex of the existing capacities.

        :param storage_energy: boolean if energy storage
        :return: capex of existing capacities
        """
        return self.capacity_existing.to_frame().apply(
            lambda _capacity_existing: self.calculate_capex_of_single_capacity(
                _capacity_existing.squeeze(), _capacity_existing.name
            ),
            axis=1,
        )

    def calculate_capex_of_single_capacity(self, capacity, index, **kwargs):
        """Calculates annualized capex of existing capacity, implemented in child class.

        :param args: arguments
        """
        raise NotImplementedError

    def calculate_fraction_of_year(self):
        """Calculate fraction of year."""
        # only account for fraction of year
        fraction_year = (
            self.config.system.unaggregated_time_steps_per_year
            / self.config.system.total_hours_per_year
        )
        return fraction_year

    def add_new_capacity_addition_tech(
        self, capacity_addition: pd.Series, capex: pd.Series, step_horizon: list
    ):
        """Adds the newly built capacity to the existing capacity.

        :param capacity_addition: pd.Series of newly built capacity of technology
        :param capex: pd.Series of capex of newly built capacity of technology
        :param step_horizon: current horizon step
        """
        system = self.config.system
        # reduce lifetime of existing capacities and add new remaining lifetime
        delta_lifetime = step_horizon[-1] - step_horizon[0]
        self.lifetime_existing = (
            self.lifetime_existing
            - system.interval_between_years * (delta_lifetime + 1)
        ).clip(lower=0)
        # new capacity
        new_capacity_addition = capacity_addition[step_horizon]
        new_capex = capex[step_horizon]
        # if at least one value unequal to zero
        if not (new_capacity_addition.stack() == 0).all():
            # add new index to set_technologies_existing
            index_step_horizon = list(range(len(step_horizon)))
            index_new_technology = [
                int(max(self.set_technologies_existing)) + 1 + idx
                for idx in index_step_horizon
            ]
            self.set_technologies_existing = np.append(
                self.set_technologies_existing, index_new_technology
            )
            # add new remaining lifetime
            lifetime = self.lifetime_existing.unstack()
            lifetime[index_new_technology] = [
                self.lifetime[0]
                - system.interval_between_years * (delta_lifetime - idx + 1)
                for idx in index_step_horizon
            ]
            self.lifetime_existing = lifetime.stack()

            for type_capacity in list(
                set(new_capacity_addition.index.get_level_values(0))
            ):
                # if power
                if type_capacity == system.set_capacity_types[0]:
                    energy_string = ""
                # if energy
                else:
                    energy_string = "_energy"
                capacity_existing = getattr(self, "capacity_existing" + energy_string)
                capex_capacity_existing = getattr(
                    self, "capex_capacity_existing" + energy_string
                )
                # add new existing capacity
                capacity_existing = capacity_existing.unstack()
                capacity_existing[index_new_technology] = new_capacity_addition.loc[
                    type_capacity
                ]
                setattr(
                    self, "capacity_existing" + energy_string, capacity_existing.stack()
                )
                # calculate capex of existing capacity
                capex_capacity_existing = capex_capacity_existing.unstack()
                capex_capacity_existing[index_new_technology] = new_capex.loc[
                    type_capacity
                ]
                setattr(
                    self,
                    "capex_capacity_existing" + energy_string,
                    capex_capacity_existing.stack(),
                )

    def add_new_capacity_investment(
        self, capacity_investment: pd.Series, step_horizon: list
    ):
        """Adds the newly invested capacity to the list of invested capacity.

        :param capacity_investment: pd.Series of newly built capacity of technology
        :param step_horizon: optimization time step
        """
        system = self.config.system
        new_capacity_investment = capacity_investment[step_horizon]
        new_capacity_investment = new_capacity_investment.fillna(0)
        if not (new_capacity_investment.stack() == 0).all():
            for type_capacity in list(
                set(new_capacity_investment.index.get_level_values(0))
            ):
                # if power
                if type_capacity == system.set_capacity_types[0]:
                    energy_string = ""
                # if energy
                else:
                    energy_string = "_energy"
                capacity_investment_existing = getattr(
                    self, "capacity_investment_existing" + energy_string
                )
                # add new existing invested capacity
                capacity_investment_existing = capacity_investment_existing.unstack()
                capacity_investment_existing[step_horizon] = (
                    new_capacity_investment.loc[type_capacity]
                )
                setattr(
                    self,
                    "capacity_investment_existing" + energy_string,
                    capacity_investment_existing.stack(),
                )
