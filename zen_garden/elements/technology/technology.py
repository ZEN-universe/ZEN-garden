"""Class defining technologies."""

import logging
from typing import ClassVar, cast

import numpy as np
import pandas as pd

from zen_garden.elements.element import Element
from zen_garden.elements.technology.parameters import TECHNOLOGY_PARAMETERS
from zen_garden.elements.technology.sets import TECHNOLOGY_SETS
from zen_garden.topology.generic_parameter import GenericParameter
from zen_garden.topology.generic_set import GenericSet

logger = logging.getLogger(__name__)


class Technology(Element):
    """Defines parameters, variables and constraints holding for all technologies."""

    # set label
    label = "set_technologies"
    location_type: str | None = None
    reference_carrier: list[str]
    lifetime: pd.Series
    lifetime_existing: pd.Series
    own_parameters: ClassVar[list[type[GenericParameter]]] = TECHNOLOGY_PARAMETERS
    own_sets: ClassVar[list[type[GenericSet]]] = TECHNOLOGY_SETS

    def initialize_reference_carrier(self):
        """Retrieves and stores information on reference."""
        self.reference_carrier = cast(
            list[str],
            self.data_input.extract_carriers(carrier_type="reference_carrier"),
        )
        self.energy_system.set_technology_of_carrier(self.name, self.reference_carrier)

    def prepare_input_data(self) -> None:
        """Load the vintage set needed by existing-capacity parameters."""
        self.set_technologies_existing = (
            self.data_input.extract_set_technologies_existing()
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
