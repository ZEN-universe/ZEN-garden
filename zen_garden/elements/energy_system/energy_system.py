"""Class defining a standard EnergySystem."""

import copy
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from zen_garden.model.config import Config
from zen_garden.model.time_steps import TimeStepsDicts
from zen_garden.preprocess.data_input import DataInput
from zen_garden.services.network_topology import NetworkTopology
from zen_garden.types import YearSpecificTs

if TYPE_CHECKING:
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.dataset_path_resolver import DatasetPathResolver
    from zen_garden.services.input_repository import InputRepository
    from zen_garden.services.scenario_dict import ScenarioDict
    from zen_garden.utils.input_data_checks import InputDataChecks

logger = logging.getLogger(__name__)


class EnergySystem:
    """Class defining a standard energy system."""

    name: str = "EnergySystem"

    def __init__(
        self,
        config: Config,
        unit_handling: "UnitHandling",
        dataset_path_resolver: "DatasetPathResolver",
        scenario_dict: "ScenarioDict",
        input_data_checks: "InputDataChecks",
        time_steps: "TimeStepsDicts",
        year_specific_ts: "YearSpecificTs",
        input_repository: "InputRepository",
    ):
        """Initialization of the energy_system.

        :param config: The Config of the entire setup
        """
        # set attributes
        self.config = config

        # empty dict of technologies of carrier
        self.dict_technology_of_carrier: dict[str, list[str]] = {}
        # The timesteps
        self.time_steps = time_steps
        self.input_repository = input_repository

        # create DataInput object
        self.data_input = DataInput(
            element=self,
            energy_system=self,
            unit_handling=unit_handling,
            config=config,
            scenario_dict=scenario_dict,
            input_data_checks=input_data_checks,
            year_specific_ts=year_specific_ts,
            folder_path=input_repository.folder_path,
            input_repository=input_repository,
        )
        self.network_topology = NetworkTopology(
            config=self.config,
            input_repository=input_repository,
            input_data_checks=input_data_checks,
            unit_handling=unit_handling,
        )
        # initialize empty set_carriers list
        self.set_carriers: list[str] = []
        # dict to save the parameter units (and save them in the results later on)
        self.units: dict[str, Any] = {}
        self.time_steps_operation_duration: pd.Series | None = None
        self.time_steps_storage_duration: pd.Series | None = None

    def store_input_data(self):
        """Retrieves and stores input data for EnergySystem as attributes."""
        # in class <EnergySystem>, all sets are constructed
        self.set_technologies = self.config.system.set_technologies
        # base time steps
        self.set_hours_all_years = list(
            range(
                0,
                self.config.system.unaggregated_time_steps_per_year
                * self.config.system.optimized_years,
            )
        )
        self.set_hours = list(
            range(0, self.config.system.unaggregated_time_steps_per_year)
        )

        # yearly time steps
        self.set_years = list(range(self.config.system.optimized_years))
        self.set_years_entire_horizon = copy.deepcopy(self.set_years)
        time_steps_yearly_duration = self.time_steps.calculate_time_step_duration(
            self.set_years, self.set_hours_all_years
        )
        self.sequence_time_steps_yearly = np.concatenate(
            [
                [time_step] * time_steps_yearly_duration[time_step]
                for time_step in time_steps_yearly_duration
            ]
        )
        self.time_steps.sequence_time_steps_yearly = self.sequence_time_steps_yearly
        # list containing simulated years
        #   (needed for DataInput._convert_real_to_generic_time_indices())
        self.set_time_steps_years = list(
            range(
                self.config.system.reference_year,
                self.config.system.reference_year
                + self.config.system.optimized_years
                * self.config.system.interval_between_years,
                self.config.system.interval_between_years,
            )
        )
        # parameters whose time-dependant data should not be interpolated
        # (for years without data) in DataInput._convert_real_to_generic_time_indices()
        self.parameters_interpolation_off = self.input_repository.read_mapping_file(
            "parameters_interpolation_off"
        )
        # technology-specific
        self.set_conversion_technologies = (
            self.config.system.set_conversion_technologies
        )
        self.set_transport_technologies = self.config.system.set_transport_technologies
        self.set_storage_technologies = self.config.system.set_storage_technologies
        self.set_retrofitting_technologies = (
            self.config.system.set_retrofitting_technologies
        )
        # discount rate
        self.discount_rate = self.data_input.extract_input_data(
            "discount_rate", index_sets=[], unit_category={}
        )
        # carbon emissions limit
        self.carbon_emissions_annual_limit = self.data_input.extract_input_data(
            "carbon_emissions_annual_limit",
            index_sets=["set_years"],
            unit_category={"emissions": 1},
        )
        _fraction_year = (
            self.config.system.unaggregated_time_steps_per_year
            / self.config.system.total_hours_per_year
        )
        self.carbon_emissions_annual_limit = (
            self.carbon_emissions_annual_limit * _fraction_year
        )  # reduce to fraction of year
        self.carbon_emissions_budget = self.data_input.extract_input_data(
            "carbon_emissions_budget", index_sets=[], unit_category={"emissions": 1}
        )
        self.carbon_emissions_cumulative_existing = self.data_input.extract_input_data(
            "carbon_emissions_cumulative_existing",
            index_sets=[],
            unit_category={"emissions": 1},
        )
        # price carbon emissions
        self.price_carbon_emissions = self.data_input.extract_input_data(
            "price_carbon_emissions",
            index_sets=["set_years"],
            unit_category={"money": 1, "emissions": -1},
        )
        self.price_carbon_emissions_budget_overshoot = (
            self.data_input.extract_input_data(
                "price_carbon_emissions_budget_overshoot",
                index_sets=[],
                unit_category={"money": 1, "emissions": -1},
            )
        )
        self.price_carbon_emissions_annual_overshoot = (
            self.data_input.extract_input_data(
                "price_carbon_emissions_annual_overshoot",
                index_sets=[],
                unit_category={"money": 1, "emissions": -1},
            )
        )
        # market share unbounded
        self.market_share_unbounded = self.data_input.extract_input_data(
            "market_share_unbounded", index_sets=[], unit_category={}
        )
        # knowledge_spillover_rate
        self.knowledge_depreciation_rate = self.data_input.extract_input_data(
            "knowledge_depreciation_rate", index_sets=[], unit_category={}
        )
        self.knowledge_spillover_rate = self.data_input.extract_input_data(
            "knowledge_spillover_rate", index_sets=[], unit_category={}
        )

    def set_technology_of_carrier(self, technology, list_technology_of_carrier):
        """Appends technology to carrier in dict_technology_of_carrier.

        :param technology: name of technology in model
        :param list_technology_of_carrier: list of carriers correspondent to technology
        """
        for carrier in list_technology_of_carrier:
            if carrier not in self.dict_technology_of_carrier:
                self.dict_technology_of_carrier[carrier] = [technology]
                self.set_carriers.append(carrier)
            elif technology not in self.dict_technology_of_carrier[carrier]:
                self.dict_technology_of_carrier[carrier].append(technology)

    def calculate_connected_edges(self, *args):
        """Calculates connected edges using the network topology.

        See
        :meth:`zen_garden.services.network_topology.NetworkTopology.calculate_connected_edges`
        """
        return self.network_topology.calculate_connected_edges(*args)

    @property
    def set_nodes(self):
        """Returns the set of nodes from the network topology."""
        return self.network_topology.set_nodes

    @property
    def set_edges(self):
        """Returns the set of edges from the network topology."""
        return self.network_topology.set_edges

    @property
    def set_nodes_on_edges(self):
        """Returns the set of nodes on edges from the network topology."""
        return self.network_topology.set_nodes_on_edges

    @property
    def set_haversine_distances_edges(self):
        """Returns the set of haversine distances on edges from the network topology."""
        return self.network_topology.set_haversine_distances_edges
