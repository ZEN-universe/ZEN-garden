"""Class defining a standard EnergySystem.
Contains methods to construct the energy system from the given input data and that
defines the variables, parameters and constraints which apply to the Energy System.
The class takes the abstract optimization model as an input.
"""

import copy
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from zen_garden.model.config import Config
from zen_garden.model.time_steps import TimeStepsDicts
from zen_garden.preprocess.data_input import DataInput
from zen_garden.types import YearSpecificTs

if TYPE_CHECKING:
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.dataset_path_resolver import DatasetPathResolver
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
    ):
        """Initialization of the energy_system.

        :param config: The Config of the entire setup
        """
        # set attributes
        self.config = config
        self.unit_handling = unit_handling

        # empty dict of technologies of carrier
        self.dict_technology_of_carrier = {}
        # The timesteps
        self.time_steps = time_steps

        # create DataInput object
        self.data_input = DataInput(
            element=self,
            energy_system=self,
            unit_handling=self.unit_handling,
            config=config,
            scenario_dict=scenario_dict,
            input_data_checks=input_data_checks,
            year_specific_ts=year_specific_ts,
            folder_path=Path(dataset_path_resolver.folder_of_set("energy_system")),
        )
        # initialize empty set_carriers list
        self.set_carriers = []
        # dict to save the parameter units (and save them in the results later on)
        self.units = {}

    def store_input_data(self):
        """Retrieves and stores input data for EnergySystem as attributes."""
        # in class <EnergySystem>, all sets are constructed
        self.set_nodes = self.data_input.extract_locations()
        self.set_nodes_on_edges = self.calculate_edges_from_nodes()
        self.set_edges = list(self.set_nodes_on_edges.keys())
        self.set_haversine_distances_edges = (
            self.calculate_haversine_distances_from_nodes()
        )
        self.set_technologies = self.config.system.set_technologies
        # base time steps
        self.set_base_time_steps = list(
            range(
                0,
                self.config.system.unaggregated_time_steps_per_year
                * self.config.system.optimized_years,
            )
        )
        self.set_base_time_steps_yearly = list(
            range(0, self.config.system.unaggregated_time_steps_per_year)
        )

        # yearly time steps
        self.set_time_steps_yearly = list(range(self.config.system.optimized_years))
        self.set_time_steps_yearly_entire_horizon = copy.deepcopy(
            self.set_time_steps_yearly
        )
        time_steps_yearly_duration = self.time_steps.calculate_time_step_duration(
            self.set_time_steps_yearly, self.set_base_time_steps
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
        self.parameters_interpolation_off = self.data_input.read_input_json(
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
            index_sets=["set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
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
            index_sets=["set_time_steps_yearly"],
            time_steps="set_time_steps_yearly",
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

    def calculate_edges_from_nodes(self):
        """Calculates set_nodes_on_edges from set_nodes.

        :return: set_nodes_on_edges: dict with edges and corresponding nodes
        """
        set_nodes_on_edges = {}
        # read edge file
        set_edges_input = self.data_input.extract_locations(extract_nodes=False)
        assert isinstance(set_edges_input, pd.DataFrame)
        for edge in set_edges_input.index:
            set_nodes_on_edges[edge] = (
                set_edges_input.loc[edge, "node_from"],
                set_edges_input.loc[edge, "node_to"],
            )
        return set_nodes_on_edges

    def calculate_haversine_distances_from_nodes(self):
        """Computes the distance (in km) between two nodes.

        The Haversine function is used to compute the distance in kilometers based on
         their lon lat coordinates.

        :return: dict containing all edges along with their distances
        """
        set_haversine_distances_of_edges = {}

        # read coords file
        df_coords_input = self.data_input.extract_locations(extract_coordinates=True)
        if not isinstance(df_coords_input, pd.DataFrame):
            raise TypeError(
                "[EnergySystem] df_coords_input is not of type pd.DataFrame"
            )
        coords = df_coords_input.set_index("node")
        # TODO: load this outside this function
        self.config.system.coords = cast(
            dict[str, dict[str, float]], coords.T.to_dict()
        )

        # convert coords from decimal degrees to radians
        df_coords_input["lon"] = df_coords_input["lon"] * np.pi / 180
        df_coords_input["lat"] = df_coords_input["lat"] * np.pi / 180
        # Radius of the Earth in kilometers
        radius = 6371.0
        for edge, nodes in self.set_nodes_on_edges.items():
            node_1, node_2 = nodes
            coords1 = df_coords_input[df_coords_input["node"] == node_1]
            coords2 = df_coords_input[df_coords_input["node"] == node_2]
            # Haversine formula
            dlon = coords2["lon"].squeeze() - coords1["lon"].squeeze()
            dlat = coords2["lat"].squeeze() - coords1["lat"].squeeze()
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(coords1["lat"].squeeze())
                * np.cos(coords2["lat"].squeeze())
                * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = radius * c
            set_haversine_distances_of_edges[edge] = distance
        multiplier = self.unit_handling.get_unit_multiplier(
            "km", attribute_name="distance"
        )
        set_haversine_distances_of_edges = {
            key: value * multiplier
            for key, value in set_haversine_distances_of_edges.items()
        }
        return set_haversine_distances_of_edges

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

    def calculate_connected_edges(self, node, direction: str):
        """Calculates connected edges going in or going out.

        :param node: current node, connected by edges
        :param direction: direction of edges, either in or out. In: node = endnode,
            out: node = startnode
        :return: _set_connected_edges: list of connected edges
        """
        if direction == "in":
            # second entry is node into which the flow goes
            _set_connected_edges = [
                edge
                for edge in self.set_nodes_on_edges
                if self.set_nodes_on_edges[edge][1] == node
            ]
        elif direction == "out":
            # first entry is node out of which the flow starts
            _set_connected_edges = [
                edge
                for edge in self.set_nodes_on_edges
                if self.set_nodes_on_edges[edge][0] == node
            ]
        else:
            raise KeyError(f"invalid direction '{direction}'")
        return _set_connected_edges

    def calculate_reversed_edge(self, edge):
        """Calculates the reversed edge corresponding to an edge.

        :param edge: input edge
        :return: _reversed_edge: edge corresponding to the reversed direction of edge
        """
        _node_out, _node_in = self.set_nodes_on_edges[edge]
        for _reversed_edge in self.set_nodes_on_edges:
            if (
                _node_out == self.set_nodes_on_edges[_reversed_edge][1]
                and _node_in == self.set_nodes_on_edges[_reversed_edge][0]
            ):
                return _reversed_edge
        raise KeyError(
            f"Edge {edge} has no reversed edge. "
            f"However, at least one transport technology is bidirectional"
        )
