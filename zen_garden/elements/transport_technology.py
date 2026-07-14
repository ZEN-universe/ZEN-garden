"""Class defining the parameters, variables and constraints that hold for all transport
technologies. The class takes the abstract optimization model as an input, and returns
the parameters, variables and constraints that hold for the transport technologies.
"""

import logging
from typing import override

import numpy as np

from zen_garden.elements.technology import Technology

logger = logging.getLogger(__name__)


class TransportTechnology(Technology):
    # set label
    label = "set_transport_technologies"
    location_type = "set_edges"
    dict_reversed_edges: dict[str, str] = {}

    @override
    def _initialize(self):
        """Retrieves and stores information on reference, input and output carriers."""
        # get reference carrier from class <Technology>
        super().initialize_reference_carrier()

    def store_input_data(self):
        """Retrieves and stores input data for element as attributes.

        Each Child class overwrites method to store different attributes.
        """
        # get attributes from class <Technology>
        super().store_input_data()
        # set attributes for parameters of child class <TransportTechnology>
        self.distance = self.data_input.extract_input_data(
            "distance", index_sets=["set_edges"], unit_category={"distance": 1}
        )
        if "/ kilometer" in str(
            self.units["carbon_intensity_technology"]["unit_in_base_units"].units
        ):
            self.carbon_intensity_technology = self.data_input.extract_input_data(
                "carbon_intensity_technology",
                index_sets=["set_edges"],
                unit_category={"emissions": 1, "energy_quantity": -1, "distance": -1},
            )
            self.carbon_intensity_technology *= self.distance
        # get transport loss factor
        self.get_transport_loss_factor()
        # get capex of transport technology
        self.get_capex_transport()
        # annualize capex
        self.convert_to_fraction_of_capex()
        # calculate capex of existing capacity
        self.capex_capacity_existing = self.calculate_capex_of_capacities_existing()

    def get_transport_loss_factor(self):
        """Get transport loss factor."""
        # check which transport loss factor is used
        assert not (
            "transport_loss_factor_linear" in self.data_input.attribute_dict
            and "transport_loss_factor_exponential" in self.data_input.attribute_dict
        ), "Only one transport loss factor can be specified."
        if "transport_loss_factor_linear" in self.data_input.attribute_dict:
            self.transport_loss_factor = self.data_input.extract_input_data(
                "transport_loss_factor_linear",
                index_sets=[],
                unit_category={"distance": -1},
            )
            self.transport_loss_factor = self.transport_loss_factor[0] * self.distance
        elif "transport_loss_factor_exponential" in self.data_input.attribute_dict:
            self.transport_loss_factor = self.data_input.extract_input_data(
                "transport_loss_factor_exponential",
                index_sets=[],
                unit_category={"distance": -1},
            )
            self.transport_loss_factor = 1 - np.exp(
                -self.transport_loss_factor[0] * self.distance
            )
            self.config.system.set_transport_technologies_loss_exponential.append(
                self.name
            )
        else:
            raise AttributeError(
                f"The transport technology {self.name} has neither of the attributes: "
                f"transport_loss_factor_linear nor transport_loss_factor_exponential."
            )

    def get_capex_transport(self):
        """Get capex of transport technology."""
        # check if there are separate capex for capacity and distance
        if self.config.system.double_capex_transport:
            # both capex terms must be specified
            self.capex_specific_transport = self.data_input.extract_input_data(
                "capex_specific_transport",
                index_sets=["set_edges", "set_years"],
                unit_category={"money": 1, "energy_quantity": -1, "time": 1},
            )
            self.capex_per_distance_transport = self.data_input.extract_input_data(
                "capex_per_distance_transport",
                index_sets=["set_edges", "set_years"],
                unit_category={"money": 1, "distance": -1},
            )
        else:
            # Only capex_specific is used, capex_per_distance_transport is set to Zero.
            if "capex_per_distance_transport" in self.data_input.attribute_dict:
                self.capex_per_distance_transport = self.data_input.extract_input_data(
                    "capex_per_distance_transport",
                    index_sets=["set_edges", "set_years"],
                    unit_category={
                        "money": 1,
                        "distance": -1,
                        "energy_quantity": -1,
                        "time": 1,
                    },
                )
                self.capex_specific_transport = (
                    self.capex_per_distance_transport * self.distance
                )
            elif "capex_specific_transport" in self.data_input.attribute_dict:
                self.capex_specific_transport = self.data_input.extract_input_data(
                    "capex_specific_transport",
                    index_sets=["set_edges", "set_years"],
                    unit_category={"money": 1, "energy_quantity": -1, "time": 1},
                )
            else:
                raise AttributeError(
                    f"The transport technology {self.name} has neither "
                    f"capex_per_distance_transport nor capex_specific_transport "
                    f"attribute."
                )
            self.capex_per_distance_transport = self.capex_specific_transport * 0.0
        if "opex_specific_fixed_per_distance" in self.data_input.attribute_dict:
            self.opex_specific_fixed_per_distance = self.data_input.extract_input_data(
                "opex_specific_fixed_per_distance",
                index_sets=["set_edges", "set_years"],
                unit_category={
                    "money": 1,
                    "distance": -1,
                    "energy_quantity": -1,
                    "time": 1,
                },
            )
            self.opex_specific_fixed = (
                self.opex_specific_fixed_per_distance * self.distance
            )
        elif "opex_specific_fixed" in self.data_input.attribute_dict:
            self.opex_specific_fixed = self.data_input.extract_input_data(
                "opex_specific_fixed",
                index_sets=["set_edges", "set_years"],
                unit_category={"money": 1, "energy_quantity": -1, "time": 1},
            )
        else:
            raise AttributeError(
                f"The transport technology {self.name} has neither "
                f"opex_specific_fixed_per_distance nor opex_specific_fixed attribute."
            )

    def convert_to_fraction_of_capex(self):
        """Converts total capex to fraction of capex.

        this method converts the total capex to fraction of capex, depending on how
        many hours per year are calculated.
        """
        fraction_year = self.calculate_fraction_of_year()
        self.opex_specific_fixed = self.opex_specific_fixed * fraction_year
        self.capex_specific_transport = self.capex_specific_transport * fraction_year
        self.capex_per_distance_transport = (
            self.capex_per_distance_transport * fraction_year
        )

    def calculate_capex_of_single_capacity(self, capacity, index, **kwargs):
        """This method calculates the capex of a single existing capacity.

        :param capacity: capacity of transport technology
        :param index: index of capacity
        :return: capex of single capacity
        """
        if np.isnan(self.capex_specific_transport[index[0]].iloc[0]) and np.isnan(
            self.capex_per_distance_transport[index[0]].iloc[0]
        ):
            return 0
        elif self.config.system.double_capex_transport and capacity != 0:
            return (
                self.capex_specific_transport[index[0]].iloc[0] * capacity
                + self.capex_per_distance_transport[index[0]].iloc[0]
                * self.distance[index[0]]
            )
        else:
            return self.capex_specific_transport[index[0]].iloc[0] * capacity

    ### --- getter/setter classmethods
    def set_reversed_edge(self, edge, reversed_edge):
        """Maps the reversed edge to an edge.

        :param edge: edge
        :param reversed_edge: reversed edge
        """
        self.dict_reversed_edges[edge] = reversed_edge

    def get_reversed_edge(self, edge):
        """Get the reversed edge corresponding to an edge.

        :param edge: edge
        :return: reversed edge
        """
        return self.dict_reversed_edges[edge]
