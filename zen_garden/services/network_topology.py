from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.attribute_data_loader import AttributeDataLoader
    from zen_garden.topology.model_schema import ModelSchema
    from zen_garden.utils.input_data_checks import InputDataChecks


class NetworkTopology:
    def __init__(
        self,
        model_schema: "ModelSchema",
        attribute_data_loader: "AttributeDataLoader",
        input_data_checks: "InputDataChecks",
        unit_handling: "UnitHandling",
    ):
        self.model_schema = model_schema
        self.attribute_data_loader = attribute_data_loader
        self.input_data_checks = input_data_checks
        self.unit_handling = unit_handling

        self.set_nodes = self._extract_nodes(False)
        self.set_nodes_on_edges = self._calculate_edges_from_nodes()
        self.set_edges = list(self.set_nodes_on_edges.keys())
        self.set_haversine_distances_edges = (
            self._calculate_haversine_distances_from_nodes()
        )

    @property
    def config(self):
        """Return the canonical configuration from the model schema."""
        return self.model_schema.config

    def _extract_nodes(self, extract_coordinates: bool):
        set_nodes_config = self.config.system.set_nodes
        df_nodes_w_coords = self.attribute_data_loader.read_csv_safe("set_nodes")
        if extract_coordinates:
            if len(set_nodes_config) != 0:
                df_nodes_w_coords = df_nodes_w_coords[
                    df_nodes_w_coords["node"].isin(set_nodes_config)
                ]
            return df_nodes_w_coords

        set_nodes_input = df_nodes_w_coords["node"].to_list()
        # if no nodes specified in system, use all nodes
        if len(set_nodes_config) == 0 and not len(set_nodes_input) == 0:
            self.config.system.set_nodes = set_nodes_input
            set_nodes_config = set_nodes_input
        else:
            missing_nodes = list(set(set_nodes_config).difference(set_nodes_input))
            assert len(missing_nodes) == 0, (
                f"The nodes {missing_nodes} were declared in the "
                "config but do not exist in the input file for set_nodes"
            )
        if not isinstance(set_nodes_config, list):
            set_nodes_config = set_nodes_config.to_list()
        set_nodes_config.sort()
        # assert that no transport technology is selected if only
        # one node is given
        assert (
            len(set_nodes_config) > 1
            or len(self.config.system.set_transport_technologies) == 0
        ), (
            f"Only one node is given in the system file. "
            f"Transport technologies are not allowed in this case. "
            f"You selected {self.config.system.set_transport_technologies}"
        )
        return set_nodes_config

    def _extract_edges(self):
        set_edges_input = self.attribute_data_loader.read_csv_safe("set_edges")
        self.input_data_checks.check_single_directed_edges(set_edges_input)
        if set_edges_input is not None:
            set_edges = set_edges_input[
                (set_edges_input["node_from"].isin(self.set_nodes))
                & (set_edges_input["node_to"].isin(self.set_nodes))
            ]
            set_edges = set_edges.set_index("edge")
            return set_edges
        else:
            raise FileNotFoundError("Input file set_edges.csv is missing")

    def _calculate_edges_from_nodes(self):
        """Calculates set_nodes_on_edges from set_nodes.

        :return: set_nodes_on_edges: dict with edges and corresponding nodes
        """
        set_edges_input = self._extract_edges()
        assert isinstance(set_edges_input, pd.DataFrame)
        return {
            edge: set_edges_input.loc[edge, ["node_from", "node_to"]].values.tolist()
            for edge in set_edges_input.index
        }

    def _calculate_haversine_distances_from_nodes(self):
        """Computes the distance (in km) between two nodes.

        The Haversine function is used to compute the distance in kilometers based on
         their lon lat coordinates.

        :return: dict containing all edges along with their distances
        """
        set_haversine_distances_of_edges = {}

        # read coords file
        df_coords_input = self._extract_nodes(extract_coordinates=True)
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
            lon1, lat1 = coords1["lon"].squeeze(), coords1["lat"].squeeze()
            lon2, lat2 = coords2["lon"].squeeze(), coords2["lat"].squeeze()
            assert isinstance(lon1, float) and isinstance(lat1, float)
            assert isinstance(lon2, float) and isinstance(lat2, float)
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = radius * c
            set_haversine_distances_of_edges[edge] = distance
        multiplier = self.unit_handling.get_unit_multiplier(
            "km", attribute_name="distance"
        )
        return {
            key: value * multiplier
            for key, value in set_haversine_distances_of_edges.items()
        }

    def calculate_connected_edges(self, node: str, direction: Literal["in", "out"]):
        """Calculates connected edges going in or going out.

        :param node: current node, connected by edges
        :param direction: direction of edges, either in or out.
            In: node = end node,
            out: node = start node
        :return: _set_connected_edges: list of connected edges
        """
        assert direction in ["in", "out"], "direction must be either 'in' or 'out'"
        return [
            edge
            for edge, nodes in self.set_nodes_on_edges.items()
            if nodes[0 if direction == "out" else 1] == node
        ]
