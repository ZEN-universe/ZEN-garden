"""Functions to extract the input data from the provided input files."""

import copy
import logging
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from zen_garden.elements.element import Element
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.attribute_data_loader import AttributeDataLoader
    from zen_garden.services.network_topology import NetworkTopology
    from zen_garden.services.scenario_dict import ScenarioDict
    from zen_garden.topology.model_schema import ModelSchema
    from zen_garden.types import YearSpecificTs
    from zen_garden.utils.input_data_checks import InputDataChecks

logger = logging.getLogger(__name__)

TIME_STEP_TYPES = [
    "set_hours_all_years",
    "set_hours",
    "set_years",
    "set_years_entire_horizon",
]
"""List of valid time step types."""

PARAMETER_CHANGE_LOG = {
    "min_full_load_hours_fraction": {
        "default_value": 0,  # only 0, 1, or 'inf' are allowed
        "unit": "min_load",
    },
    "capacity_lower_limit": {
        "default_value": 0,  # only 0, 1, or 'inf' are allowed
        "unit": "capacity_limit",
    },
    "capacity_lower_limit_energy": {
        "default_value": 0,  # only 0, 1, or 'inf' are allowed
        "unit": "capacity_limit_energy",
    },
}
"""Dictionary to log changes in parameter values.

The keys are the new parameter names. The values are dictionaries with the default value
and the unit of the new parameter. The unit is taken from an existing parameter
with the same unit.

Example
-------

.. code-block:: python

    "new_parameter_name": {
        "default_value": 0, # only 0, 1, or 'inf' are allowed
        "unit": "existing_parameter_name_with_same_unit"
    }
"""


class ElementDataLoader:
    """Class to extract input data."""

    def __init__(
        self,
        element: "Element",
        model_schema: "ModelSchema",
        network_topology: "NetworkTopology",
        unit_handling: "UnitHandling",
        scenario_dict: "ScenarioDict",
        input_data_checks: "InputDataChecks",
        year_specific_ts: "YearSpecificTs",
        folder_path: Path,
        attribute_data_loader: "AttributeDataLoader",
    ):
        """Data input object to extract input data.

        :param element: element for which data is extracted
        :param system: dictionary defining the system
        :param analysis: dictionary defining the analysis framework
        :param solver: dictionary defining the solver
        :param model_schema: global model schema
        :param unit_handling: instance of class <UnitHandling> to convert units
        """
        self.element = element
        self.model_schema = model_schema
        self.network_topology = network_topology
        self.unit_handling = unit_handling
        self.scenario_dict = scenario_dict
        self.input_data_checks = input_data_checks
        self.year_specific_ts = year_specific_ts
        # extract folder path
        self.folder_path = folder_path
        self.attribute_data_loader = attribute_data_loader
        # get names of indices
        self.index_names = self.model_schema.config.analysis.header_data_inputs
        # load attributes file
        self.attribute_dict = self.attribute_data_loader.load_attribute_file()

    def extract_input_data(self, file_name, index_sets, unit_category, subelement=None):
        """Loads and restructures input data for the current scenario.

        Defaults and units are taken from the attributes file, then values from the
        selected parameter CSV and any scenario-specific CSV are applied, converted
        to base units, and scaled by the scenario factor.

        Args:
            file_name: name of selected file.
            index_sets: ordered configured model sets defining the output index,
                e.g., location, year, or hour sets. An empty
                list denotes a scalar parameter.
            unit_category: dict defining the dimensions of the parameter's unit
            subelement: string specifying dependent element

        Returns:
            numeric values as a pandas Series with a MultiIndex in ``index_sets``
            order, or index ``[0]`` for a scalar parameter.
        """
        # generic time steps
        yearly_variation = False
        if "set_hours" in index_sets:
            yearly_variation = True
            self.extract_yearly_variation(file_name, index_sets)

        # if existing capacities and existing capacities not used
        if (
            file_name in ["capacity_existing", "capacity_existing_energy"]
        ) and not self.model_schema.config.system.use_capacities_existing:
            df_output, *_ = self.create_default_output(
                index_sets,
                unit_category,
                file_name=file_name,
                manual_default_value=0,
            )
            return df_output
        # use distances computed with node coordinates as default values
        elif file_name == "distance":
            df_output, default_value, index_name_list = self.create_default_output(
                index_sets,
                unit_category,
                file_name=file_name,
                manual_default_value=self.network_topology.set_haversine_distances_edges,
            )
        else:
            df_output, default_value, index_name_list = self.create_default_output(
                index_sets,
                unit_category,
                file_name=file_name,
                subelement=subelement,
            )
        # read input file
        f_name, scenario_factor = self.scenario_dict.get_param_file(
            self.element.name, file_name
        )
        df_input = self.attribute_data_loader.read_csv(f_name)
        if f_name != file_name and yearly_variation and df_input is None:
            logger.info(
                f"{f_name} for current scenario is missing from "
                f"{self.folder_path}. {file_name} is used as input file"
            )
            df_input = self.attribute_data_loader.read_csv(file_name)

        assert df_input is not None or default_value is not None, (
            f"input file for attribute {file_name} could not be imported and no "
            "default value is given."
        )
        if df_input is not None and not df_input.empty:
            # get subelement dataframe
            if subelement is not None and subelement in df_input.columns:
                cols = df_input.columns.intersection(index_name_list + [subelement])
                df_input = df_input[cols]
            # fill output dataframe
            df_output = self._extract_general_input_data(
                df_input,
                df_output,
                file_name,
                index_name_list,
                default_value,
                index_sets,
            )
            # overwrite parts of the output file with scenario specific data
            part_file_name = self.scenario_dict.get_param_part_file(
                self.element.name, file_name
            )
            if part_file_name is not None:
                df_input_part = self.attribute_data_loader.read_csv(part_file_name)
                if df_input_part is None:
                    logger.info(
                        f"{part_file_name} for current scenario is missing "
                        f"from {self.folder_path}. The base case is used as input file"
                    )
                else:
                    df_output = self._extract_general_input_data(
                        df_input_part,
                        df_output,
                        file_name,
                        index_name_list,
                        default_value,
                        index_sets,
                    )
        # copy output data as otherwise overwritten
        df_output_generic = df_output.copy()
        if "set_hours" in index_sets:
            self._extract_year_specific_ts(
                file_name,
                index_name_list,
                index_sets,
                subelement,
                default_value,
                df_output_generic=df_output,
            )
        # finally apply the scenario_factor and return df_output
        return df_output_generic * scenario_factor

    def _extract_general_input_data(
        self,
        df_input,
        df_output,
        file_name,
        index_name_list,
        default_value,
        index_sets: list[str],
    ):
        """Fills df_output with data from df_input.

        :param df_input: raw input dataframe
        :param df_output: empty output dataframe, only filled with default_value
        :param file_name: name of selected file
        :param index_name_list: list of name of indices
        :param default_value: default for dataframe
        :param index_sets: index sets of attribute
        :return: df_output: filled output dataframe
        """
        df_output_copy = copy.deepcopy(df_output)
        df_input = self._convert_real_to_generic_time_indices(
            df_input, index_sets, file_name, index_name_list
        )

        assert df_input.columns is not None, f"Input file '{file_name}' has no columns"
        # set index by index_name_list
        missing_index = list(
            set(index_name_list)
            - set(index_name_list).intersection(set(df_input.columns))
        )
        assert len(missing_index) <= 1, (
            f"More than one the requested index sets ({missing_index}) are "
            "missing from input file for {file_name}"
        )

        # no indices missing
        if len(missing_index) == 0:
            df_input = ElementDataLoader.extract_from_input_without_missing_index(
                df_input, index_name_list, file_name
            )
        else:
            missing_index = missing_index[0]
            # check if special case of existing Technology
            if "technology_existing" in missing_index:
                df_output = (
                    ElementDataLoader.extract_from_input_for_capacities_existing(
                    df_input, df_output_copy, index_name_list, file_name, missing_index
                    )
                )
                if isinstance(default_value, dict):
                    df_output_copy = df_output_copy * default_value["multiplier"]
                return df_output_copy
            # index missing
            else:
                df_input = ElementDataLoader.extract_from_input_with_missing_index(
                    df_input,
                    df_output_copy,
                    copy.deepcopy(index_name_list),
                    file_name,
                    missing_index,
                )

        # check for duplicate indices
        df_input = self.input_data_checks.check_duplicate_indices(
            df_input, file_name, self.folder_path
        )

        # apply multiplier to input data
        df_input = df_input * default_value["multiplier"]
        # delete nans
        df_input = df_input.dropna()

        # get common index of df_output_copy and df_input
        if not isinstance(df_input.index, pd.MultiIndex) and isinstance(
            df_output_copy.index, pd.MultiIndex
        ):
            index_list = df_input.index.to_list()
            if len(index_list) == 1:
                index_multi_index = pd.MultiIndex.from_tuples(
                    [(index_list[0],)], names=[df_input.index.name]
                )
            else:
                index_multi_index = pd.MultiIndex.from_product(
                    [index_list], names=[df_input.index.name]
                )
            df_input = pd.Series(
                index=index_multi_index, data=df_input.to_list(), dtype=float
            )
        common_index = df_output_copy.index.intersection(df_input.index)
        assert default_value is not None or len(common_index) == len(
            df_output_copy.index
        ), (
            f"Input for {file_name} does not provide entire dataset and no "
            "default given in the attributes file"
        )
        df_output_copy.loc[common_index] = df_input.loc[common_index]
        return df_output_copy

    def get_attribute_dict(self, attribute_name: str) -> tuple[dict, float]:
        """Get attribute dict and factor for attribute.

        :param attribute_name: name of selected attribute
        :return: attribute_dict: attribute dict
        :return: factor: factor for attribute
        """
        filename, factor = self.scenario_dict.get_default(
            self.element.name, attribute_name
        )

        if filename == "attributes":
            return self.attribute_dict, factor

        attribute_dict = self.attribute_data_loader.load_attribute_file(filename)
        return attribute_dict, factor

    def extract_attribute(
        self, attribute_name, unit_category, return_unit=False, subelement=None
    ):
        """Reads input data and restructures the dataframe to return
        (multi)indexed dict.

        :param attribute_name: name of selected attribute
        :param unit_category: dict defining the dimensions of the parameter's unit
        :param return_unit: only returns unit
        :param subelement: dependent element for which data is extracted
        :return: attribute value and multiplier
        :return: unit of attribute
        """
        attribute_dict, factor = self.get_attribute_dict(attribute_name)
        attribute_value, attribute_unit = self._extract_attribute_value(
            attribute_name, attribute_dict
        )
        if subelement is not None:
            assert (
                subelement in attribute_value.keys()
            ), f"{subelement} not in {attribute_name} of {self.element.name}"
            attribute_unit = attribute_value[subelement]["unit"]
            attribute_value = attribute_value[subelement]["default_value"]
        if return_unit:
            return attribute_unit
        if attribute_unit is None:
            return attribute_value
        if attribute_value is not None:
            multiplier, attribute_unit_in_base_units = (
                self.unit_handling.convert_unit_into_base_units(
                    attribute_unit,
                    get_multiplier=True,
                    attribute_name=attribute_name,
                    path=self.folder_path,
                )
            )
            # don't convert unit of conversion factor to base units since e.g.
            # kWh/kWh would become 1 (however, conversion factors' unit consistency
            # must be checked with the corresponding carriers)
            if attribute_name == "conversion_factor":
                if attribute_name not in self.element.units:
                    self.element.units[attribute_name] = {}
                self.element.units[attribute_name][subelement] = {
                    "unit_category": unit_category,
                    "unit": attribute_unit,
                }
            elif attribute_name == "retrofit_flow_coupling_factor":
                self.element.units[attribute_name] = {
                    str(self.element.reference_carrier[0]): {
                        "unit_category": unit_category,
                        "unit": attribute_unit,
                    }
                }
            # don't try to save input-/output carrier if they don't exist for a
            # conversion technology
            elif not (
                pd.isna(attribute_value)
                and attribute_name in ["input_carrier", "output_carrier"]
            ):
                self.element.units[attribute_name] = {
                    "unit_category": unit_category,
                    "unit_in_base_units": attribute_unit_in_base_units,
                }
            try:
                attribute = {
                    "value": float(attribute_value) * multiplier * factor,
                    "multiplier": multiplier,
                }
                return attribute
            except ValueError:
                if factor != 1:
                    logger.warning(
                        f"WARNING: Attribute {attribute_name} of "
                        f"{self.element.name} is not a number "
                        f"but has custom factor {factor}, factor will be "
                        f"ignored..."
                    )
                attribute = attribute_value
                return attribute
        else:
            return None

    def _extract_attribute_value(self, attribute_name, attribute_dict):
        """Reads attribute value from dict.

        :param attribute_name: name of selected attribute
        :param attribute_dict: name of selected attribute
        :return: attribute value, attribute unit
        """
        if attribute_name not in attribute_dict:
            # The attribute is not found because of an update
            if attribute_name in PARAMETER_CHANGE_LOG:
                # CASE 1: There is a new attribute
                if isinstance(PARAMETER_CHANGE_LOG[attribute_name], dict):
                    missing_attribute = PARAMETER_CHANGE_LOG[attribute_name]

                    if missing_attribute["default_value"] not in [0, 1, "inf"]:
                        raise AttributeError(
                            f"Default value of attribute {attribute_name} must "
                            f"be 0 , 1, or 'inf' but is "
                            f"{missing_attribute['default_value']}"
                        )

                    attribute_dict[attribute_name] = {
                        "default_value": missing_attribute["default_value"],
                        "unit": attribute_dict[missing_attribute["unit"]]["unit"],
                    }

                    warnings.warn(
                        f"\nAttribute {attribute_name} is not yet included in "
                        f"your model. Automatic assign default_value:"
                        f"{attribute_dict[attribute_name]['default_value']}, "
                        f"unit: {attribute_dict[attribute_name]['unit']}\n",
                        DeprecationWarning,
                        stacklevel=2,
                    )

                # CASE 2: The attribute has a new name
                else:
                    old_name = PARAMETER_CHANGE_LOG[attribute_name]
                    attribute_dict[attribute_name] = attribute_dict.pop(old_name)

                    warnings.warn(
                        f"Attribute {old_name} is now called {attribute_name}",
                        DeprecationWarning,
                        stacklevel=2,
                    )

            else:
                raise AttributeError(
                    f"Attribute {attribute_name} does not exist in input data "
                    f"of {self.element.name}"
                )
        try:
            attribute_value = float(attribute_dict[attribute_name]["default_value"])
            attribute_unit = attribute_dict[attribute_name]["unit"]
        # for string attributes
        except ValueError:
            attribute_value = attribute_dict[attribute_name]["default_value"]
            attribute_unit = attribute_dict[attribute_name]["unit"]
        # for list of attributes
        except (TypeError, KeyError):
            if "default_value" in attribute_dict[attribute_name]:
                attribute_value = attribute_dict[attribute_name]["default_value"]
            else:
                attribute_value = attribute_dict[attribute_name]
            attribute_unit = None
        return attribute_value, attribute_unit

    def _extract_year_specific_ts(
        self,
        file_name,
        index_name_list,
        index_sets: list[str],
        subelement,
        default_value,
        df_output_generic,
    ):
        """Reads and saves the year specific time series data. The year specific
        time series are saved in the dictionary self.year_specific_ts.

        :param file_name: name of selected file
        :param index_name_list: list of name of indices
        :param default_value: default for dataframe
        :param index_sets: index sets of attribute
        :param subelement: string specifying dependent element
        :param df_output_generic: original/generic time series data (base case)
        """
        # years of optimization model
        years = self.model_schema.set_time_steps_years
        # files to check
        file_names = os.listdir(self.folder_path)
        for file in file_names:
            for i, year in enumerate(years):
                filename = file_name + "_" + str(year)
                if filename not in file:
                    continue

                # read input data
                f_name, scenario_factor = self.scenario_dict.get_param_file(
                    self.element.name, filename
                )
                df_input = self.attribute_data_loader.read_csv(f_name)
                if df_input is not None and not df_input.empty:
                    # get subelement dataframe
                    if subelement is not None and subelement in df_input.columns:
                        cols = df_input.columns.intersection(
                            index_name_list + [subelement]
                        )
                        df_input = df_input[cols]
                    df_output_specific = self._extract_general_input_data(
                        df_input,
                        df_output_generic,
                        file_name,
                        index_name_list,
                        default_value,
                        index_sets,
                    )
                if i not in self.year_specific_ts:
                    self.year_specific_ts[i] = {}
                self.year_specific_ts[i][(self.element.name, file_name)] = (
                    df_output_specific * scenario_factor
                )

    def extract_yearly_variation(self, file_name, index_sets):
        """Reads the yearly variation of a time dependent quantity.

        Args:
            file_name: name of selected file.
            index_sets: index sets of attribute. Creates (multi)index.
                Corresponds to order in pe.Set/pe.Param
        """
        # remove intra-yearly time steps from index set and add inter-yearly
        # time steps
        index_sets = copy.deepcopy(index_sets)
        index_sets.remove("set_hours")
        index_sets.append("set_years")
        # add Yearly_variation to file_name
        file_name += "_yearly_variation"
        # read input data
        f_name, scenario_factor = self.scenario_dict.get_param_file(
            self.element.name, file_name
        )
        df_input = self.attribute_data_loader.read_csv(f_name)
        if f_name != file_name and df_input is None:
            logger.info(
                f"{f_name} is missing from {self.folder_path}. {file_name} is "
                "used as input file"
            )
            df_input = self.attribute_data_loader.read_csv(file_name)
        if df_input is not None:
            df_output, default_value, index_name_list = self.create_default_output(
                index_sets,
                unit_category=None,
                file_name=file_name,
                manual_default_value=1,
            )
            # set yearly variation attribute to df_output
            name_yearly_variation = file_name
            df_output = self._extract_general_input_data(
                df_input,
                df_output,
                file_name,
                index_name_list,
                default_value,
                index_sets,
            )
            # apply the scenario_factor
            df_output = df_output * scenario_factor
            setattr(self, name_yearly_variation, df_output)

    def extract_carriers(self, carrier_type):
        """Reads input data and extracts conversion carriers.

        Returns:
            list: list with input, output or reference carriers of technology
        """
        assert carrier_type in [
            "input_carrier",
            "output_carrier",
            "reference_carrier",
            "retrofit_reference_carrier",
        ], f"invalid carrier_type {carrier_type} for {self.element.name}. "
        carrier_list = self.extract_attribute(carrier_type, unit_category=None)
        assert carrier_type != "reference_carrier" or len(carrier_list) == 1, (
            f"Reference_carrier must be a single carrier, but {carrier_list} "
            f"are given for {self.element.name}"
        )
        if carrier_list == [""]:
            carrier_list = []
        return carrier_list

    def extract_retrofit_base_technology(self):
        """Extract base technologies for retrofitting technology.

        :return: return base technology of retrofit technology
        """
        attribute_name = "retrofit_flow_coupling_factor"
        technology_type = "base_technology"
        attribute_dict, _ = self.get_attribute_dict(attribute_name)
        base_technology = attribute_dict[attribute_name][technology_type]
        if isinstance(base_technology, str):
            base_technology = base_technology.strip().split(" ")
        assert len(base_technology) == 1, (
            f"retrofit base technology must be a single technology, "
            f"but {base_technology} are given for {self.element.name}"
        )
        return base_technology

    def extract_set_technologies_existing(self, storage_energy=False):
        """Reads input data and creates setExistingCapacity for each technology.

        Args:
            storage_energy: boolean if existing energy capacity of storage
                technology (instead of power)

        Returns:
            set_technologies_existing: return set existing technologies
        """
        # TODO merge changes in extract input data and optimization setup
        set_technologies_existing = np.array([0])
        if self.model_schema.config.system.use_capacities_existing:
            if storage_energy:
                _energy_string = "_energy"
            else:
                _energy_string = ""

            # here we ignore the factor
            f_name, _ = self.scenario_dict.get_param_file(
                self.element.name, f"capacity_existing{_energy_string}"
            )
            df_input = self.attribute_data_loader.read_csv(f_name)
            if df_input is None:
                return [0]
            if (
                self.element.name
                in self.model_schema.config.system.set_transport_technologies
            ):
                location = "edge"
            else:
                location = "node"
            _max_node_count = df_input[location].value_counts().max()
            if _max_node_count is not np.nan:
                set_technologies_existing = np.arange(0, _max_node_count)

        return set_technologies_existing

    def extract_lifetime_existing(self, file_name: str, index_sets: list[str]):
        """Reads input data and restructures the dataframe to return
        (multi)indexed dict.

        Args:
            file_name:  name of selected file
            index_sets: index sets of attribute. Creates (multi)index.
                Corresponds to order in pe.Set/pe.Param
        Returns:
            df_output: return existing capacity and existing lifetime
        """
        index_list, index_name_list = self.construct_index_list(index_sets)
        multi_idx = pd.MultiIndex.from_product(index_list, names=index_name_list)
        df_output = pd.Series(index=multi_idx, data=0, dtype=int)
        # if no existing capacities
        if not self.model_schema.config.system.use_capacities_existing:
            return df_output
        f_name, scenario_factor = self.scenario_dict.get_param_file(
            self.element.name, file_name
        )
        if f"{f_name}.csv" in os.listdir(self.folder_path):
            df_input = self.attribute_data_loader.read_csv(f_name)
            # fill output dataframe
            df_output = self._extract_general_input_data(
                df_input,
                df_output,
                "year_construction",
                index_name_list,
                default_value=0,
                index_sets=index_sets,
            )
            # get reference year
            reference_year = self.model_schema.config.system.reference_year
            if not hasattr(self.element, "lifetime"):
                raise TypeError("Construction years require a technology element")
            # calculate remaining lifetime
            df_output[df_output > 0] = (
                -reference_year + df_output[df_output > 0] + self.element.lifetime[0]
            )
        # apply scenario factor
        return df_output * scenario_factor

    def create_default_output(
        self,
        index_sets,
        unit_category,
        file_name=None,
        manual_default_value=None,
        subelement=None,
    ):
        """Creates default output dataframe.

        Args:
            index_sets: index sets of attribute. Creates (multi)index.
                Corresponds to order in pe.Set/pe.Param
            unit_category: dict defining the dimensions of the parameter's unit
            file_name: name of selected file.
            manual_default_value: if given, use manual_default_value instead
                of searching for a default value in the attributes file
            subelement: dependent element for which data is extracted
        """
        # select index
        index_list, index_name_list = self.construct_index_list(index_sets)
        # create pd.MultiIndex and select data
        if index_sets:
            index_multi_index = pd.MultiIndex.from_product(
                index_list, names=index_name_list
            )
        else:
            index_multi_index = pd.Index([0])
        # use distances computed with node coordinates as default values
        if file_name == "distance":
            default_name = file_name
            default_value = self.extract_attribute(default_name, unit_category)
            default_value["value"] = manual_default_value
        elif manual_default_value:
            default_value = {"value": manual_default_value, "multiplier": 1}
            default_name = None
        else:
            default_name = file_name
            default_value = self.extract_attribute(
                default_name, unit_category, subelement=subelement
            )

        # create output Series filled with default value
        if default_value is None:
            df_output = pd.Series(index=index_multi_index, dtype=float)
        # use distances computed with node coordinates as default values
        elif file_name == "distance":
            df_output = pd.Series(index=index_multi_index, dtype=float)
            for key, value in default_value["value"].items():
                df_output[key] = value
        else:
            df_output = pd.Series(
                index=index_multi_index, data=default_value["value"], dtype=float
            )
        return df_output, default_value, index_name_list

    def construct_index_list(self, index_sets: list[str]) -> tuple[list[list], list]:
        """Constructs index list from index sets and returns list of indices and
        list of index names.

        Args:
            index_sets: index sets of attribute. Creates (multi)index.
                Corresponds to order in pe.Set/pe.Param
        Returns:
            index_list: list of indices
            index_name_list: list of name of indices
        """
        index_list = []
        index_name_list = []
        # add rest of indices
        for index in index_sets:
            index_name_list.append(self.index_names[index])
            if index in TIME_STEP_TYPES:
                index_list.append(getattr(self.model_schema, index))
            elif index == "set_technologies_existing" and hasattr(
                self.element, "set_technologies_existing"
            ):
                index_list.append(self.element.set_technologies_existing)  # type: ignore[attr-defined]
            elif index in type(self.model_schema.config.system).model_fields:
                index_list.append(self.model_schema.config.system[index])
            elif hasattr(self.model_schema, index):
                index_list.append(getattr(self.model_schema, index))
            elif hasattr(self.network_topology, index):
                index_list.append(getattr(self.network_topology, index))
            else:
                raise AttributeError(f"Index '{index}' cannot be found.")
        return index_list, index_name_list

    def _convert_real_to_generic_time_indices(
        self, df_input, index_sets, file_name, index_name_list
    ):
        """Convert yearly time indices to generic time indices.

        :param df_input: raw input dataframe
        :param index_sets: index sets of attribute
        :param file_name: name of selected file
        :param index_name_list: list of name of indices
        :return: df_input: input dataframe with generic time indices
        """
        yearly_ts = next(
            (s for s in index_sets if s in {"set_years", "set_years_entire_horizon"}),
            None,
        )
        # check if input data is time-dependent and has yearly time steps
        idx_name_year = self.index_names["set_years"]
        if yearly_ts is not None:
            # check if temporal header of input data is still given as 'time'
            # instead of 'year'
            if "time" in df_input.axes[1]:
                warnings.warn(
                    f"The column header 'time' (used in {file_name}) will not "
                    "be supported for input data with yearly time steps any "
                    "longer! Use the header 'year' instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                df_input = df_input.rename(
                    {self.index_names["set_hours"]: self.index_names["set_years"]},
                    axis=1,
                )
            # does not contain annual index
            elif idx_name_year not in df_input.axes[1]:
                idx_name_list = [idx for idx in index_name_list if idx != idx_name_year]
                # no other index called, return original time series
                if not idx_name_list:
                    return df_input
                df_input = df_input.set_index(idx_name_list)
                df_input = df_input.rename(
                    columns={
                        col: int(col) for col in df_input.columns if col.isnumeric()
                    }
                )
                requested_index_values = set(getattr(self.model_schema, yearly_ts))
                requested_index_values_years = set(
                    self.model_schema.set_time_steps_years
                )
                requested_index_values_in_columns = requested_index_values.intersection(
                    df_input.columns
                )
                requested_index_values_years_in_columns = (
                    requested_index_values_years.intersection(df_input.columns)
                )
                if (
                    not requested_index_values_in_columns
                    and not requested_index_values_years_in_columns
                ):
                    return df_input.reset_index()
                elif requested_index_values_in_columns:
                    requested_index_values = requested_index_values_in_columns
                else:
                    requested_index_values = requested_index_values_years_in_columns
                df_input.columns = df_input.columns.set_names(idx_name_year)
                df_input = df_input[list(requested_index_values)].stack()
                df_input = df_input.reset_index()
            # check if input data is still given with generic time indices
            temporal_header = self.index_names["set_years"]
            if (
                max(df_input.loc[:, temporal_header])
                < self.model_schema.config.analysis.earliest_year_of_data
            ):
                warnings.warn(
                    f"Generic time indices (used in {file_name}) will not be "
                    "supported for input data with yearly time steps any "
                    "longer! Use the corresponding years (e.g. 2022,2023,...) "
                    "as time indices instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return df_input
            # assert that correct temporal index_set to get corresponding
            # index_name is given (i.e. set_years for input data
            # with yearly time steps)(otherwise _extract_general_input_data()
            # will find a missing_index)
            assert temporal_header in index_name_list, (
                "Input data with yearly time steps and therefore the temporal "
                "header 'year' needs to be extracted with "
                "index_sets=['set_years'] instead of "
                "index_sets=['set_hours']"
            )
            # set index
            index_names_column = df_input.columns.intersection(
                index_name_list
            ).to_list()
            df_input = df_input.set_index(index_names_column)
            if df_input.index.nlevels == 1:
                combined_index = df_input.index.union(
                    self.model_schema.set_time_steps_years
                )
                is_single_index = True
            else:
                index_list = []
                for index_name in index_names_column:
                    if index_name == temporal_header:
                        index_list.append(
                            df_input.index.get_level_values(index_name)
                            .unique()
                            .union(self.model_schema.set_time_steps_years)
                        )
                    else:
                        index_list.append(
                            df_input.index.get_level_values(index_name).unique()
                        )
                combined_index = pd.MultiIndex.from_product(
                    index_list, names=index_names_column
                ).sort_values()
                is_single_index = False
            df_input_temp = pd.DataFrame(index=combined_index, columns=df_input.columns)
            common_index = df_input.index.intersection(combined_index)
            df_input_temp.loc[common_index] = df_input.loc[common_index]
            # df_input_temp.loc[df_input.index] = df_input
            df_input = df_input_temp.astype(float)
            # interpolate missing data
            file_names_int_off = []
            if self.model_schema.parameters_interpolation_off is not None:
                file_names_int_off = self.model_schema.parameters_interpolation_off[
                    "parameter_name"
                ]
            if file_name not in file_names_int_off:
                parameters = df_input.axes[1]
                for param in parameters:
                    if param not in index_names_column and df_input[param].isna().any():
                        if is_single_index:
                            df_input[param] = (
                                df_input[param]
                                .astype(float)
                                .interpolate(method="index")
                            )
                        else:
                            df_input_temp = df_input[param].unstack(
                                df_input.index.names.difference([temporal_header])
                            )
                            df_input[param] = (
                                df_input_temp.interpolate(method="index", axis=0)
                                .stack()
                                .reorder_levels(df_input.index.names)
                            )
            else:
                logger.info(
                    f"Parameter {file_name} data won't be interpolated to "
                    "cover years without given values"
                )
            df_input = df_input.reset_index()
            # remove data of years that won't be simulated
            df_input = df_input[
                df_input[temporal_header].isin(self.model_schema.set_time_steps_years)
            ]
            # convert yearly time indices to generic ones
            year2step = {
                year: step
                for year, step in zip(
                    self.model_schema.set_time_steps_years,
                    getattr(self.model_schema, yearly_ts),
                    strict=False,
                )
            }
            df_input[temporal_header] = df_input[temporal_header].apply(
                lambda year: year2step[year]
            )
        return df_input

    @staticmethod
    def extract_from_input_without_missing_index(df_input, index_name_list, file_name):
        """Extracts the demanded values from Input dataframe and
        reformulates dataframe.

        :param df_input: raw input dataframe
        :param index_name_list: list of name of indices
        :param file_name: name of selected file
        :return: df_input: reformulated input dataframe
        """
        if index_name_list:
            df_input = df_input.set_index(index_name_list)
        assert len(df_input.columns) == 1, (
            f"Input file for {file_name} has more than one value "
            "column: {df_input.columns.to_list()}"
        )
        df_input = df_input.squeeze(axis=1)
        return df_input

    @staticmethod
    def extract_from_input_with_missing_index(
        df_input, df_output, index_name_list, file_name, missing_index
    ):
        """Extracts the demanded values from Input dataframe and reformulates
        dataframe if the index is missing. Either, the missing index is
        the column of df_input, or it is actually missing in df_input.
        Then, the values in df_input are extended to all missing index
        values.

        Args:
            df_input: raw input dataframe
            df_output: default output dataframe
            index_name_list: list of name of indices
            file_name: name of selected file
            missing_index: missing index in df_input

        Returns:
            pandas.DataFrame: reformulated input dataframe
        """
        index_name_list.remove(missing_index)
        if not index_name_list:
            # assert that single value
            assert df_input.size == 1, (
                f"Cannot establish unique values for file {file_name} because "
                "of too many columns or not overlapping index"
            )
            val_input = df_input.squeeze()
            df_output[:] = val_input
            df_input = df_output.copy()
            return df_input
        df_input = df_input.set_index(index_name_list)
        # missing index values
        requested_index_values = set(df_output.index.get_level_values(missing_index))
        # the missing index is the columns of df_input
        requested_index_values_in_columns = requested_index_values.intersection(
            df_input.columns
        )
        if requested_index_values_in_columns:
            requested_index_values = requested_index_values_in_columns
            df_input.columns = df_input.columns.set_names(missing_index)
            df_input = df_input[list(requested_index_values)].stack()
            df_input = df_input.reorder_levels(df_output.index.names)
        # the missing index does not appear in df_input
        # the values in df_input are extended to all missing index values
        else:
            df_input_index_temp = pd.MultiIndex.from_product(
                [df_input.index, requested_index_values],
                names=df_input.index.names + [missing_index],
            )
            df_input_temp = pd.Series(index=df_input_index_temp, dtype=float)
            if isinstance(df_input, pd.Series):
                df_input = df_input.to_frame()
            if df_input.shape[1] == 1:
                df_input = df_input.loc[
                    df_input_index_temp.get_level_values(df_input.index.names[0])
                ].squeeze(axis=1)
            else:
                assert df_input_temp.index.names[-1] != "time", (
                    "Only works if columns contain time index and not for "
                    "{df_input_temp.index.names[-1]}"
                )
                df_input = df_input_temp.to_frame().apply(
                    lambda row: df_input.loc[row.name[0:-1], str(row.name[-1])], axis=1
                )
            df_input.index = df_input_temp.index
            df_input = df_input.reorder_levels(order=df_output.index.names)
            if isinstance(df_input, pd.DataFrame):
                df_input = df_input.squeeze(axis=1)
        return df_input

    @staticmethod
    def extract_from_input_for_capacities_existing(
        df_input, df_output, index_name_list, column, missing_index
    ):
        """Extracts the demanded values from input dataframe if extracting
        existing capacities.

        :param df_input: raw input dataframe
        :param df_output: default output dataframe
        :param index_name_list: list of name of indices
        :param column: select specific column
        :param missing_index: missing index in df_input
        :return: df_output: filled output dataframe
        """
        index_name_list.remove(missing_index)
        df_input = df_input.set_index(index_name_list)
        set_location = df_input.index.unique()
        for location in set_location:
            if location in df_output.index.get_level_values(index_name_list[0]):
                values = df_input[column].loc[location].tolist()
                if isinstance(values, int):
                    index = [0]
                    is_float = False
                    int_check = True
                elif isinstance(values, float):
                    index = [0]
                    is_float = True
                    int_check = values.is_integer()
                else:
                    index = list(range(len(values)))
                    is_float = any(isinstance(v, float) for v in values)
                    int_check = all([float(v).is_integer() for v in values])
                # check that correct dtype of values
                if df_output.dtype == int and is_float:
                    if int_check:
                        if isinstance(values, list):
                            values = [int(v) for v in values]
                        else:
                            values = int(values)
                    else:
                        raise ValueError(
                            f"Values in {column} are not integers, but should be"
                        )
                df_output.loc[location, index] = values
        return df_output
