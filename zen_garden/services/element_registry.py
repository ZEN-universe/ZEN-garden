import copy
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd

from zen_garden.model import ELEMENT_TYPE_CLASSES, Context
from zen_garden.model.config import Config
from zen_garden.model.element import Element
from zen_garden.model.energy_system import EnergySystem
from zen_garden.utils.input_data_checks import InputDataChecks

if TYPE_CHECKING:
    from zen_garden.preprocess.unit_handling import UnitHandling

logger = logging.getLogger(__name__)


class ElementRegistry:
    def __init__(
        self,
        config: Config,
        context: Context,
        energy_system: EnergySystem,
        input_data_checks: InputDataChecks,
        unit_handling: "UnitHandling",
    ):
        self.config = config
        self.context = context
        self.energy_system = energy_system
        self.input_data_checks = input_data_checks
        self.unit_handling = unit_handling

    def add_elements(self):
        """Set up the parameters, variables and constraints of the carriers."""
        logger.info("\n--- Add elements to model--- \n")
        for element_id in ELEMENT_TYPE_CLASSES.keys():
            element_class = ELEMENT_TYPE_CLASSES[element_id]
            element_name = element_class.label
            element_set = self.config.system[element_name]

            # before adding the carriers, get set_carriers
            # check if carrier data exists
            if element_name == "set_carriers":
                # TODO: Eliminate this hidden dependency on ConversionTechnology,
                # which modifies set_carriers in EnergySystem
                element_set: list[str] = self.energy_system.set_carriers
                self._check_existing_carrier_data(element_set)

            # check if element_set has a subset and remove subset from element_set
            element_subset: list[str] = []
            if element_name in self.config.analysis.subsets.keys():
                if isinstance(self.config.analysis.subsets[element_name], list):
                    subset_names = self.config.analysis.subsets[element_name]
                elif isinstance(self.config.analysis.subsets[element_name], dict):
                    subset_names = self.config.analysis.subsets[element_name].keys()
                else:
                    raise ValueError(
                        f"Subset {element_name} has to be either a list or a dict"
                    )
                element_subset = [
                    item
                    for subset in subset_names
                    for item in self.config.system[subset]
                ]
            else:
                stack = [
                    _dict
                    for _dict in copy.deepcopy(self.config.analysis.subsets).values()
                    if isinstance(_dict, dict)
                ]
                while stack:  # check if element_set is a subset of a subset
                    cur_dict = stack.pop()
                    element_subset = []
                    for set_name, subsets in cur_dict.items():
                        if element_name == set_name:
                            if isinstance(subsets, list):
                                element_subset += [
                                    item
                                    for subset_name in subsets
                                    for item in self.config.system[subset_name]
                                ]
                        if isinstance(subsets, dict):
                            stack.append(subsets)

            # add element class
            element_names = list(set(element_set) - set(element_subset))
            for element_name in sorted(element_names):
                self._add_element(element_class, element_name)

    def _check_existing_carrier_data(self, carriers: list[str]):
        """Checks the existing carrier data and only regards those carriers for
        which folders exist.
        """
        # check if carriers exist
        for carrier in carriers:
            if carrier not in self.context.paths["set_carriers"].keys():
                # raise error if carrier is not in input data
                raise FileNotFoundError(
                    f"Carrier {carrier} selected in config does not exist ininput data"
                )
            elif "attributes.json" not in self.context.paths["set_carriers"][carrier]:
                raise FileNotFoundError(
                    f"The file attributes.json does not exist for the carrier {carrier}"
                )

    def _add_element(self, element_class: type[Element], element_name: str):
        """Add an element to the element_dict with the class labels as key.

        Args:
            element_class: Class of the element
            name: Name of the element
        """
        instance = element_class(
            element_name,
            self.config,
            self.context,
            self.energy_system,
            self,
            self.unit_handling,
        )
        # Add instance to all classes that element_class inherits from, including itself
        # MRO (Method Resolution Order) gives the order in which base classes
        # are searched when looking for a method.
        for class_name in element_class.__mro__:
            self.context.dict_elements[class_name.__name__].append(instance)

    def get_all_elements(self, class_name: type[Element]) -> list[Element]:
        """Get all elements of the class in the energy system."""
        return self.context.dict_elements[class_name.__name__]

    def get_elements(self) -> list[Element]:
        """Get all elements in the energy system."""
        return self.get_all_elements(Element)

    def get_all_names_of_elements(self, class_name):
        """Get all names of elements in class.

        :param cls: class of the elements to return
        :return: names_of_elements: list of elements in this class
        """
        # _elements_in_class = self.get_all_elements(class_name)
        # names_of_elements = []
        # for _element in _elements_in_class:
        #     names_of_elements.append(_element.name)
        # return names_of_elements
        return [_element.name for _element in self.get_all_elements(class_name)]

    def get_element(
        self, class_name: type[Element], element_name: str
    ) -> Element | None:
        """Get single element in class by name.

        :param name: name of element
        :param cls: class of the elements to return
        :return: element: return element whose name is matched
        """
        for element in self.get_all_elements(class_name):
            if element.name == element_name:
                return element
        return None

    def get_element_class(self, name: str) -> type[Element] | None:
        """Get element class by name. If not an element class, return None.

        :param name: name of element class
        :return: element_class: return element whose name is matched
        """

        # element_classes = {
        #     DICT_ELEMENT_CLASSES[class_name].label: DICT_ELEMENT_CLASSES[class_name]
        #     for class_name in DICT_ELEMENT_CLASSES
        # }
        # if name in element_classes.keys():
        #     return element_classes[name]
        # else:
        #     return None
        #
        for class_name in ELEMENT_TYPE_CLASSES:
            if ELEMENT_TYPE_CLASSES[class_name].label == name:
                return ELEMENT_TYPE_CLASSES[class_name]
        return None

    def get_attribute_of_all_elements(
        self,
        cls,
        attribute_name: str,
        capacity_types=False,
        return_attribute_is_series=False,
    ):
        """Get attribute values of all elements in a class.

        Args:
            cls: class of the elements to return
            attribute_name (str): name of attribute
            capacity_types (boolean): if attributes extracted for all capacity types
            return_attribute_is_series (boolean): if information on attribute type is
                returned
            dict_of_attributes (dict): dict of attribute values
            attribute_is_series: return information on attribute type
        """
        class_elements = self.get_all_elements(cls)
        dict_of_attributes = {}
        dict_of_units = {}
        attribute_is_series = False
        for element in class_elements:
            if not capacity_types:
                dict_of_attributes, attribute_is_series_temp, dict_of_units = (
                    self.append_attribute_of_element_to_dict(
                        element, attribute_name, dict_of_attributes, dict_of_units
                    )
                )
                if attribute_is_series_temp:
                    attribute_is_series = attribute_is_series_temp
            # if extracted for both capacity types
            else:
                for capacity_type in self.config.system.set_capacity_types:
                    # append energy only for storage technologies
                    if (
                        capacity_type == self.config.system.set_capacity_types[0]
                        or element.name in self.config.system.set_storage_technologies
                    ):
                        dict_of_attributes, attribute_is_series_temp, dict_of_units = (
                            self.append_attribute_of_element_to_dict(
                                element,
                                attribute_name,
                                dict_of_attributes,
                                dict_of_units,
                                capacity_type,
                            )
                        )
                        if attribute_is_series_temp:
                            attribute_is_series = attribute_is_series_temp
        if return_attribute_is_series:
            return dict_of_attributes, dict_of_units, attribute_is_series
        else:
            return dict_of_attributes

    def append_attribute_of_element_to_dict(
        self,
        element: "Element",
        attribute_name,
        dict_of_attributes: dict[str | tuple[str, str], pd.DataFrame | pd.Series | Any],
        dict_of_units,
        capacity_type=None,
    ):
        """Get attribute values of all elements in this class.

        Args:
            element: element of class
            attribute_name (str): str name of attribute
            dict_of_attributes (dict): dict of attribute values
            capacity_type: capacity type for which attribute extracted. If None,
                not listed in key
            dict_of_attributes: returns dict of attribute values
        """
        attribute_is_series = False
        # add Energy for energy capacity type
        if capacity_type == self.config.system.set_capacity_types[1]:
            attribute_name += "_energy"
        # if element does not have attribute
        if not hasattr(element, attribute_name):
            # if attribute is time series that does not exist
            if (
                attribute_name in element.raw_time_series
                and element.raw_time_series[attribute_name] is None
            ):
                return dict_of_attributes, None, dict_of_units
            else:
                raise AssertionError(
                    f"Element {element.name} does not have attribute {attribute_name}"
                )
        attribute: dict[str, Any] | pd.Series | int | Any | None = getattr(
            element, attribute_name
        )
        assert not isinstance(attribute, pd.DataFrame), (
            "Not yet implemented for pd.DataFrames. Wrong format for "
            f"element {element.name}"
        )
        # add attribute to dict_of_attributes
        if attribute is None:
            return dict_of_attributes, False, dict_of_units

        if isinstance(attribute, dict):
            dict_of_attributes.update(
                {(element.name,) + (key,): val for key, val in attribute.items()}
            )
        elif isinstance(attribute, pd.Series):
            if capacity_type:
                combined_key = (element.name, capacity_type)
            else:
                combined_key = element.name
            if attribute_name in element.units:
                if attribute_name in [
                    "conversion_factor",
                    "retrofit_flow_coupling_factor",
                ]:
                    dict_of_units[combined_key] = element.units[attribute_name]
                else:
                    dict_of_units[combined_key] = element.units[attribute_name][
                        "unit_in_base_units"
                    ].units
            else:
                # needed since these
                if attribute_name == "capex_capacity_existing":
                    dict_of_units[combined_key] = element.units["opex_specific_fixed"][
                        "unit_in_base_units"
                    ].units
                elif attribute_name == "capex_capacity_existing_energy":
                    dict_of_units[combined_key] = element.units[
                        "opex_specific_fixed_energy"
                    ]["unit_in_base_units"].units
                elif attribute_name == "capex_specific_transport":
                    dict_of_units[combined_key] = element.units["opex_specific_fixed"][
                        "unit_in_base_units"
                    ].units
                elif attribute_name == "capex_per_distance_transport":
                    base_units = self.unit_handling.base_units.items()
                    length_base_unit = [
                        key for key, value in base_units if value == "[length]"
                    ][0]
                    dict_of_units[combined_key] = element.units["opex_specific_fixed"][
                        "unit_in_base_units"
                    ].units / self.unit_handling.ureg(length_base_unit)
            if len(attribute) > 1:
                dict_of_attributes[combined_key] = attribute
                attribute_is_series = True
            else:
                if attribute.index == 0:
                    dict_of_attributes[combined_key] = attribute.squeeze()
                    attribute_is_series = False
                # since single-directed edges are allowed to exist (e.g. CH-DE exists,
                # DE-CH doesn't), TransportTechnology attributes shared with other
                # technologies (such as capacity existing)
                # mustn't be squeezed even-though the attributes length is smaller than
                # 1. Otherwise, pd.concat(dict_of_attributes) messes up in
                # initialize_component(), leading to an error further on in the code.
                else:
                    dict_of_attributes[combined_key] = attribute
                    attribute_is_series = True
        elif isinstance(attribute, int):
            if capacity_type:
                dict_of_attributes[(element.name, capacity_type)] = [attribute]
            else:
                dict_of_attributes[element.name] = [attribute]
        else:
            if capacity_type:
                dict_of_attributes[(element.name, capacity_type)] = attribute
            else:
                dict_of_attributes[element.name] = attribute
        return dict_of_attributes, attribute_is_series, dict_of_units

    def get_attribute_of_specific_element(
        self, cls, element_name: str, attribute_name: str
    ):
        """Get attribute of specific element in class.

        :param cls: class of the elements to return
        :param element_name: str name of element
        :param attribute_name: str name of attribute
        :return: attribute_value: value of attribute
        """
        # get element
        element = self.get_element(cls, element_name)
        # assert that _element exists and has attribute
        assert element, f"Element {element_name} not in class {cls.__name__}"
        assert hasattr(
            element, attribute_name
        ), f"Element {element_name} does not have attribute {attribute_name}"
        attribute_value = getattr(element, attribute_name)
        return attribute_value
