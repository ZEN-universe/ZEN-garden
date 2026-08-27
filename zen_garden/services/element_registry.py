import copy
import logging
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd

from zen_garden.elements.element import Element
from zen_garden.elements.energy_system import EnergySystem
from zen_garden.services.service_container import ServiceContainer

if TYPE_CHECKING:
    from zen_garden.model.time_steps import TimeStepsDicts
    from zen_garden.preprocess.unit_handling import UnitHandling
    from zen_garden.services.dataset_path_resolver import DatasetPathResolver
    from zen_garden.services.scenario_dict import ScenarioDict
    from zen_garden.topology.model_schema import ModelSchema
    from zen_garden.types import YearSpecificTs
    from zen_garden.utils.input_data_checks import InputDataChecks

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Element)


class ElementRegistry:
    def __init__(
        self,
        service_container: "ServiceContainer",
        model_schema: "ModelSchema",
        input_data_checks: "InputDataChecks",
        unit_handling: "UnitHandling",
        dataset_path_resolver: "DatasetPathResolver",
        scenario_dict: "ScenarioDict",
        time_steps: "TimeStepsDicts",
        year_specific_ts: "YearSpecificTs",
    ):
        self.service_container = service_container
        self.model_schema = model_schema
        self.input_data_checks = input_data_checks
        self.unit_handling = unit_handling
        self.dataset_path_resolver = dataset_path_resolver
        self.scenario_dict = scenario_dict
        self.time_steps = time_steps
        self.year_specific_ts = year_specific_ts


    def register_elements(self):
        """Set up the parameters, variables and constraints of the carriers."""
        logger.info("\n--- Add elements to model--- \n")
        for element_class in self.model_schema.element_classes:
            if element_class is EnergySystem:
                self._register_element(EnergySystem, EnergySystem.name)
                continue
            element_name = element_class.label
            element_set = self.model_schema.config.system[element_name]

            # before adding the carriers, get set_carriers
            # check if carrier data exists
            if element_name == "set_carriers":
                element_set: list[str] = self.model_schema.set_carriers
                self.input_data_checks.check_existing_carrier_data(element_set)

            # check if element_set has a subset and remove subset from element_set
            element_subset: list[str] = []
            if element_name in self.model_schema.config.analysis.subsets.keys():
                if isinstance(
                    self.model_schema.config.analysis.subsets[element_name], list
                ):
                    subset_names = self.model_schema.config.analysis.subsets[
                        element_name
                    ]
                elif isinstance(
                    self.model_schema.config.analysis.subsets[element_name], dict
                ):
                    subset_names = self.model_schema.config.analysis.subsets[
                        element_name
                    ].keys()
                else:
                    raise ValueError(
                        f"Subset {element_name} has to be either a list or a dict"
                    )
                element_subset = [
                    item
                    for subset in subset_names
                    for item in self.model_schema.config.system[subset]
                ]
            else:
                stack = [
                    _dict
                    for _dict in copy.deepcopy(
                        self.model_schema.config.analysis.subsets
                    ).values()
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
                                    for item in self.model_schema.config.system[
                                        subset_name
                                    ]
                                ]
                        if isinstance(subsets, dict):
                            stack.append(subsets)

            # add element class
            element_names = list(set(element_set) - set(element_subset))
            for element_name in sorted(element_names):
                self._register_element(element_class, element_name)

    def _register_element(self, element_class: type[Element], element_name: str):
        """Add an element to the element_dict with the class labels as key.

        Args:
            element_class: Class of the element
            name: Name of the element
        """
        instance = self.service_container.build(
            element_class, element_name=element_name
        )
        # Add instance to all classes that element_class inherits from, including itself
        # MRO (Method Resolution Order) gives the order in which base classes
        # are searched when looking for a method.
        self.model_schema.register_element(instance)

    def all_elements_of_type(self, class_name: type[T]) -> list[T]:
        """Get all elements of the class in the energy system."""
        return self.model_schema.all_elements_of_type(class_name)

    def all_elements(self) -> list[Element]:
        """Get all elements in the energy system."""
        return self.model_schema.all_elements()

    def all_names_of_elements(self, class_name: type[Element]) -> list[str]:
        """Get all names of elements in class.

        :param cls: class of the elements to return
        :return: names_of_elements: list of elements in this class
        """
        return [_element.name for _element in self.all_elements_of_type(class_name)]

    def get_element(self, class_name: type[T], element_name: str) -> T | None:
        """Get single element in class by name.

        :param name: name of element
        :param cls: class of the elements to return
        :return: element: return element whose name is matched
        """
        for element in self.all_elements_of_type(class_name):
            if element.name == element_name:
                return element
        return None

    def get_element_class(self, name: str) -> type[Element] | None:
        """Get element class by name. If not an element class, return None.

        :param name: name of element class
        :return: element_class: return element whose name is matched
        """
        for element_class in self.model_schema.element_classes:
            if element_class.label == name:
                return element_class
        return None

    def get_attribute_of_all_elements_with_units(
        self, cls, attribute_name: str, capacity_types=False
    ):
        """Get attribute values and units of all elements in a class.

        Args:
            cls: class of the elements to return
            attribute_name (str): name of attribute
            capacity_types (boolean): if attributes extracted for all capacity types

        Returns:
            dict_of_attributes (dict): dict of attribute values
            dict_of_units (dict): dict of attribute units
            attribute_is_series: return information on attribute type
        """
        class_elements = self.all_elements_of_type(cls)
        dict_of_attributes: dict[str | tuple[str, ...], Any] = {}
        dict_of_units: dict[str | tuple[str, ...], Any] = {}
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
                for capacity_type in self.model_schema.config.system.set_capacity_types:
                    # append energy only for storage technologies
                    if (
                        capacity_type
                        == self.model_schema.config.system.set_capacity_types[0]
                        or element.name
                        in self.model_schema.config.system.set_storage_technologies
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
        return dict_of_attributes, dict_of_units, attribute_is_series

    def get_attribute_of_all_elements(
        self, cls, attribute_name: str, capacity_types=False
    ):
        """Get attribute values of all elements in a class.

        Args:
            cls: class of the elements to return
            attribute_name (str): name of attribute
            capacity_types (boolean): if attributes extracted for all capacity types
        """
        return self.get_attribute_of_all_elements_with_units(
            cls, attribute_name, capacity_types
        )[0]

    def append_attribute_of_element_to_dict(
        self,
        element: "Element",
        attribute_name,
        dict_of_attributes: dict[str | tuple[str, ...], Any],
        dict_of_units: dict[str | tuple[str, ...], Any],
        capacity_type=None,
    ):
        """Get attribute values of all elements in this class.

        Args:
            element: element of class
            attribute_name (str): str name of attribute
            dict_of_attributes (dict): dict of attribute values
            dict_of_units (dict): dict of attribute units
            capacity_type: capacity type for which attribute extracted. If None,
                not listed in key
        """
        attribute_is_series = False
        # add Energy for energy capacity type
        if capacity_type == self.model_schema.config.system.set_capacity_types[1]:
            attribute_name = f"{attribute_name}_energy"
        # if element does not have attribute
        if not hasattr(element, attribute_name):
            is_missing_time_series = (
                attribute_name in element.raw_time_series
                and element.raw_time_series[attribute_name] is None
            )
            if is_missing_time_series:
                return dict_of_attributes, None, dict_of_units
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
        if attribute is None:
            return dict_of_attributes, False, dict_of_units

        element_key = (element.name, capacity_type) if capacity_type else element.name
        attribute_is_series = False

        if isinstance(attribute, dict):
            dict_of_attributes.update(
                {(element.name, key): value for key, value in attribute.items()}
            )
        elif isinstance(attribute, pd.Series):
            fallback_units = {
                "capex_capacity_existing": ("opex_specific_fixed", None),
                "capex_capacity_existing_energy": (
                    "opex_specific_fixed_energy",
                    None,
                ),
                "capex_specific_transport": ("opex_specific_fixed", None),
                "capex_per_distance_transport": (
                    "opex_specific_fixed",
                    "[length]",
                ),
            }
            unit_attribute = attribute_name
            divisor_dimension = None
            if attribute_name not in element.units:
                unit_attribute, divisor_dimension = fallback_units.get(
                    attribute_name, (None, None)
                )

            if unit_attribute is not None:
                unit = element.units[unit_attribute]
                if attribute_name not in {
                    "conversion_factor",
                    "retrofit_flow_coupling_factor",
                }:
                    unit = unit["unit_in_base_units"].units
                if divisor_dimension is not None:
                    base_unit = next(
                        key
                        for key, dimension in self.unit_handling.base_units.items()
                        if dimension == divisor_dimension
                    )
                    unit = unit / self.unit_handling.ureg(base_unit)
                dict_of_units[element_key] = unit

            # Preserve non-default indices, such as single-directed transport edges,
            # so that pd.concat retains their dimension downstream.
            is_scalar = len(attribute) == 1 and attribute.index[0] == 0
            dict_of_attributes[element_key] = (
                attribute.squeeze() if is_scalar else attribute
            )
            attribute_is_series = not is_scalar
        else:
            dict_of_attributes[element_key] = (
                [attribute] if isinstance(attribute, int) else attribute
            )

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
