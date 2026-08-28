"""Factory that instantiates the model's elements from the schema and config."""

import copy
import logging
from typing import TYPE_CHECKING

from zen_garden.elements.energy_system import EnergySystem
from zen_garden.model.element import Element

if TYPE_CHECKING:
    from zen_garden.input.input_data_checks import InputDataChecks
    from zen_garden.model.schema import ModelSchema
    from zen_garden.service_container import ServiceContainer

logger = logging.getLogger(__name__)


class ElementFactory:
    """Instantiates every model element and registers it in the model schema.

    The concrete elements to build (carriers, technologies, the energy system)
    are derived from ``model_schema.element_classes`` together with the
    configured system sets. Each element is created through the service
    container so its dependencies are injected, then handed to
    :meth:`ModelSchema.register_element`. Querying the resulting elements is the
    job of :class:`ElementRegistry`.
    """

    def __init__(
        self,
        service_container: "ServiceContainer",
        model_schema: "ModelSchema",
        input_data_checks: "InputDataChecks",
    ):
        self.service_container = service_container
        self.model_schema = model_schema
        self.input_data_checks = input_data_checks

    def register_elements(self):
        """Instantiate every configured element and register it in the schema."""
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
        # Injected services: model_schema, network_topology, element_registry,
        # unit_converter, dataset_path_resolver, scenario_dict, input_data_checks,
        # year_specific_ts; explicit argument: element_name.
        instance = self.service_container.build(
            element_class, element_name=element_name
        )
        # Add instance to all classes that element_class inherits from, including itself
        # MRO (Method Resolution Order) gives the order in which base classes
        # are searched when looking for a method.
        self.model_schema.register_element(instance)
