"""Unit tests for all Variable classes in the elements folder."""

import inspect
import pytest
from typing import get_args

from zen_garden.elements.carrier.variables import CARRIER_VARIABLES
from zen_garden.elements.conversion_technology.variables import (
    CONVERSION_TECHNOLOGY_VARIABLES,
)
from zen_garden.elements.energy_system.variables import ENERGY_SYSTEM_VARIABLES
from zen_garden.elements.storage_technology.variables import (
    STORAGE_TECHNOLOGY_VARIABLES,
)
from zen_garden.elements.technology.variables import TECHNOLOGY_VARIABLES
from zen_garden.elements.transport_technology.variables import (
    TRANSPORT_TECHNOLOGY_VARIABLES,
)
from zen_garden.elements.retrofitting_technology.variables import (
    RETROFITTING_TECHNOLOGY_VARIABLES,
)
from zen_garden.topology.generic_variable import GenericVariable


# Collect all variables from all elements
ALL_VARIABLE_LISTS = {
    "carrier": CARRIER_VARIABLES,
    "energy_system": ENERGY_SYSTEM_VARIABLES,
    "technology": TECHNOLOGY_VARIABLES,
    "conversion_technology": CONVERSION_TECHNOLOGY_VARIABLES,
    "transport_technology": TRANSPORT_TECHNOLOGY_VARIABLES,
    "storage_technology": STORAGE_TECHNOLOGY_VARIABLES,
    "retrofitting_technology": RETROFITTING_TECHNOLOGY_VARIABLES,
}


class TestVariableInheritance:
    """Test that all variable classes are properly defined and accessible."""
    def test_all_variables_are_subclasses_of_generic_variable(self):
        """Test that all variables inherit from GenericVariable."""
        for element_name, var_list in ALL_VARIABLE_LISTS.items():
            for variable_class in var_list:
                assert issubclass(
                    variable_class, GenericVariable
                ), f"{variable_class.__name__} is not a subclass of GenericVariable"

class TestVariableUniqueness:
    """Test that variables are uniquely named globally across all elements."""

    def test_variable_names_are_globally_unique(self):
        """Test that all variable names are unique across all elements."""
        all_names = []
        name_to_element = {}
        
        for element_name, var_list in ALL_VARIABLE_LISTS.items():
            for variable_class in var_list:
                var_name = variable_class.name
                all_names.append(var_name)
                if var_name in name_to_element:
                    name_to_element[var_name].append(element_name)
                else:
                    name_to_element[var_name] = [element_name]
        
        # Check for duplicates
        duplicates = {name: elements for name, elements in name_to_element.items() if len(elements) > 1}
        
        assert (
            len(all_names) == len(set(all_names))
        ), f"Duplicate variable names found: {duplicates}"

