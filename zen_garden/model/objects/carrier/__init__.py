"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      March-2022
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  function that loads all classes and subclasses of carrier directory.
==========================================================================================================================================================================="""
from inspect import isclass
from pkgutil import iter_modules
from pathlib import Path
from importlib import import_module
from ..energy_system import EnergySystem

# iterate through files
packageDir = Path(__file__).resolve().parent
# store classes in a dictionary
carrier_classes = dict()
for (_, moduleName, _) in iter_modules([packageDir]):
    # import file and iterate through its attributes
    module = import_module(f"{__name__}.{moduleName}")
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        # if attribute is class, add class to variables
        if isclass(attribute) and "Carrier" in attribute_name:
            if attribute_name not in carrier_classes.keys():
                globals()[attribute_name] = attribute
                carrier_classes[attribute_name] = attribute

# update dict_element_classes
EnergySystem.dict_element_classes.update(carrier_classes)
