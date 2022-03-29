"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      March-2022
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  function that loads all classes and subclasses of technology directory.
==========================================================================================================================================================================="""
from inspect   import isclass
from pkgutil   import iter_modules
from pathlib   import Path
from importlib import import_module


# iterate through files
packageDir = Path(__file__).resolve().parent
# store class names in a list
technologyList = []
for (_, moduleName, _) in iter_modules([packageDir]):
    # import file and iterate through its attributes
    module = import_module(f"{__name__}.{moduleName}")
    for attributeName in dir(module):
        attribute = getattr(module, attributeName)
        # if attribute is class, add class to variables
        if isclass(attribute) and "Technology" in attributeName:
            technologyList.append(attributeName)
            globals()[attributeName] = attribute
technologyList = set(technologyList)
technologyList.remove("Technology")
globals()["technologyList"] = list(technologyList)