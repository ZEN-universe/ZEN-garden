"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      March-2022
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  function that loads all classes and subclasses of technology directory.
==========================================================================================================================================================================="""
from inspect                     import isclass
from pkgutil                     import iter_modules
from pathlib                     import Path
from importlib                   import import_module
from model.objects.energy_system import EnergySystem


# iterate through files
packageDir = Path(__file__).resolve().parent
# store class names in a list
technologyClasses = dict()
for (_, moduleName, _) in iter_modules([packageDir]):
    # import file and iterate through its attributes
    module = import_module(f"{__name__}.{moduleName}")
    for attributeName in dir(module):
        attribute = getattr(module, attributeName)
        # if attribute is class, add class to variables
        if isclass(attribute) and "Technology" in attributeName:
            if attributeName not in technologyClasses.keys():
                globals()[attributeName]         = attribute
                technologyClasses[attributeName] = attribute
# update dictElementClasses
EnergySystem.dictElementClasses.update(technologyClasses)