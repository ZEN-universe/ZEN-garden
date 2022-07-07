"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      March-2022
Authors:      Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  function that loads all classes and subclasses of carrier directory.
==========================================================================================================================================================================="""
from inspect                     import isclass
from pkgutil                     import iter_modules
from pathlib                     import Path
from importlib                   import import_module
from model.objects.energy_system import EnergySystem


# iterate through files
packageDir = Path(__file__).resolve().parent
# store classes in a dictionary
carrierClasses = dict()
for (_, moduleName, _) in iter_modules([packageDir]):
    # import file and iterate through its attributes
    module = import_module(f"{__name__}.{moduleName}")
    for attributeName in dir(module):
        attribute = getattr(module, attributeName)
        # if attribute is class, add class to variables
        if isclass(attribute) and "Carrier" in attributeName:
            if attributeName not in carrierClasses.keys():
                globals()[attributeName]      = attribute
                carrierClasses[attributeName] = attribute

# update dictElementClasses
EnergySystem.dictElementClasses.update(carrierClasses)
