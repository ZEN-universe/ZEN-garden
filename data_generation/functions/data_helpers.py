"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Helper functions for data creation to keep create_inputd_data.py cleaner
==========================================================================================================================================================================="""
from cmath import inf
import pandas as pd
from sympy import source

def getFolderNames():
    """ this function returns the necessary folders in a input data structure """
    folderNames =   [
        "setCarriers",
        "setConversionTechnologies",
        "setNodes",
        "setScenarios",
        "setStorageTechnologies",
        "setTimeSteps",
        "setTransportTechnologies"
    ]
    return folderNames

def getTechnologies(sourcePath,technologyType,sourceName = None):
    """ this function returns the selected  technologies as read from csv. 
    If sourceName, return dict with names of technologies as used in source
    :param sourcePath: path of source data folder
    :param technologyType: type of technology
    :param sourceName: name of input data source. They often use different nomenclature
    :return Technologies: list/dict of  technologies"""
    technologyFiles = {
        "conversion"    : "setConversionTechnologies.csv",
        "transport"     : "setTransportTechnologies.csv",
        "storage"       : "setStorageTechnologies.csv",
    }
    assert technologyType in technologyFiles.keys(), f"Technology type {technologyType} unknown"
    technologyFile = technologyFiles[technologyType]
    dfTechnologies = pd.read_csv(sourcePath / "sets" / technologyFile)
    if not sourceName:
        technologies = dfTechnologies["technology"].to_list()
    else:
        assert sourceName in dfTechnologies.columns, f"Source name {sourceName} not in columns of {technologyFile}"
        dfTechnologies = dfTechnologies.set_index("technology")
        technologies = dfTechnologies[sourceName].to_dict()
    return technologies

def getDefaultValue(attribute):
    """ this function returns the default value of an attribute.
    :param attribute: attribute for which default value is returned
    :return defaultValue: default value of attribute """
    defaultValues={                                 # unit
        "minBuiltCapacity"          : 0,            # MW,MWh
        "maxBuiltCapacity"          : 100,          # MW,MWh
        "minLoad"                   : 0,            # -
        "maxLoad"                   : 1,            # -
        "lifetime"                  : 20,           # a
        "opexSpecific"              : 0,            # EUR/MWh
        "capacityLimit"             : inf,          # MW
        "carbonIntensity"           : 0,            # tCO2/MWh
        "demandCarrier"             : 0,            # MW
        "availabilityCarrierImport" : inf,          # MW
        "availabilityCarrierExport" : inf,          # MW
        "exportPriceCarrier"        : 0,            # EUR/MWh
        "importPriceCarrier"        : 0,            # EUR/MWh
        "referenceCarrier"          : "electricity",# -     
        "storageLevelRepetition"    : 1             # -    
    }
    # remove "Default" from attribute name
    attribute = attribute.replace("Default","")
    if attribute in defaultValues:
        return defaultValues[attribute]
    else:
        return None

def getAttributesOfSet(setName):
    """ this function returns the names of the attributes for each element type (set) 
    :param setName: name of element set 
    :returns attributes: list of attributes of element set """
    attributesOfSets = {
        "setTechnologies": [
            "minBuiltCapacity",
            "maxBuiltCapacity",
            "minLoad",
            "maxLoad",
            "lifetime",
            "opexSpecificDefault",
            "referenceCarrier",
            "capacityLimitDefault",
            "carbonIntensityDefault"
        ],
        "setConversionTechnologies": [
            "inputCarrier",
            "outputCarrier",
        ],
        "setTransportTechnologies": [
            "lossFlow",
            "capexPerDistanceDefault",
        ],
        "setStorageTechnologies": [
            "efficiencyCharge",
            "efficiencyDischarge",
            "selfDischarge",
            "capexSpecificDefault"
        ],
        "setCarriers": [
            "carbonIntensityDefault",
            "demandCarrierDefault",
            "availabilityCarrierImportDefault",
            "availabilityCarrierExportDefault",
            "exportPriceCarrierDefault",
            "importPriceCarrierDefault"
        ]
    } 
    attributesOfSets["setConversionTechnologies"]   = attributesOfSets["setTechnologies"] + attributesOfSets["setConversionTechnologies"]
    attributesOfSets["setTransportTechnologies"]    = attributesOfSets["setTechnologies"] + attributesOfSets["setTransportTechnologies"]
    attributesOfSets["setStorageTechnologies"]      = attributesOfSets["setTechnologies"] + attributesOfSets["setStorageTechnologies"]

    attributes = attributesOfSets[setName]
    return attributes

def setManualAttributesConversion(elementName,dfAttribute):
    """ sets manual attributes for conversion technologies
    :param elementName: name of technology
    :param dfAttribute: attribute dataframe
    :return dfAttribute: attribute dataframe """
    
    return dfAttribute

def setManualAttributesTransport(elementName,dfAttribute):
    """ sets manual attributes for transport technologies
    :param elementName: name of technology
    :param dfAttribute: attribute dataframe
    :return dfAttribute: attribute dataframe """
    # source: link-techs in euro-calliope
    if elementName == "power_line":
        dfAttribute.loc["lossFlow"]                 = 5E-5  # 1/km 
        dfAttribute.loc["lifetime"]                 = 60    # a
        dfAttribute.loc["capexPerDistanceDefault"]  = 900/2 # EUR/km/MW
        dfAttribute.loc["capacityLimitDefault"]     = 6000  # MW (loosely chosen from highest capacity in ENTSO-E TYNDP)
    return dfAttribute

def setManualAttributesStorage(elementName,dfAttribute):
    """ sets manual attributes for storage technologies
    :param elementName: name of technology
    :param dfAttribute: attribute dataframe
    :return dfAttribute: attribute dataframe """
    # source is FactSheet_Energy_Storage_0219
    if elementName == "battery":
        dfAttribute.loc["efficiencyCharge"]     = 0.95
        dfAttribute.loc["efficiencyDischarge"]  = 0.95
        dfAttribute.loc["selfDischarge"]        = 0.1/100                           # ESM_Final_Report_05-Nov-2019
        dfAttribute.loc["maxLoad"]              = 1/2                               # 1/(typical discharge time)
        dfAttribute.loc["maxBuiltCapacity"]     = 100/dfAttribute.loc["maxLoad"]    # MWh, discharge in 1/maxLoad --> E_max = P_rated/maxLoad
        dfAttribute.loc["capexSpecificDefault"] = 3000*dfAttribute.loc["maxLoad"]   # EUR/MWh, 
    elif elementName == "pumped_hydro":
        dfAttribute.loc["efficiencyCharge"]     = 0.9
        dfAttribute.loc["efficiencyDischarge"]  = 0.9
        dfAttribute.loc["selfDischarge"]        = 0                                 # ESM_Final_Report_05-Nov-2019
        dfAttribute.loc["maxLoad"]              = 1/16                              # 1/(typical discharge time)
        dfAttribute.loc["maxBuiltCapacity"]     = 3000/dfAttribute.loc["maxLoad"]   # MWh, discharge in 1/maxLoad --> E_max = P_rated*maxLoad
        dfAttribute.loc["capexSpecificDefault"] = 2700*dfAttribute.loc["maxLoad"]   # EUR/MWh, 
    return dfAttribute

def setManualAttributesCarriers(elementName,dfAttribute):
    """ sets manual attributes for carriers
    :param elementName: name of carrier
    :param dfAttribute: attribute dataframe
    :return dfAttribute: attribute dataframe """

    return dfAttribute

def setInputOutputCarriers(elementName,inputOutputType):
    """ returns input output carriers for conversion technology """
    carriers = {
        "photovoltaics": {
            "input":    None,
            "output":   "electricity",
        },
        "wind_onshore": {
            "input":    None,
            "output":   "electricity",
        },
        "hard_coal": {
            "input":    "hard_coal",
            "output":   "electricity",
        },
        "natural_gas_turbine": {
            "input":    "natural_gas",
            "output":   "electricity",
        },
        "nuclear": {
            "input":    "uranium",
            "output":   "electricity",
        },
        "run-of-river_hydro": {
            "input":    None,
            "output":   "electricity",
        },
    }
    assert elementName in carriers, f"Technology {elementName} not in list of technologies {list(carriers.keys())}"
    return carriers[elementName][inputOutputType]

def getNumberOfNewPlants(elementName):
    """ define arbitrary maximum number of new plants"""
    numberOfNewPlants = {
        "photovoltaics":        1000,
        "wind_onshore":         1000,
        "hard_coal":            55,
        "natural_gas_turbine":  200,
        "nuclear":              50,
        "run-of-river_hydro":   500,
    }
    assert elementName in numberOfNewPlants, f"Technology {elementName} not in list of technologies {list(numberOfNewPlants.keys())}"
    return numberOfNewPlants[elementName]

def getCostIndex(technology):
    """ this method returns the index to look up cost data in Potencia Assumption """
    size = "M"
    cogeneration = "Electricity only"
    # Nuclear
    if technology == "Nuclear":
        type_proc = "Nuclear power plants"
        technology = "Nuclear - current"
    # coal
    elif technology == "Coal fired":
        type_proc = "Coal fired power plants"
        technology = "Steam turbine" 
    # lignite
    elif technology == "Lignite fired":
        type_proc = "Lignite fired power plants"
        technology = "Steam turbine" 
    # gas
    elif technology == "Gas fired":
        type_proc = "Gas fired power plants (Natural gas, biogas)"
        technology = "Gas turbine combined cycle" 
    # onshore
    elif technology == "Onshore":
        type_proc = "Wind power plants"
        technology = "Onshore" 
    # offshore
    elif technology == "Offshore":
        type_proc = "Wind power plants"
        technology = "Offshore" 
    # run-of-river
    elif technology == "Run-of-river":
        type_proc = "Hydro plants"
        technology = "Run-of-river" 
    # geothermal
    elif technology == "Geothermal":
        type_proc = "Geothermal power plants"
        technology = "Geothermal power plants"
    # pv 
    elif technology == "Solar photovoltaics":
        type_proc = "Solar PV power plants"
        technology = "Solar PV power plants" 
    # solarthermal
    elif technology == "Solar thermal":
        type_proc = "Solar thermal power plants"
        technology = "Solar thermal power plants" 
    else:
        type_proc = "-1"
        technology = "-1"
    
    return(type_proc,technology,cogeneration,size)

def getConstants(constant):
    """ returns constant value """
    constants = {
        "MWh2toe": 0.0859845
    }
    return constants[constant]