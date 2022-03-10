"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Helper functions for data creation to keep create_inputd_data.py cleaner
==========================================================================================================================================================================="""
from cmath import inf
import pandas as pd
import pickle 
import os

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
        "minBuiltCapacity"          : 0,            # GW,GWh
        "maxBuiltCapacity"          : 1,            # GW,GWh
        "minLoad"                   : 0.1,          # -
        "maxLoad"                   : 1,            # -
        "lifetime"                  : 20,           # a
        "opexSpecific"              : 0,            # kEUR/GWh
        "capacityLimit"             : inf,          # GW
        "carbonIntensity"           : 0,            # ktCO2/GWh
        "demandCarrier"             : 0,            # GW
        "availabilityCarrierImport" : inf,          # GW
        "availabilityCarrierExport" : inf,          # GW
        "exportPriceCarrier"        : 0,            # kEUR/GWh
        "importPriceCarrier"        : 0,            # kEUR/GWh
        "referenceCarrier"          : "electricity",# -     
        "storageLevelRepetition"    : 1,             # -
        "distanceEuclidean"         : inf
    }
    # remove "Default" from attribute name
    attribute = attribute.replace("Default","")
    if attribute in defaultValues:
        return defaultValues[attribute]
    else:
        return None

def getDefaultUnit(attribute):
    """ this function returns the default unit of an attribute.
    :param attribute: attribute for which default unit is returned
    :return defaultUnit: default value of attribute """
    defaultUnits={                                 # unit
        "minBuiltCapacity"          : "GW",            # GW,GWh
        "maxBuiltCapacity"          : "GW",            # GW,GWh
        "minLoad"                   : "",            # -
        "maxLoad"                   : "",            # -
        "lifetime"                  : "",           # a
        "opexSpecific"              : "kiloEuro/GWh",            # kEUR/GWh
        "capacityLimit"             : "GW",          # GW
        "carbonIntensity"           : "kilotons/GWh",            # ktCO2/GWh
        "demandCarrier"             : "GW",            # GW
        "availabilityCarrierImport" : "GW",          # GW
        "availabilityCarrierExport" : "GW",          # GW
        "exportPriceCarrier"        : "kiloEuro/GWh",            # kEUR/GWh
        "importPriceCarrier"        : "kiloEuro/GWh",            # kEUR/GWh
        "referenceCarrier"          : "GW",# -
        "storageLevelRepetition"    : "",             # -
        "distanceEuclidean"         : "km"
    }
    # remove "Default" from attribute name
    attribute = attribute.replace("Default","")
    if attribute in defaultUnits:
        return defaultUnits[attribute]
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
            "distanceEuclideanDefault"
        ],
        "setStorageTechnologies": [
            "efficiencyCharge",
            "efficiencyDischarge",
            "selfDischarge",
            "capexSpecificDefault",
            "storageLevelRepetition"
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
        dfAttribute.loc["lossFlow","value"]                 = 5E-5  # 1/km
        dfAttribute.loc["lifetime","value"]                 = 60    # a
        dfAttribute.loc["capexPerDistanceDefault","value"]  = 900/2 # kEUR/km/GW
        dfAttribute.loc["capacityLimitDefault","value"]     = 6     # GW (loosely chosen from highest capacity in ENTSO-E TYNDP)
        dfAttribute.loc["lossFlow", "unit"]                 = "1/km"
        dfAttribute.loc["capexPerDistanceDefault", "unit"]  = "kiloEuro/km/GW"
    return dfAttribute

def setManualAttributesStorage(elementName,dfAttribute):
    """ sets manual attributes for storage technologies
    :param elementName: name of technology
    :param dfAttribute: attribute dataframe
    :return dfAttribute: attribute dataframe """
    # source is FactSheet_Energy_Storage_0219
    if elementName == "battery":
        dfAttribute.loc["efficiencyCharge","value"]     = 0.95
        dfAttribute.loc["efficiencyDischarge","value"]  = 0.95
        dfAttribute.loc["selfDischarge","value"]        = 0.1/100                               # ESM_Final_Report_05-Nov-2019
        dfAttribute.loc["maxLoad","value"]              = 1/2                                   # 1/(typical discharge time)
        dfAttribute.loc["maxBuiltCapacity","value"]     = 0.1/dfAttribute.loc["maxLoad","value"]        # GWh, discharge in 1/maxLoad --> E_max = P_rated/maxLoad
        dfAttribute.loc["capexSpecificDefault","value"] = 3000*dfAttribute.loc["maxLoad","value"]*1000  # kEUR/GWh,
    elif elementName == "pumped_hydro":
        dfAttribute.loc["efficiencyCharge","value"]     = 0.9
        dfAttribute.loc["efficiencyDischarge","value"]  = 0.9
        dfAttribute.loc["selfDischarge","value"]        = 0                                     # ESM_Final_Report_05-Nov-2019
        dfAttribute.loc["maxLoad","value"]              = 1/16                                  # 1/(typical discharge time)
        dfAttribute.loc["maxBuiltCapacity","value"]     = 3/dfAttribute.loc["maxLoad","value"]          # GWh, discharge in 1/maxLoad --> E_max = P_rated*maxLoad
        dfAttribute.loc["capexSpecificDefault","value"] = 2700*dfAttribute.loc["maxLoad","value"]*1000  # kEUR/GWh,
    dfAttribute.loc["efficiencyCharge", "unit"]         = ""
    dfAttribute.loc["efficiencyDischarge", "unit"]      = ""
    dfAttribute.loc["selfDischarge", "unit"]            = ""
    dfAttribute.loc["maxLoad", "unit"]                  = "GW/GWh"
    dfAttribute.loc["maxBuiltCapacity", "unit"]         = "GWh"
    dfAttribute.loc["capexSpecificDefault", "unit"]     = "kiloEuro/GWh"
    return dfAttribute

def setManualAttributesCarriers(elementName,dfAttribute):
    """ sets manual attributes for carriers
    :param elementName: name of carrier
    :param dfAttribute: attribute dataframe
    :return dfAttribute: attribute dataframe """
    if elementName == "electricity":
        dfAttribute.loc["importPriceCarrierDefault","value"]    = 30000  # kEUR/GWh, current maximum market clearing price at coupled European Power Exchange
        dfAttribute.loc["carbonIntensityDefault","value"]       = 0.127 # kt_CO2/GWh, value from Gabrielli et al.
    return dfAttribute

def setInputOutputCarriers(elementName,inputOutputType):
    """ returns input output carriers for conversion technology """
    carriers = {
        "photovoltaics": {
            "input"     : None,
            "output"    : "electricity",
            "reference" : "electricity",
        },
        "wind_onshore": {
            "input"     : None,
            "output"    : "electricity",
            "reference" : "electricity",
        },
        "hard_coal_plant": {
            "input"     : "hard_coal",
            "output"    : "electricity",
            "reference" : "electricity",
        },
        "natural_gas_turbine": {
            "input"     : "natural_gas",
            "output"    : "electricity",
            "reference" : "electricity",
        },
        "nuclear": {
            "input"     : "uranium",
            "output"    : "electricity",
            "reference" : "electricity",
        },
        "run-of-river_hydro": {
            "input"     : None,
            "output"    : "electricity",
            "reference" : "electricity",
        },
    }
    assert elementName in carriers, f"Technology {elementName} not in list of technologies {list(carriers.keys())}"
    return carriers[elementName][inputOutputType]

def getCarrierUnits(carrier):
    """ get units of carrier flows """
    carrierUnits = {
        "electricity"   : "GW",
        "hard_coal"     : "GW",
        "natural_gas"   : "GW",
        "uranium"       : "GW",
        "lignite"       : "GW"
    }
    if carrier:
        assert carrier in carrierUnits, f"Technology {carrier} not in list of technologies {list(carrierUnits.keys())}"
        return carrierUnits[carrier]
    else:
        return None

def getNumberOfNewPlants(elementName):
    """ define arbitrary maximum number of new plants"""
    numberOfNewPlants = {
        "photovoltaics":        100000,
        "wind_onshore":         100000,
        "hard_coal_plant":      2000,
        "natural_gas_turbine":  50000,
        "nuclear":              2000,
        "run-of-river_hydro":   200000,
    }
    assert elementName in numberOfNewPlants, f"Technology {elementName} not in list of technologies {list(numberOfNewPlants.keys())}"
    return numberOfNewPlants[elementName]

def getAttributeIndex(technology):
    """ this method returns the index to look up attributes in Potencia Assumption """
    size = "M"
    cogeneration = "Electricity only"
    # Nuclear
    if technology == "Nuclear":
        typeTech = "Nuclear power plants"
        technology = "Nuclear - current"
    # coal
    elif technology == "Coal fired":
        typeTech = "Coal fired power plants"
        technology = "Steam turbine" 
    # lignite
    elif technology == "Lignite fired":
        typeTech = "Lignite fired power plants"
        technology = "Steam turbine" 
    # gas
    elif technology == "Gas fired":
        typeTech = "Gas fired power plants (Natural gas, biogas)"
        technology = "Gas turbine combined cycle" 
    # onshore
    elif technology == "Onshore":
        typeTech = "Wind power plants"
        technology = "Onshore" 
    # offshore
    elif technology == "Offshore":
        typeTech = "Wind power plants"
        technology = "Offshore" 
    # run-of-river
    elif technology == "Run-of-river":
        typeTech = "Hydro plants"
        technology = "Run-of-river" 
    # geothermal
    elif technology == "Geothermal":
        typeTech = "Geothermal power plants"
        technology = "Geothermal power plants"
    # pv 
    elif technology == "Solar photovoltaics":
        typeTech = "Solar PV power plants"
        technology = "Solar PV power plants" 
    # solarthermal
    elif technology == "Solar thermal":
        typeTech = "Solar thermal power plants"
        technology = "Solar thermal power plants" 
    else:
        typeTech = "-1"
        technology = "-1"
    
    return(typeTech,technology,cogeneration,size)

def getEntsoeTechnologyIdentifier(technology):
    """ this method returns the PSR technology identifier to get the generation profile of the technology from the entsoe database """
    entsoeTechnologies = {
        "photovoltaics":        "B16",
        "wind_onshore":         "B19",
        "hard_coal_plant":      "B05",
        "natural_gas_turbine":  "B04",
        "nuclear":              "B14",
        "run-of-river_hydro":   "B11"
    }
    return entsoeTechnologies[technology]

def getCarrierIdentifier(carrier):
    """ this method returns the index to look up the carrier in Potencia fuel costs """
    if carrier == "hard_coal":
        carrierIdentifier = "Coal fired power plants"
    elif carrier == "natural_gas":
        carrierIdentifier = "Gas fired power plants (Natural gas, biogas)"
    elif carrier == "uranium":
        carrierIdentifier = "Nuclear power plants"
    elif carrier == "lignite":
        carrierIdentifier = "Lignite fired power plants"
    else:
        print("carrier {} not known. Skip cost calculation.".format(carrier))
        carrierIdentifier = None
    return carrierIdentifier
    
def getConstants(constant):
    """ returns constant value """
    constants = {
        "MWh2toe"           : 0.0859845,
        "yearFirstDemand"   : 2017,
        "maximumInvestYears": 5,
        "apiYear"           : 2019 # if EMHIRES database used, use year 2015
    }
    return constants[constant]

def getDemandDataframe(demandPath):
    """ either load pickle or load xlxs 
    :param demandPath: path where raw demand data is found
    :return demandHourly: hourly load profiles """
    # load European electricity demand
    # source: ENTSOE, MHLV_data-2015-2017_demand_hourly for 2017 (2015 incomplete, 2016 leap year)
    if not os.path.exists(demandPath / "demandHourly.pickle"):
        demandRawData = pd.read_excel(demandPath / "MHLV_data-2015-2017_demand_hourly.xlsx",sheet_name = "2015-2017")
        # correct time stamp
        correct_dateformat = "%d.%m.%Y %H:%M"
        demandRawData["DateUTC"] = demandRawData["DateUTC"].dt.strftime(correct_dateformat)
        demandHourly = demandRawData[pd.to_datetime(demandRawData["DateUTC"]).dt.year==getConstants("yearFirstDemand")]
        demandHourly.drop(["MeasureItem","DateShort","TimeFrom","TimeTo","Value","Cov_ratio"],axis=1,inplace=True)
        # rearrange index and drop duplicate indices
        demandHourly.set_index(["DateUTC","CountryCode"],inplace=True)
        demandHourly = demandHourly[~demandHourly.index.duplicated()]
        demandHourly = demandHourly.unstack()
        demandHourly.columns = demandHourly.columns.droplevel(0)
        with open(demandPath / "demandHourly.pickle","wb") as inputFile:
            pickle.dump(demandHourly,inputFile)
    else:
        with open(demandPath / "demandHourly.pickle","rb") as inputFile:
            demandHourly = pickle.load(inputFile)
    # do not use datetime index but (for the time being) range from 0-8759
    demandHourly        = demandHourly.reset_index(drop=True)
    # change Greece (GR) in (EL)
    demandHourly        = demandHourly.rename(columns={"GR":"EL","GB":"UK"})
    demandHourly        = demandHourly/1000 # GW
    return demandHourly