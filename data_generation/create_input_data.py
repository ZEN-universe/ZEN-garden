"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class containing the methods for the generation of the input dataset respecting the platform's data structure.
==========================================================================================================================================================================="""
import numpy as np
import pandas as pd
import os, sys
import functions.data_helpers as helpers
from pathlib import Path
import pickle

class DataCreation():
    """ this class creates the NUTS0 data for the electricity sector """
    def __init__(self,folderName,sourceName):
        self.folderName = folderName
        self.sourceName = sourceName
        self.cwd = Path(os.getcwd())
        self.folderPath = self.cwd / "data" / self.folderName
        # create new folder structure if it does not exist yet
        self.createNewFolder()
        # check if source path exists
        self.sourcePath = self.cwd.parent / self.sourceName
        assert os.path.exists(self.sourcePath), f"Source data folder {self.sourceName} does not exist in {self.cwd.parent}" 
        # create input data
        self.createInputData()

    def createNewFolder(self):
        """ creates new folder structure for input data """
        if not os.path.exists(self.folderPath):
            # create top level folder
            os.mkdir(self.folderPath)
            # create subfolders
            subfolders = helpers.getFolderNames()
            for subfolder in subfolders:
                os.mkdir(f"{self.folderPath}/{subfolder}")
            print(f"New folder structure created for {self.folderName}")

    def createInputData(self):
        """ extracts input data structure from input """
        self.carriersInModel = []
        # create nodes
        self.createNodes()
        # create time steps
        self.createTimeSteps()
        # create conversion technologies
        self.createConversionTechnologies()
        # create transport technologies
        self.createTransportTechnologies()
        # create storage technologies
        self.createStorageTechnologies()
        # create carriers
        self.createCarriers()
        
        a=1

    def createNodes(self):
        """ fills the setNodes folder"""
        pass

    def createTimeSteps(self):
        """ fills the setTimeSteps folder"""
        pass

    def createConversionTechnologies(self):
        """ fills the setConversionTechnologies folder"""
        self.conversionTechnologiesPotencia = helpers.getTechnologies(self.sourcePath,"conversion","nameJRC")
        # source: POTEnCIA energy model, 2018 EU28 Technology assumptions
        if not os.path.exists(self.sourcePath / "technologyAttributes" / "technologyPotenciaAssumption.pickle"):
            self.technologyPotenciaAssumption = pd.read_excel(self.sourcePath / "technologyAttributes" / "PG_technology_Central_2018.xlsx", sheet_name = "tech_base").set_index(["Type","Technology","Co-generation","Size"])
            with open(self.sourcePath / "technologyAttributes" / "technologyPotenciaAssumption.pickle","wb") as inputFile:
                pickle.dump(self.technologyPotenciaAssumption,inputFile)
        else:
            with open(self.sourcePath / "technologyAttributes" / "technologyPotenciaAssumption.pickle","rb") as inputFile:
                self.technologyPotenciaAssumption = pickle.load(inputFile)

        conversionTechnologies = list(self.conversionTechnologiesPotencia.keys())
        for conversionTechnology in conversionTechnologies:
            self.createNewElementFolder("setConversionTechnologies",conversionTechnology)
            # create attribute file
            self.createDefaultAttributeDataFrame("setConversionTechnologies",conversionTechnology)
    
    def createTransportTechnologies(self):
        """ fills the setTransportTechnologies folder"""
        transportTechnologies = helpers.getTechnologies(self.sourcePath,"transport")
        for transportTechnology in transportTechnologies:
            self.createNewElementFolder("setTransportTechnologies",transportTechnology)
            # create attribute file
            self.createDefaultAttributeDataFrame("setTransportTechnologies",transportTechnology)
    
    def createStorageTechnologies(self):
        """ fills the setStorageTechnologies folder"""
        storageTechnologies = helpers.getTechnologies(self.sourcePath,"storage")
        for storageTechnology in storageTechnologies:
            self.createNewElementFolder("setStorageTechnologies",storageTechnology)
            # create attribute file
            self.createDefaultAttributeDataFrame("setStorageTechnologies",storageTechnology)

    def createCarriers(self):
        """ fills the setCarriers folder"""
        a=1
        pass
    
    def createNewElementFolder(self,setName,elementName):
        """ this method creates a new folder for an element 
        :param setName: name of set in which element is created
        :param elementName: name of element """
        # if element does not exist, create new folder
        if not os.path.exists(self.folderPath / setName / elementName):
            os.mkdir(self.folderPath / setName / elementName)

    def createDefaultAttributeDataFrame(self,setName,elementName):
        """ this method creates a default attribute DataFrame
        :param setName: name of set to which element belongs
        :param elementName: name of element """
        dfAttribute                                 = pd.DataFrame(index = helpers.getAttributesOfSet(setName),columns=["attributes"])
        dfAttribute.index.name                      = "index"
        dfAttribute["attributes"]                   = dfAttribute.index.map(lambda index: helpers.getDefaultValue(index))
        if setName == "setConversionTechnologies":
            # input and output carrier
            dfAttribute.loc["inputCarrier"],dfAttribute.loc["outputCarrier"] = self.getInputOutputCarrier(elementName)
            # Potencia assumptions
            _potenciaAssumptions                    = self.technologyPotenciaAssumption.loc[helpers.getCostIndex(self.conversionTechnologiesPotencia[elementName])]
            dfAttribute.loc["lifetime"]             = _potenciaAssumptions["Technical lifetime (years)"]
            _maximumNumberNewPlants                 = helpers.getNumberOfNewPlants(elementName) # TODO choose sensible number
            dfAttribute.loc["maxBuiltCapacity"]     = _potenciaAssumptions["Typical unit size of a new power plant (kW)"]/1000*_maximumNumberNewPlants
            dfAttribute.loc["opexSpecificDefault"]  = _potenciaAssumptions["Variable O&M  costs €2010/MWh"]
            # dfAttribute["carbonIntensityDefault"]   = (_potenciaAssumptions["Default emissions factor (t of CO2 / toe input)"] #tCO2/MWh output
            #                                         /_potenciaAssumptions["Net efficiency"]
            #                                         *helpers.getConstants("MWh2toe"))
        elif setName == "setTransportTechnologies":
            dfAttribute = helpers.setManualAttributesTransport(elementName,dfAttribute)
        elif setName == "setStorageTechnologies":
            dfAttribute = helpers.setManualAttributesStorage(elementName,dfAttribute)
        elif setName == "setCarriers":
            dfAttribute = helpers.setManualAttributesCarriers(elementName,dfAttribute)
        # write csv
        dfAttribute.to_csv(self.folderPath / setName / elementName / "attributes.csv")

    def getInputOutputCarrier(self,elementName):
        """ retrieves input and output carriers of conversion technology """
        _inputCarrier   = helpers.setInputOutputCarriers(elementName,"input")
        _outputCarrier  = helpers.setInputOutputCarriers(elementName,"output")
        # append carriers to list
        if _inputCarrier:
            for _carrier in _inputCarrier.split(" "):
                if _carrier not in self.carriersInModel:
                    self.carriersInModel.append(_carrier)
        if _outputCarrier:
            for _carrier in _outputCarrier.split(" "):
                if _carrier not in self.carriersInModel:
                    self.carriersInModel.append(_carrier)
        return _inputCarrier,_outputCarrier
        
def main():
    """ This is the main function to create NUTS0 """
    # set folder name
    folderName = "NUTS0_electricity"
    sourceName = "NUTS0_Source_Data"
    # enable or disable creation scripts. 
    DataCreation(folderName = folderName,sourceName = sourceName)

if __name__ == "__main__":
    main()

####       
data = dict()
data['mainFolder'] = 'NUTS2'
data['sourceData'] = pd.read_csv('{}.csv'.format(data['mainFolder']), header=0, index_col=None)
data['PWAData'] = pd.read_csv('PWA.csv', header=0, index_col=None)
data['NLData'] = pd.read_csv('NL.csv', header=0, index_col=None)
Create = Create(data, analysis)

###                                                                                                                  ###
# Data are added in the following ways:
# 1. source data file defined in dictionary data: the script looks for the items in headerInSource among the columns
#   of the source data file
# 2. input data in the dictionary <data>
# 3. Any other required input data from <analysis['headerDataInputs']> is assigned to zero
###                                                                                                                  ###

data['scenario'] = ['a']
data['time'] = [0]
### Nodes
headerInSource = {'node': 'ID', 'x': "('X', 'km')", 'y': "('Y', 'km')"}
Create.independentData('Nodes', headerInSource)
### Scenarios
headerInSource = {}
Create.independentData('Scenarios', headerInSource)
### TimeSteps
headerInSource = {}
Create.independentData('TimeSteps', headerInSource)

# compute Euclidean matrix
Create.distanceMatrix('euclidean')

### Carriers
# Inputs from file
headerInSource = {
    'electricity': {'availabilityCarrier':"('TotalGreen_potential', 'MWh')"},
    'water': {},
    'biomass': {'availabilityCarrier':"('Biomass_potential', 'MWh')"},
    'natural_gas': {},
    'hydrogen': {'demandCarrier': "('hydrogen_demand', 'MWh')"},
    'oxygen': {}
    }

## Manual inputs
# biomass
data['importPriceCarrier_biomass'] = [0.07*1000] # EUR/MWh, value from Gabrielli et al. - dry biomass
# electricity
data['exportPriceCarrier_electricity'] = [0.05*1000] # EUR/MWh, value from Gabrielli et al.
data['importPriceCarrier_electricity'] = [0.12*1000] # EUR/MWh, value from Gabrielli et al.
# natural gas
data['importPriceCarrier_natural_gas'] = [0.06*1000] # EUR/MWh, value from Gabrielli et al.
# water
data['availabilityCarrier_water'] = [1e12] # unlimited availability
## create datasets
Create.nodalData('Carriers', headerInSource)

## attributes - customised dataframe
inputDataFrame = {
    'biomass':
        {'carbon_intensity': 13/1000   # tonCO2/MWh, value from Gabrielli et al. - dry biomass
         },
    'electricity':
        {'carbon_intensity': 127/1000  # tonCO2/MWh, value from Gabrielli et al.
         },
    'natural_gas':
        {'carbon_intensity': 237/1000  # tonCO2/MWh, value from Gabrielli et al.
         },
    'oxygen':
        {'carbon_intensity': 0
         },
    'hydrogen':
        {'carbon_intensity': 0
         },
}
## create datasets
Create.attributesDataFrame('Carriers', inputDataFrame)

## ConversionTechnologies
# nodal data
headerInSource = {
    'electrolysis': {},
    'SMR':{}
}
data['availability_electrolysis'] = [1e5]   # MW, value from Gabrielli et al
data['availability_SMR'] = [1e5]            # MW, value from Gabrielli et al
Create.nodalData('ConversionTechnologies', headerInSource)
# attributes
inputDataFrame = {
    'electrolysis': {
        'minBuiltCapacity':0,
        'maxBuiltCapacity':data['NLData'][['capacity_capex_electrolysis']].dropna().values[-1][0],
        'minLoad':0.07,
        'lifetime':10, # h, value from Gabrielli et al
        'costVariable':10*10**6,
        'referenceCarrier':'hydrogen',
        'inputCarrier':'electricity water',
        'outputCarrier':'hydrogen oxygen'},
    'SMR': {
        'minBuiltCapacity':1, # MW, value from Gabrielli et al
        'maxBuiltCapacity':1,
        'lifetime': 20, # h, value from Gabrielli et al
        'minLoad':0.1,
        'maxLoad':1,
     'referenceCarrier':'hydrogen', 'inputCarrier':'natural_gas', 'outputCarrier':'hydrogen carbon_dioxide'},
}
Create.attributesDataFrame('ConversionTechnologies', inputDataFrame)

# files variable approximation
inputDataFrame = {
    'electrolysis': {
        'breakpointsPWACapex':{
            'columns':['capacity'],
            'values':data['PWAData']['breakpoints_capex_electrolysis'].dropna().values
        },
        'breakpointsPWAConverEfficiency':{
            'columns':['hydrogen'],
            'values':data['PWAData']['breakpoints_efficiency_electrolysis'].dropna().values
        },
        'nonlinearCapex':{
            'columns':['capacity', 'capex'],
            'values':data['NLData'][['capacity_capex_electrolysis', 'capex_capex_electrolysis']].dropna().values
        },
        'nonlinearConverEfficiency':{
            'columns':['hydrogen','oxygen','electricity', 'water'],
            'values':data['NLData'][['hydrogen_efficiency_electrolysis', 'hydrogen_efficiency_electrolysis',
                                     'electricity_efficiency_electrolysis', 'water_efficiency_electrolysis']].dropna().values
        }
    },
    'SMR': {
        'breakpointsPWACapex':{
            'columns':['capacity'],
            'values':data['PWAData']['breakpoints_capex_SMR'].dropna().values
        },
        'breakpointsPWAConverEfficiency':{
            'columns':['hydrogen'],
            'values':data['PWAData']['breakpoints_efficiency_SMR'].dropna().values
        },
        'nonlinearCapex':{
            'columns':['capacity', 'capex'],
            'values':data['NLData'][['capacity_capex_SMR', 'capex_capex_SMR']].dropna().values
        },
        'nonlinearConverEfficiency':{
            'columns':['hydrogen','natural_gas','carbon_dioxide',],
            'values':data['NLData'][['hydrogen_efficiency_SMR', 'natural_gas_efficiency_SMR',
                                     'carbon_dioxide_efficiency_SMR']].dropna().values
        }
    },
}
Create.generalDataFrame('ConversionTechnologies', inputDataFrame)

## TransportTechnologies
# datasets based on nodes combination
headerInSource = {
    'pipeline_hydrogen': {},
    'truck_hydrogen': {}
}
data['availability_pipeline_hydrogen'] = [85] # MW, value from Gabrielli et al.
data['availability_truck_hydrogen'] = [38] # MW, value from Gabrielli et al.
data['distanceEuclidean_pipeline_hydrogen'] = Create.eucledian_distance
data['distanceEuclidean_truck_hydrogen'] = Create.eucledian_distance
data['efficiencyPerDistance_pipeline_hydrogen'] = [1]
data['efficiencyPerDistance_truck_hydrogen'] = [1]
# cost per distance dependent on the way the total capex is computed: 1/2 factor to avoid accounting a connection twice
data['costPerDistance_pipeline_hydrogen'] = [(8.2*10**3)/2] # EUR/km/MW, value fixed cost from Gabrielli et al.
Create.edgesData('TransportTechnologies', headerInSource)

# attributes
inputDataFrame = {
    'pipeline_hydrogen': {
        'minBuiltCapacity':1.6, # MW, value from Gabrielli et al.
        'maxBuiltCapacity':1e12,
        'minLoad':0,
        'lifetime':8760*50, # h, value from Gabrielli et al.
        'costVariable':1.6*10**(-6), # EUR/MW, value from Gabrielli et al.
        'lossFlow':0.00012, # 1/km, value from Gabrielli et al.
        'referenceCarrier':'hydrogen'},
    'truck_hydrogen': {
        'minBuiltCapacity': 1.6,  # MW, value from Gabrielli et al.
        'maxBuiltCapacity': 1e12,
        'minLoad': 0,
        'lifetime': 8760*10,  # h, value from Gabrielli et al.
        'carbon_intensity': 4.2*10**(-6),  # tonCO2eq/km/MWh, value from Gabrielli et al.
        'costVariable':1.6*10**(-5),  # EUR/MW, value from Gabrielli et al.
        'costFixed': 13*10**3,  # EUR/MW, value from Gabrielli et al.
        'lossFlow': 0.00012,  # 1/km, value from Gabrielli et al.
        'referenceCarrier': 'hydrogen'},
}
Create.attributesDataFrame('TransportTechnologies', inputDataFrame)