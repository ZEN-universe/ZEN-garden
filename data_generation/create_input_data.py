"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class containing the methods for the generation of the input dataset respecting the platform's data structure.
==========================================================================================================================================================================="""
import copy

import numpy as np
import pandas as pd
import os, sys
import functions.data_helpers as helpers
import functions.query_api_data as queryApi
from pathlib import Path
import pickle

class DataCreation():
    """ this class creates the NUTS0 data for the electricity sector """
    def __init__(self,folderName,sourceName):
        self.folderName = folderName
        self.sourceName = sourceName
        self.cwd        = Path(os.getcwd())
        self.folderPath = self.cwd / "data" / self.folderName
        # create new folder structure if it does not exist yet
        self.createNewFolder()
        # check if source path exists
        self.sourcePath = self.cwd.parent / self.sourceName
        assert os.path.exists(self.sourcePath), f"Source data folder {self.sourceName} does not exist in {self.cwd.parent}"
        # initialize extraction from ENTSO-E transparency platform
        self.useApiData = True
        self.apiData    = queryApi.ExtractApis(self,helpers)
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
        self.carriersInModel        = []
        self.carbonIntensityCarrier = {}
        # create nodes
        self.createNodes()
        # create conversion technologies
        self.createConversionTechnologies()
        # create transport technologies
        self.createTransportTechnologies()
        # create storage technologies
        self.createStorageTechnologies()
        # create carriers
        self.createCarriers()

    def createNodes(self):
        """ fills the setNodes folder"""
        assert os.path.exists(self.folderPath / "setNodes" / "setNodes.csv"), "setNodes not yet created! Implement!"
        self.setNodes = pd.read_csv(self.folderPath / "setNodes" / "setNodes.csv")
        self.createEdges()
        self.apiData.setNodesAndEdges(self.setNodes,self.setEdges)

    def createEdges(self):
        """ creates the edges """
        setEdges = pd.DataFrame(columns=["nodeFrom","nodeTo"])
        # nodes from ENTSOE TYNDP 2020-scenario.xlsx
        # load nodes and edges
        _nodes  = pd.read_csv(self.sourcePath / "nodesEdges" / "Nodes_Dict.csv",delimiter=";")
        _edges  = pd.read_csv(self.sourcePath / "nodesEdges" / "Lines_Dict.csv",delimiter=";")
        _edges  = _edges[~_edges["line_id"].str.contains("Exp")]
        # iterate through nodes to find corresponding edges
        setNodes = copy.deepcopy(self.setNodes["node"])
        setNodes[setNodes == "EL"]  = "GR"
        for _node in setNodes:
            _nodeIds = _nodes["node_id"][_nodes["country"]==_node].reset_index(drop=True)
            _connectedNodeIdFromNode  = []
            _connectedNodeIdToNode    = []
            for _nodeId in _nodeIds:
                _connectedNodeIdFromNode.extend(list(_edges["node_b"][(_edges["node_a"] == _nodeId)]))
                _connectedNodeIdToNode.extend(list(_edges["node_a"][(_edges["node_b"] == _nodeId)]))
            # append to setEdges
            _setEdgesTempFromNode               = pd.DataFrame(columns=["nodeFrom","nodeTo"])
            _connectedNodeFromNode              = _nodes["country"][_nodes["node_id"].apply(lambda node: node in _connectedNodeIdFromNode)]
            # only nodes which are in self.setNodes
            _setEdgesTempFromNode["nodeTo"]     = _connectedNodeFromNode[_connectedNodeFromNode.isin(setNodes)]
            _setEdgesTempFromNode["nodeFrom"]   = _node
            _setEdgesTempToNode                 = pd.DataFrame(columns=["nodeFrom", "nodeTo"])
            _connectedNodeToNode                = _nodes["country"][_nodes["node_id"].apply(lambda node: node in _connectedNodeIdToNode)]
            # only nodes which are in self.setNodes
            _setEdgesTempToNode["nodeFrom"]     = _connectedNodeToNode[_connectedNodeToNode.isin(setNodes)]
            _setEdgesTempToNode["nodeTo"]       = _node
            setEdges            = pd.concat([setEdges,_setEdgesTempFromNode,_setEdgesTempToNode])
        # remove edges where nodeFrom = nodeTo
        setEdges                    = setEdges[setEdges["nodeFrom"] != setEdges["nodeTo"]]
        # flip direction of edge
        setEdgesFlipped             = setEdges.rename(columns={"nodeFrom":"nodeTo","nodeTo":"nodeFrom"})
        setEdges                    = pd.concat([setEdges,setEdgesFlipped]).drop_duplicates()
        # substitute greece
        setEdges[setEdges == "GR"]  = "EL"
        # set edge name
        setEdges["edge"]    = setEdges.apply(lambda row: row["nodeFrom"]+"-"+row["nodeTo"],axis=1)
        setEdges            = setEdges.set_index("edge")
        self.setEdges       = setEdges
        # write csv
        setEdges.to_csv(self.folderPath / "setNodes" / "setEdges.csv")


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
            # create PWA files
            self.createPWAfiles(conversionTechnology)
            # create PV and wind onshore maxLoad files
            self.createMaxLoadFiles(conversionTechnology)
            # extract historic generation profile
            self.extractHistoricGeneration(conversionTechnology)
            # extract existing capacity
            self.extractExistingCapacity(conversionTechnology)

    def createTransportTechnologies(self):
        """ fills the setTransportTechnologies folder"""
        transportTechnologies = helpers.getTechnologies(self.sourcePath,"transport")
        for transportTechnology in transportTechnologies:
            self.createNewElementFolder("setTransportTechnologies",transportTechnology)
            # create attribute file
            self.createDefaultAttributeDataFrame("setTransportTechnologies",transportTechnology)
            # create distance matrix
            self.createDistanceMatrix(transportTechnology)
            # create capacityLimit matrix
            self.createCapacityLimitTransportTechnology(transportTechnology)
    
    def createStorageTechnologies(self):
        """ fills the setStorageTechnologies folder"""
        storageTechnologies = helpers.getTechnologies(self.sourcePath,"storage")
        for storageTechnology in storageTechnologies:
            self.createNewElementFolder("setStorageTechnologies",storageTechnology)
            # create attribute file
            self.createDefaultAttributeDataFrame("setStorageTechnologies",storageTechnology)

    def createCarriers(self):
        """ fills the setCarriers folder"""
        # load fuel prices 
        # source: POTEnCIA energy model, 2018 EU28 Fuel Prices
        if not os.path.exists(self.sourcePath / "technologyAttributes" / "fuelPricesPotencia.pickle"):
            self.fuelPricesPotencia = pd.read_excel(self.sourcePath / "technologyAttributes" / "PG_technology_Central_2018.xlsx", sheet_name = "fuel_costs").set_index("International fuel prices (€2010/toe)")[2018]
            with open(self.sourcePath / "technologyAttributes" / "fuelPricesPotencia.pickle","wb") as inputFile:
                pickle.dump(self.fuelPricesPotencia,inputFile)
        else:
            with open(self.sourcePath / "technologyAttributes" / "fuelPricesPotencia.pickle","rb") as inputFile:
                self.fuelPricesPotencia = pickle.load(inputFile)
        # load European demand
        # source: ENTSOE, MHLV_data-2015-2017_demand_hourly for 2017 (2015 incomplete, 2016 leap year)
        self.electricityDemand = helpers.getDemandDataframe(self.sourcePath / "demand")

        for carrier in self.carriersInModel:
            self.createNewElementFolder("setCarriers",carrier)
            # create attribute file
            self.createDefaultAttributeDataFrame("setCarriers",carrier)
            # create demand
            self.createDemand(carrier)
    
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
        dfAttribute             = pd.DataFrame(index = helpers.getAttributesOfSet(setName),columns=["value","unit"])
        dfAttribute.index.name  = "index"
        dfAttribute["value"]    = dfAttribute.index.map(lambda index: helpers.getDefaultValue(index))
        dfAttribute["unit"]     = dfAttribute.index.map(lambda index: helpers.getDefaultUnit(index))
        if setName == "setConversionTechnologies":
            # overwrite minLoad, if no min load, set to 0
            if not helpers.getTechnologies(self.sourcePath,"conversion","minLoad")[elementName]:
                dfAttribute.loc["minLoadDefault","value"]   = 0
            # input and output carrier
            _inputCarrier,_outputCarrier                    = self.getInputOutputCarrier(elementName)
            dfAttribute.loc["inputCarrier","value"],dfAttribute.loc["outputCarrier","value"]    = _inputCarrier,_outputCarrier
            dfAttribute.loc["inputCarrier","unit"],dfAttribute.loc["outputCarrier", "unit"]     = helpers.getCarrierUnits(_inputCarrier), helpers.getCarrierUnits(_outputCarrier)
            # Potencia assumptions
            _potenciaAssumptions                                = self.technologyPotenciaAssumption.loc[helpers.getAttributeIndex(self.conversionTechnologiesPotencia[elementName])]
            dfAttribute.loc["lifetimeDefault","value"]          = _potenciaAssumptions["Technical lifetime (years)"]
            _maximumNumberNewPlants                             = helpers.getNumberOfNewPlants(elementName) # TODO choose sensible number
            dfAttribute.loc["maxBuiltCapacityDefault","value"]  = _potenciaAssumptions["Typical unit size of a new power plant (kW)"]/1e6*_maximumNumberNewPlants   # GW
            dfAttribute.loc["opexSpecificDefault","value"]      = _potenciaAssumptions["Variable O&M  costs €2010/MWh"]                                             # kEUR/GWh
            # save carbon intensity of input carrier
            if _inputCarrier:
                self.carbonIntensityCarrier[_inputCarrier]      = _potenciaAssumptions["Default emissions factor (t of CO2 / toe input)"]*helpers.getConstants("MWh2toe") #ktCO2/GWh

        elif setName == "setTransportTechnologies":
            dfAttribute = helpers.setManualAttributesTransport(elementName,dfAttribute)
        elif setName == "setStorageTechnologies":
            dfAttribute = helpers.setManualAttributesStorage(elementName,dfAttribute)
        elif setName == "setCarriers":
            # set fuel prices
            _carrierIdentifier = helpers.getCarrierIdentifier(elementName)
            if _carrierIdentifier:
                dfAttribute.loc["importPriceCarrierDefault","value"]    = self.fuelPricesPotencia[_carrierIdentifier]*helpers.getConstants("MWh2toe")           # kEUR/GWh
            # carbon intensity
            if elementName in self.carbonIntensityCarrier:  
                dfAttribute.loc["carbonIntensityDefault","value"]       = self.carbonIntensityCarrier[elementName]                                              # ktCO2/GWh
            # manual attributes
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
    
    def createDemand(self,carrier):
        """ creates demand file for selected carriers 
        :param carrier: carrier in model """
        if carrier == "electricity":
            if self.useApiData:
                electricityDemandSelection = self.apiData.getEntsoeDemand()
            else:
                commonCountries = set(self.electricityDemand.columns).intersection(self.setNodes["node"])
                missingCountries = list(set(self.setNodes["node"]).difference(commonCountries))
                if missingCountries:
                    print(f"electricity demand missing for countries {missingCountries}. Default demand is used.")
                electricityDemandSelection = self.electricityDemand[sorted(commonCountries)]
                electricityDemandSelection.index.name = "time"
            electricityDemandSelection.to_csv(self.folderPath / "setCarriers" / carrier / "demandCarrier.csv")

    def createPWAfiles(self,elementName):
        """ creates PWA for conversion technologies """
        _potenciaAssumptions            = self.technologyPotenciaAssumption.loc[helpers.getAttributeIndex(self.conversionTechnologiesPotencia[elementName])]
        minCapacity                     = 0
        _maximumNumberNewPlants         = helpers.getNumberOfNewPlants(elementName) # TODO choose sensible number
        maxBuiltCapacity                = _potenciaAssumptions["Typical unit size of a new power plant (kW)"]/1e6*_maximumNumberNewPlants   # GW
        maxTotalCapacity                = maxBuiltCapacity*helpers.getConstants("maximumInvestYears")
        # carriers
        _inputCarrier                   = helpers.setInputOutputCarriers(elementName,"input")
        _outputCarrier                  = helpers.setInputOutputCarriers(elementName,"output")
        _referenceCarrier               = helpers.setInputOutputCarriers(elementName,"reference")
        _carriers                       = set()
        if _inputCarrier:
            _carriers                   = _carriers.union(set([_inputCarrier]))
        if _outputCarrier:
            _carriers                   = _carriers.union(set([_outputCarrier]))
        _dependentCarrier               = list(_carriers-set([_referenceCarrier]))
        assert len(_dependentCarrier) <= 1, f"Not yet implemented for technologies ({elementName}) with more than 1 dependent carrier {_dependentCarrier}"
        # create csv
        # capex
        dfCapex                         = pd.DataFrame([minCapacity,maxBuiltCapacity],columns=["capacity"])
        dfCapex["capex"]                = _potenciaAssumptions["Capital costs €2010/kW"] * 1000  # kEUR/GW
        # append units
        dfCapex.loc[len(dfCapex.index)] = ["GW","kiloEuro/GW"]
        # save as csv
        dfCapex["capacity"].to_csv(self.folderPath / "setConversionTechnologies" / elementName / "breakpointsPWACapex.csv",index = False)
        dfCapex.to_csv(self.folderPath / "setConversionTechnologies" / elementName / "nonlinearCapex.csv",index = False)
        # converEfficiency
        dfConverEfficiency              = pd.DataFrame([minCapacity,maxTotalCapacity],columns=[_referenceCarrier])
        if len(_dependentCarrier) == 1:
            dfConverEfficiency[_dependentCarrier[0]]            = dfConverEfficiency[_referenceCarrier]/_potenciaAssumptions["Net efficiency"]
        # append units
        dfConverEfficiency.loc[len(dfConverEfficiency.index)]   = "GW"
        # save as csv
        dfConverEfficiency[_referenceCarrier].to_csv(self.folderPath / "setConversionTechnologies" / elementName / "breakpointsPWAConverEfficiency.csv",index = False)
        dfConverEfficiency.to_csv(self.folderPath / "setConversionTechnologies" / elementName / "nonlinearConverEfficiency.csv",index = False)

    def createMaxLoadFiles(self,elementName):
        """ creates maxLoad files for photovoltaics and wind onshore conversion technologies """
        if elementName == "photovoltaics" or elementName == "wind_onshore":
            if self.useApiData:
                maxLoad = self.apiData.getRenewableNinjaData(elementName)
            else:
                # if already converted
                if os.path.exists(self.sourcePath / "maxLoad" / f"maxLoad_{elementName}.pickle"):
                    with open(self.sourcePath / "maxLoad" / f"maxLoad_{elementName}.pickle", "rb") as input_file:
                        maxLoad = pickle.load(input_file)
                else:
                    # elementName specific
                    if elementName == "photovoltaics":
                        # from EMHIRESPV_TSh_CF_Country_only2015, reduced version
                        maxLoad = pd.read_excel(self.sourcePath / "maxLoad" / "EMHIRESPV_TSh_CF_Country_only2015.xlsx").set_index("Date")
                    elif elementName == "wind_onshore":
                        # from EMHIRES_WIND_COUNTRY_only2015_June2019, reduced version
                        maxLoad = pd.read_excel(self.sourcePath / "maxLoad" / "EMHIRES_WIND_COUNTRY_only2015_June2019.xlsx").set_index("Date")
                    # only select specific countries
                    commonCountries = set(maxLoad.columns).intersection(self.setNodes["node"])
                    missingCountries = list(set(self.setNodes["node"]).difference(commonCountries))
                    if missingCountries:
                        print(f"MaxLoad for {elementName} missing for countries {missingCountries}. Default maxLoad is used.")
                    maxLoad = maxLoad[sorted(commonCountries)]
                    # dump pickle
                    with open(self.sourcePath / "maxLoad" / f"maxLoad_{elementName}.pickle", "wb") as input_file:
                        pickle.dump(maxLoad , input_file)
            # do not use datetime index but (for the time being) range from 0-8759
            maxLoad             = maxLoad.reset_index(drop=True)
            # create csv
            maxLoad.index.name  = "time"
            maxLoad.to_csv(self.folderPath / "setConversionTechnologies" / elementName / "maxLoad.csv")

    def extractHistoricGeneration(self,elementName):
        """ extracts historic entsoe generation profiles """
        historicGeneration = self.apiData.getEntsoeGeneration(elementName)
        historicGeneration.to_csv(self.folderPath / "setConversionTechnologies" / elementName / "historicGeneration.csv")

    def extractExistingCapacity(self,elementName):
        """ extracts existing capacity from  ENTSOE"""
        existingCapacity = self.apiData.getExistingCapacity(elementName)
        existingCapacity.to_csv(
            self.folderPath / "setConversionTechnologies" / elementName / "existingCapacity.csv")
    def createDistanceMatrix(self,elementName):
        """
        Compute a matrix containing the distance between any two points in the domain based on the Euclidean distance
        """

        def f_eucl_dist(P0, P1):
            """
                Compute the Eucledian distance of two points in 2D
            """
            return ((P0[0] - P1[0]) ** 2 + (P0[1] - P1[1]) ** 2) ** 0.5

        setNodes    = self.setNodes.set_index("node")
        nodes       = setNodes.index
        xArr        = setNodes["x"]
        yArr        = setNodes["y"]
        # create empty distance matrix
        distanceMatrix              = pd.DataFrame(index = self.setEdges.index, columns= ["distance"])
        # calculate distance
        distanceMatrix["distance"]  = self.setEdges.apply(lambda row: f_eucl_dist((xArr[row["nodeFrom"]], yArr[row["nodeFrom"]]),(xArr[row["nodeTo"]], yArr[row["nodeTo"]])),axis=1)
        distanceMatrix.to_csv(self.folderPath / "setTransportTechnologies" / elementName / "distanceEuclidean.csv")

    def createCapacityLimitTransportTechnology(self,elementName):
        """ extract the capacity limit of the transmission technology lines """
        if elementName == "power_line":
            capacityLimit = self.apiData.getEntsoeTransmissionCapacity()
            capacityLimit.to_csv(self.folderPath / "setTransportTechnologies" / elementName / "capacityLimit.csv")

def main():
    """ This is the main function to create NUTS0 """
    # set folder name
    folderName = "NUTS0_electricity"
    sourceName = "NUTS0_Source_Data"
    # enable or disable creation scripts.
    DataCreation(folderName = folderName,sourceName = sourceName)

if __name__ == "__main__":
    main()
