"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      January-2022
Authors:      Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Functions to extract the input data from the ENTSO-E Transparency Database
==========================================================================================================================================================================="""
from entsoe import EntsoeRawClient,EntsoePandasClient
from entsoe.mappings import lookup_area, Area
from entsoe.parsers import parse_loads, parse_generation
import pandas as pd
import numpy as np
import os
import pickle
import requests
import json

class ExtractApis():
    """ this class queries the ENTSO-E transparency platform and extracts the necessary data """
    def __init__(self,dataCreation,helpers):
        """ this method initializes the extraction object """
        self.dataCreation   = dataCreation
        self.helpers        = helpers
        self.nodes          = None
        # toggle extraction of data
        self.useExistingDemand          = True
        self.useExistingGeneration      = True
        self.useExistingCapacityFactor  = True
        # create entsoe client
        self.createApiClient()
        # calculate time interval for which the data is extracted
        self.createTimeInterval()

    def createApiClient(self):
        """ this method creates the entsoe and renewable.ninja clients """
        self.sourcePath = self.dataCreation.sourcePath
        # api tokens
        apiTokens       = {
            "entsoe":           "00a0c1f2-5b7e-431d-8106-45e020c9c229",
            "renewablesNinja": "63d9a2d5e3106dee2fc7f84c24b18668fe7eae6d"
        }
        # create entsoe client
        self.entsoeClient           = EntsoeRawClient(apiTokens["entsoe"])
        self.entsoeClientPd         = EntsoePandasClient(apiTokens["entsoe"])
        # create renewables.ninja client
        self.ninjaApiBase           = 'https://www.renewables.ninja/'
        self.ninjaClient            = requests.session()
        self.ninjaClient.headers    = {'Authorization': 'Token ' + apiTokens["renewablesNinja"]}

    def createTimeInterval(self):
        """ this method sets a year and then creates the time interval for which the data is extracted """
        # set year for which data is extracted
        self.apiYear    = self.helpers.getConstants("apiYear")
        self.timeStart  = pd.Timestamp(year=self.apiYear,month=1,day=1,hour=0, tz='Europe/Brussels')
        self.timeEnd    = pd.Timestamp(year=self.apiYear+1,month=1,day=1,hour=0, tz='Europe/Brussels')
        self.timeRange  = pd.date_range(self.timeStart,self.timeEnd,freq="H")[:-1]

    def setNodes(self,setNodes):
        """ this method sets the nodes to the extraction object """
        self.nodes = setNodes

    def getEntsoeDemand(self):
        """ this method gets the electricity demand """
        if not os.path.exists(self.sourcePath / "demand" / "demandENTSOE.pickle") or not self.useExistingDemand:
            self.entsoeDemand = pd.DataFrame(index=self.timeRange, columns=self.nodes["node"])
            areaNames  = [area.name for area in Area]
            for idx,node in enumerate(self.nodes["node"]):
                print(f"Extract demand for {node} - {idx+1}/{len(self.nodes['node'])}")
                if node == "EL":
                    nodeEntsoe = "GR"
                else:
                    nodeEntsoe = node
                assert nodeEntsoe in areaNames, f"node {nodeEntsoe} not in ENTSO-E area names"
                area    = lookup_area(nodeEntsoe)
                # query entsoe database, for some reason, the EntsoePandasClient deletes a lot of values
                try:
                    text = self.entsoeClient.query_load(country_code=area, start=self.timeStart, end=self.timeEnd)
                except:
                    try:
                        text = self.entsoeClient.query_load_forecast(country_code=area, start=self.timeStart, end=self.timeEnd)
                        print(f"Forecasted demand data used for {node}")
                    except:
                        print(f"No matching demand data found for {node}")
                        continue
                demand  = parse_loads(text, process_type='A16')
                demand  = demand.tz_convert(area.tz)
                demand  = demand.truncate(before=self.timeStart, after=self.timeEnd)
                # resample to hourly resolution
                demand  = demand.resample('h').mean()
                self.entsoeDemand[node] = demand
            # dump pickle
            with open(self.sourcePath / "demand" / "demandENTSOE.pickle","wb") as inputFile:
                pickle.dump(self.entsoeDemand, inputFile)
        else:
            with open(self.sourcePath / "demand" / "demandENTSOE.pickle","rb") as inputFile:
                self.entsoeDemand = pickle.load(inputFile)
        # set zeros to nan and interpolate nans
        self.entsoeDemand[self.entsoeDemand==0] = np.nan
        self.entsoeDemand = self.entsoeDemand.interpolate()
        # fill remaining nans (where entire time series is nan - MT) with 0
        self.entsoeDemand = self.entsoeDemand.fillna(0)
        # convert to GW
        self.entsoeDemand = self.entsoeDemand/1000
        # drop index
        self.entsoeDemand = self.entsoeDemand.reset_index(drop=True)
        self.entsoeDemand.index.name = "time"
        return self.entsoeDemand

    def getEntsoeGeneration(self,technology):
        """ this method gets the electricity generation of each technology """
        if not os.path.exists(self.sourcePath / "historicGeneration" / f"generationENTSOE{technology}.pickle") or not self.useExistingGeneration:
            entsoeGeneration = pd.DataFrame(index=self.timeRange, columns=self.nodes["node"])
            areaNames  = [area.name for area in Area]
            for idx,node in enumerate(self.nodes["node"]):
                print(f"Extract historic generation for {node} of technology {technology} - {idx+1}/{len(self.nodes['node'])}")
                if node == "EL":
                    nodeEntsoe = "GR"
                else:
                    nodeEntsoe = node
                assert nodeEntsoe in areaNames, f"node {nodeEntsoe} not in ENTSO-E area names"
                area    = lookup_area(nodeEntsoe)
                psr     = self.helpers.getEntsoeTechnologyIdentifier(technology)
                # query entsoe database, for some reason, the EntsoePandasClient deletes a lot of values
                try:
                    text = self.entsoeClient.query_generation(country_code=area, start=self.timeStart, end=self.timeEnd,psr_type = psr)
                except:
                    try:
                        text = self.entsoeClient.query_generation_forecast(country_code=area, start=self.timeStart, end=self.timeEnd,psr_type = psr)
                        print(f"Forecasted generation data used for {technology} on {node}")
                    except:
                        print(f"No matching generation data found for {technology} on {node}")
                        continue
                generation  = parse_generation(text, nett=True) # nett: condense generation and consumption into one
                generation  = generation.tz_convert(area.tz)
                generation  = generation.truncate(before=self.timeStart, after=self.timeEnd)
                # resample to hourly resolution
                generation  = generation.resample('h').mean()
                entsoeGeneration[node] = generation
            # dump pickle
            with open(self.sourcePath / "historicGeneration" / f"generationENTSOE{technology}.pickle","wb") as inputFile:
                pickle.dump(entsoeGeneration, inputFile)
        else:
            with open(self.sourcePath / "historicGeneration" / f"generationENTSOE{technology}.pickle","rb") as inputFile:
                entsoeGeneration = pickle.load(inputFile)
        # interpolate nans
        entsoeGeneration = entsoeGeneration.interpolate()
        # convert to GW
        entsoeGeneration = entsoeGeneration/1000
        # drop index
        entsoeGeneration = entsoeGeneration.reset_index(drop=True)
        entsoeGeneration.index.name = "time"
        return entsoeGeneration

    def getRenewableNinjaData(self,technology):
        """ this method extracts the data from the renewable ninja database for PV and wind onshore """
        if not os.path.exists(self.sourcePath / "maxLoad" / f"maxLoadRenwableNinja_{technology}.pickle") or not self.useExistingCapacityFactor:
            capacityFactorRN = pd.DataFrame(index=self.timeRange, columns=self.nodes["node"])
            merra2Idx = 0
            if technology == "photovoltaics":
                techIdxRN = 0
            else:
                techIdxRN = 1
            for idx, node in enumerate(self.nodes["node"]):
                print(
                    f"Extract capacity factors for {node} of technology {technology} - {idx + 1}/{len(self.nodes['node'])}")
                if node == "EL":
                    nodeRN = "GR"
                elif node == "UK":
                    nodeRN = "GB"
                else:
                    nodeRN = node
                url = self.ninjaApiBase + "api/countries/" + nodeRN
                text = self.ninjaClient.get(url)
                countryJson = json.loads(text.text)
                storageOptions = {'User-Agent': 'Mozilla/5.0'}
                try:
                    capacityFactor = pd.read_csv(self.ninjaApiBase+countryJson["downloads"][techIdxRN]["files"][merra2Idx]["url"],storage_options=storageOptions,low_memory=False)
                except IndexError:
                    print(f"Node {node} has no matching data for onshore wind. Skipped")
                    continue
                capacityFactor.columns = capacityFactor.loc[1]
                capacityFactor = capacityFactor.loc[2:].set_index("time")
                if technology == "wind_onshore":
                    if "onshore" in capacityFactor.columns:
                        capacityFactor = capacityFactor["onshore"]
                    elif "national" in capacityFactor.columns:
                        capacityFactor = capacityFactor["national"]
                    else:
                        print(f"Node {node} has no matching data for onshore wind. Skipped")
                        continue
                capacityFactor.index = pd.to_datetime(capacityFactor.index,utc=True)
                capacityFactor = capacityFactor.truncate(before=self.timeStart, after=self.timeEnd-pd.Timedelta(hours=1))
                capacityFactorRN[node] = capacityFactor
            # dump pickle
            with open(self.sourcePath / "maxLoad" / f"maxLoadRenwableNinja_{technology}.pickle",
                      "wb") as inputFile:
                pickle.dump(capacityFactorRN, inputFile)
        else:
            with open(self.sourcePath / "maxLoad" / f"maxLoadRenwableNinja_{technology}.pickle",
                      "rb") as inputFile:
                capacityFactorRN = pickle.load(inputFile)
        return capacityFactorRN