"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to convert the dictionary into a Pyomo compatible dictionary to be passed to the compile routine.
==========================================================================================================================================================================="""
from model.preprocess.functions.add_parameters import add_parameter

class FillPyoDict:
    
    def __init__(self):
        pass
 
    def sets(self):
        """
        This method adds the sets of the models based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data        
        """
        
        for setName in self.data.keys():
            # create sets
            self.pyoDict[None][setName] = {None: self.system[setName]}

        
    def carrierParameters(self):
        """
        This method adds the parameters of the models dependent on the energy carriers based on config
        If two parameters are called with the same and the carriers appear in two subsets, the parameter is overwritten
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data        
        """        
        parameterNames = {
            'setCarriersIn': ['availabilityCarrier', 'exportPriceCarrier', 'importPriceCarrier'],
            'setCarriersOut': ['demandCarrier', 'exportPriceCarrier', 'importPriceCarrier']            
            }        
        
        for carrierSubset in self.analysis['carrierSubsets']:
            for carrierName in self.system[carrierSubset]:
                for nodeName in self.system['setNodes']:
                    for timeName in self.system['setTimeSteps']:
                        for scenarioName in self.system['setScenarios']:
                            # assumption: all the following parameters have the same data structure
                            for parameterName in parameterNames[carrierSubset]:                                
    
                                df = self.data[carrierSubset][carrierName][parameterName]
                                dfIndexNames = [self.analysis['dataInputs']['nameScenarios'], self.analysis['dataInputs']['nameTimeSteps'], self.analysis['dataInputs']['nameNodes']]
                                dfIndex = (scenarioName, timeName, nodeName)
                                dfColumn = parameterName
                                key = (carrierName, nodeName, timeName, scenarioName)  
                                
                                add_parameter(self.pyoDict[None], df, dfIndexNames, dfIndex, dfColumn, key, parameterName)                            
                                
    def technologyTranspParameters(self):
        """
        This method adds the parameters of the models dependent on the technologies based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data        
        """  
        
        technologySubset = 'setTransport'
        for technologyName in self.system[technologySubset]:
            for nodeName in self.system['setNodes']:
                for nodeNameAlias in self.system['setNodes']:
                    for timeName in self.system['setTimeSteps']:
                        for scenarioName in self.system['setScenarios']:
                            # assumption: all the following parameters have the same data structure
                            for parameterName in ['availabilityTransport', 'costPerDistance', 'distance', 'efficiencyPerDistance']:
        
                                df = self.data[technologySubset][technologyName][parameterName]
                                dfIndexNames = [self.analysis['dataInputs']['nameNodes']]
                                dfIndex = nodeName
                                dfColumn = nodeNameAlias
                                key = (technologyName, nodeName, nodeNameAlias, timeName, scenarioName)
                                
                                add_parameter(self.pyoDict[None], df, dfIndexNames, dfIndex, dfColumn, key, parameterName)   
                                        
    def technologyProductionStorageParameters(self):
        """
        This method adds the parameters of the models dependent on the technologies based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data        
        """  
        parameterNames = {
            'setProduction': ['availabilityProduction'],
            'setStorage': ['availabilityStorage']            
            }       
        
        for technologySubset in parameterNames.keys():
            for technologyName in self.system[technologySubset]:
                for nodeName in self.system['setNodes']:
                    for timeName in self.system['setTimeSteps']:
                        for scenarioName in self.system['setScenarios']:
                            # assumption: all the following parameters have the same data structure
                            for parameterName in parameterNames[technologySubset]:
        
                                df = self.data[technologySubset][technologyName][parameterName]
                                dfIndexNames = [self.analysis['dataInputs']['nameScenarios'], self.analysis['dataInputs']['nameTimeSteps'], self.analysis['dataInputs']['nameNodes']]
                                dfIndex = (scenarioName, timeName, nodeName)
                                dfColumn = parameterName
                                key = (technologyName, nodeName, timeName, scenarioName)
                                
                                add_parameter(self.pyoDict[None], df, dfIndexNames, dfIndex, dfColumn, key, parameterName) 
        
    def attributes(self):
        
        parameterName = 'attributes'
        technologySubset = 'setProduction'        
        for attribute in ['minCapacityProduction', 'maxCapacityProduction']:
            self.pyoDict[None][attribute] = {}
            for technologyName in self.system[technologySubset]:
                df = self.data[technologySubset][technologyName][parameterName].set_index(['index'])
                self.pyoDict[None][attribute][technologyName] = df.loc[attribute, parameterName]
                
        technologySubset = 'setStorage'        
        for parameterName in ['minCapacityStorage', 'maxCapacityStorage']:
            self.pyoDict[None][parameterName] = {}
            for nodeName in self.system['setNodes']:
                for technologyName in self.system[technologySubset]:
                    df = self.data[technologySubset][technologyName][parameterName].set_index(self.analysis['dataInputs']['nameNodes'])
                    key = (technologyName, nodeName)
                    value = df.loc[nodeName, parameterName]
                    self.pyoDict[None][parameterName][key] = value     
        
        