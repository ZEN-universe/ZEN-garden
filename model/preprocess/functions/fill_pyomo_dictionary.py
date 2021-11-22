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
        This method adds the sets of the models based on config and creates new sets used in the creation of the model instance
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data        
        """
        
        for setName in self.data.keys():
            # create sets
            self.pyoDict[None][setName] = {None: self.system[setName]}
        
        ## create new sets derived from the sets from input data
        # 'setCarriers' is composed by all the sets in the data containing 'Carrier' in the name
        self.pyoDict[None]['setCarriers'] = {None:[]}
        subsetCarriers = [subsetName for subsetName in self.data.keys() if 'Carrier' in subsetName]
        if subsetCarriers != []:      
            for setName in subsetCarriers:
                self.pyoDict[None]['setCarriers'][None].extend(self.system[setName])
        
        # 'setTransportCarriers' is defined in config.py
        self.pyoDict[None]['setTransportCarriers'] = {None:[]}
        self.pyoDict[None]['setTransportCarriers'][None].extend(self.system['setTransportCarriers'])
        
    def carrierParameters(self):
        """
        This method adds the parameters of the models dependent on the energy carriers based on config
        If two parameters are called with the same and the carriers appear in two subsets, the parameter is overwritten
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing the input data        
        """        
        
        parameterNames = {
            'setInputCarriers': ['availabilityCarrier', 'exportPriceCarrier', 'importPriceCarrier'],
            'setOutputCarriers': ['demandCarrier', 'exportPriceCarrier', 'importPriceCarrier']            
            }        
        
        scenarioName = self.system['setScenarios']
        
        for carrierSubset in self.analysis['carrierSubsets']:
            for carrierName in self.system[carrierSubset]:
                for nodeName in self.system['setNodes']:
                    for timeName in self.system['setTimeSteps']:
                        # warning: all the following parameters must have the same data structure
                        for parameterName in parameterNames[carrierSubset]:                                
                            # dataframe stored in data 
                            df = self.data[carrierSubset][carrierName][parameterName]
                            # list of columns of the dataframe to use as indexes
                            dfIndexNames = [self.analysis['dataInputs']['nameScenarios'],\
                                            self.analysis['dataInputs']['nameTimeSteps'],\
                                            self.analysis['dataInputs']['nameNodes']]
                            # index of the single cell in the dataframe to add to the dictionary
                            dfIndex = (scenarioName, timeName, nodeName)
                            # column of the single cell in the dataframe to add to the dictionary                                
                            dfColumn = parameterName
                            # key to use in the Pyomo dictionary
                            key = (carrierName, nodeName, timeName)  
                            # add the paramter to the Pyomo dictionary based on the key and the dataframe value in [dfIndex,dfColumn]
                            add_parameter(self.pyoDict[None], df, dfIndexNames, dfIndex, dfColumn, key, parameterName)                            
                                
    def technologyTranspParameters(self):
        """
        This method adds the parameters of the models dependent on the transport technologies based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing the input data        
        """  
        
        technologySubset = 'setTransportTechnologies'
        for technologyName in self.system[technologySubset]:
            for nodeName in self.system['setNodes']:
                for nodeNameAlias in self.system['setNodes']:
                    for timeName in self.system['setTimeSteps']:
                        # warning: all the following parameters must have the same data structure
                        for parameterName in ['availabilityTransport', 'costPerDistance', 'distance', 'efficiencyPerDistance']:
                            # dataframe stored in data 
                            df = self.data[technologySubset][technologyName][parameterName]
                            # list of columns of the dataframe to use as indexes
                            dfIndexNames = [self.analysis['dataInputs']['nameNodes']]
                            # index of the single cell in the dataframe to add to the dictionary
                            dfIndex = nodeName
                            # column of the single cell in the dataframe to add to the dictionary  
                            dfColumn = nodeNameAlias
                            # key to use in the Pyomo dictionary
                            key = (technologyName, nodeName, nodeNameAlias, timeName)
                            # add the paramter to the Pyomo dictionary based on the key and the dataframe value in [dfIndex,dfColumn]
                            add_parameter(self.pyoDict[None], df, dfIndexNames, dfIndex, dfColumn, key, parameterName)   
                                        
    def technologyProductionStorageParameters(self):
        """
        This method adds the parameters of the models dependent on the production technologies based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing the input data        
        """  
        
        parameterNames = {
            'setProductionTechnologies': ['availabilityProduction'],
            'setStorageTechnologies': ['availabilityStorage']            
            }       
        
        scenarioName = self.system['setScenarios']
        
        for technologySubset in parameterNames.keys():
            for technologyName in self.system[technologySubset]:
                for nodeName in self.system['setNodes']:
                    for timeName in self.system['setTimeSteps']:
                        # warning: all the following parameters must have the same data structure
                        for parameterName in parameterNames[technologySubset]:
                            # dataframe stored in data 
                            df = self.data[technologySubset][technologyName][parameterName]
                            # list of columns of the dataframe to use as indexes
                            dfIndexNames = [self.analysis['dataInputs']['nameScenarios'],\
                                            self.analysis['dataInputs']['nameTimeSteps'],\
                                            self.analysis['dataInputs']['nameNodes']]
                            # index of the single cell in the dataframe to add to the dictionary
                            dfIndex = (scenarioName, timeName, nodeName)
                            # column of the single cell in the dataframe to add to the dictionary 
                            dfColumn = parameterName
                            # key to use in the Pyomo dictionary
                            key = (technologyName, nodeName, timeName)
                            # add the paramter to the Pyomo dictionary based on the key and the dataframe value in [dfIndex,dfColumn]
                            add_parameter(self.pyoDict[None], df, dfIndexNames, dfIndex, dfColumn, key, parameterName) 
        
    def attributes(self):
        """
        This method adds the parameters of the models dependent on the production and storage technologies based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing the input data
        """          
        
        parameterName = 'attributes'
        technologySubset = 'setProductionTechnologies'        
        for attribute in ['minCapacityProduction', 'maxCapacityProduction']:
            self.pyoDict[None][attribute] = {}
            for technologyName in self.system[technologySubset]:
                # dataframe stored in data 
                df = self.data[technologySubset][technologyName][parameterName].set_index(['index'])
                # add the paramter to the Pyomo dictionary
                self.pyoDict[None][attribute][technologyName] = df.loc[attribute, parameterName]
                
        technologySubset = 'setStorageTechnologies'        
        for parameterName in ['minCapacityStorage', 'maxCapacityStorage']:
            self.pyoDict[None][parameterName] = {}
            for nodeName in self.system['setNodes']:
                for technologyName in self.system[technologySubset]:
                    # dataframe stored in data 
                    df = self.data[technologySubset][technologyName][parameterName].set_index(self.analysis['dataInputs']['nameNodes'])
                    # key to use in the Pyomo dictionary
                    key = (technologyName, nodeName)
                    # value to use in the Pyomo dictionary
                    value = df.loc[nodeName, parameterName]
                    # add the paramter to the Pyomo dictionary
                    self.pyoDict[None][parameterName][key] = value     
        
        