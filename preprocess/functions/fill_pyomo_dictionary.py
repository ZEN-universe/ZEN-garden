"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com), Alissa Ganter (aganter@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to convert the dictionary into a Pyomo compatible dictionary to be passed to the compile routine.
==========================================================================================================================================================================="""
from preprocess.functions.add_parameters import add_parameter

class FillPyoDict:
    
    def __init__(self):
        pass
 
    def sets(self):
        """
        This method adds the sets of the models based on config and creates new sets used in the creation of the model instance
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data        
        """
        
        for setName in self.system.keys():
            if 'set' in setName:
                # create sets
                self.pyoDict[None][setName] = {None: self.system[setName]}
        
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
        
        for carrierSubset in self.analysis['subsets']['setCarriers']:
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
                        for parameterName in ['availability', 'costPerDistance', 'distanceEucledian', 'efficiencyPerDistance']:
                            # dataframe stored in data 
                            df = self.data[technologySubset][technologyName][parameterName]
                            # list of columns of the dataframe to use as indexes
                            dfIndexNames = [self.analysis['dataInputs']['nameNodes']]
                            # index of the single cell in the dataframe to add to the dictionary
                            dfIndex = nodeName
                            # column of the single cell in the dataframe to add to the dictionary  
                            dfColumn = nodeNameAlias
                            # key to use in the Pyomo dictionary
                            key = (nodeName, nodeNameAlias, timeName)
                            # key to use in the Pyomo dictionary
                            name = parameterName + technologyName
                            # add the paramter to the Pyomo dictionary based on the key and the dataframe value in [dfIndex,dfColumn]
                            add_parameter(self.pyoDict[None], df, dfIndexNames, dfIndex, dfColumn, key, name)
                                        
    def technologyProductionStorageParameters(self):
        """
        This method adds the parameters of the models dependent on the production technologies based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing the input data        
        """  
        
        parameterNames = {
            'setConversionTechnologies': ['availability'],
            'setStorageTechnologies': ['availability']
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
                            key = (nodeName, timeName)
                            # name to use in the Pyomo dictionary
                            name = parameterName + technologyName
                            # add the paramter to the Pyomo dictionary based on the key and the dataframe value in [dfIndex,dfColumn]
                            add_parameter(self.pyoDict[None], df, dfIndexNames, dfIndex, dfColumn, key, name)
    
    def attributes(self):
        """
        This method adds the parameters of the models dependent on the production and storage technologies based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing the input data
        """

        # Alissa: could we get information like the technologySubsets/ attributes from the dataframe directly
        #         so it is less "hard-coded"?

        parameterName = 'attributes'
        attributes = ['minCapacity', 'maxCapacity', 'minLoad', 'maxLoad', 'valueCapex']

        for technologySubset in ['setConversionTechnologies', 'setTransportTechnologies']:
            for technologyName in self.system[technologySubset]:
                for attribute in attributes:
                    self.pyoDict[None][attribute+technologyName] = {}
                    # dataframe stored in data
                    df = self.data[technologySubset][technologyName][parameterName].set_index(['index'])
                    # Pyomo dictionary key
                    key = None
                    # add the parameter to the Pyomo dictionary
                    self.pyoDict[None][attribute + technologyName][key] = df.loc[attribute, parameterName]

        technologySubset = 'setTransportTechnologies'
        for technologyName in self.system[technologySubset]:
            for carrierName in set(self.system['setOutputCarriers']+self.system['setInputCarriers']):
                if carrierName in technologyName:
                    self.pyoDict[None]['setTransportCarriers'+technologyName] = {None:[carrierName]}

                
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
                    # add the parameter to the Pyomo dictionary
                    self.pyoDict[None][parameterName][key] = value           

    def conversionBalanceParameters(self):
        """
        This method adds the parameters of the models dependent on the production and storage technologies based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing the input data
        """           
        
        technologySubset = 'setConversionTechnologies'
        parameterName = 'converEfficiency'
        
        for technologyName in self.system[technologySubset]:
            # dataframe stored in data
            df = self.data[technologySubset][technologyName][parameterName]
            # list of columns of the dataframe to use as indexes
            dfIndexName = self.analysis['dataInputs']['nameCarrier']
            # get input and output carriers
            setInputCarriers  = df[dfIndexName].tolist()
            setOutputCarriers = df.columns.tolist()[1:]
            # add input and output carrier set to pyoDict
            self.pyoDict[None][f'setInputCarriers{technologyName}'] = {None: setInputCarriers}
            self.pyoDict[None][f'setOutputCarriers{technologyName}'] = {None: setOutputCarriers}

            for carrierIn in setInputCarriers:
                 for carrierOut in setOutputCarriers:
                    # index of the single cell in the dataframe to add to the dictionary
                    dfIndex = carrierIn
                    # column of the single cell in the dataframe to add to the dictionary
                    dfColumn = carrierOut
                    # key to use in the Pyomo dictionary
                    key = (carrierIn, carrierOut)
                    #  name to use in the Pyomo dictionary
                    name = parameterName + technologyName
                    # add the paramter to the Pyomo dictionary based on the key and the dataframe value in [dfIndex,dfColumn]
                    add_parameter(self.pyoDict[None], df, dfIndexName, dfIndex, dfColumn, key, name)

    def dataPWAApproximation(self):
        
        # add the parameters associated to the PWA approximation
        technologySubset = 'setConversionTechnologies'
        types = self.analysis['linearTechnologyApproximation'].keys()
        for technologyName in self.system[technologySubset]:
            for type in types:
                if technologyName in self.analysis['linearTechnologyApproximation'][type]:
                    df = self.data[technologySubset][technologyName][f'linear{type}']
                elif technologyName in self.analysis['nonlinearTechnologyApproximation'][type]:
                    pass
                else:
                    df = self.data[technologySubset][technologyName][f'PWA{type}']

                df['index'] = df.index.values
                for parameterName in self.analysis['dataInputs']['PWA'].values():
                    for supportPointPWA in df.index.values:
                        name = f'{parameterName}{type}{technologyName}'
                        # list of columns of the dataframe to use as indexes
                        dfIndexNames = ['index']
                        # index of the single cell in the dataframe to add to the dictionary
                        dfIndex = supportPointPWA
                        # column of the single cell in the dataframe to add to the dictionary
                        dfColumn = self.analysis['dataInputs']['PWA'][parameterName]
                        # key to use in the Pyomo dictionary
                        key = (supportPointPWA)
                        # add the paramter to the Pyomo dictionary based on the key and the dataframe value in [dfIndex,dfColumn]
                        add_parameter(self.pyoDict[None], df, dfIndexNames, dfIndex, dfColumn, key, name)