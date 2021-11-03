"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to convert the dictionary into a Pyomo compatible dictionary to be passed to the compile routine.
==========================================================================================================================================================================="""


class FillPyoDict:
    
    def __init__(self):
        pass
 
    def sets(self):
        """
        This method adds the sets of the models based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data        
        """

        # create carrier sets
        self.pyoDict[None]['setCarriersIn'] = {None: self.system['setCarriersIn']}
        self.pyoDict[None]['setCarriersOut'] = {None: self.system['setCarriersOut']}
        
        # create technology sets
        self.pyoDict[None]['setProduction'] = {None: self.system['setProduction']}
        self.pyoDict[None]['setTransport'] = {None: self.system['setTransport']}
        
        # create nodes set
        self.pyoDict[None]['setNodes'] = {None:self.system['setNodes']}    
        
        # create times set
        self.pyoDict[None]['setTimeSteps'] = {None:self.system['setTimeSteps']}
        
        # create scenarios set
        self.pyoDict[None]['setScenarios'] = {None:self.system['setScenarios']}
        
    def carrierParameters(self):
        """
        This method adds the parameters of the models dependent on the energy carriers based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data        
        """        
        
        for carrierType in self.carrierTypes.keys():
            for parameterName in self.dataTypes[carrierType]:
                for carrierName in self.system[self.carrierTypes[carrierType]]:
                    for nodeName in self.system['setNodes']:
                        for timeName in self.system['setTimeSteps']:
                            for scenarioName in self.system['setScenarios']:
    
                                df = self.data['output_carriers'][carrierName][parameterName]
                                
                                if df.empty:
                                    pass
                                
                                else:
                                    key = (carrierName, nodeName, timeName, scenarioName)                                
                                    value = df.loc[(scenarioName, timeName, nodeName), parameterName]                            
                                    self.pyoDict[None][parameterName] = {key: value}  
                                
                                
    def technologiesParameters(self):
        """
        This method adds the parameters of the models dependent on the technologies based on config
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data        
        """  
        
        technologiesProdParametersList = ['availability']
        for parameterName in technologiesProdParametersList:
            for technologyName in self.system['setProduction']:
                for nodeName in self.system['setNodes']:
                    for timeName in self.system['setTimeSteps']:
                        for scenarioName in self.system['setScenarios']:
                            
                            technologyKey = '{}{}'.format(technologyName,'')
                            key = (technologyKey, nodeName, timeName, scenarioName)
                            df = self.data['production_technologies'][technologyName][parameterName]
                            
                            if df.empty:
                                pass
                            
                            else:
                                value = df.loc[(scenarioName, timeName, nodeName), parameterName]                            
                                self.pyoDict[None][parameterName] = {key: value}            
        
        
        
        # technologiesTranspParametersList = ['availability']        
    
        
                                