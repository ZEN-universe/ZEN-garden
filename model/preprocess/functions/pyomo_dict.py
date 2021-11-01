"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Collections of methods used in the class Prepare to collect the data from dictionary containing all the inputs and to create 
                a new dictionary in a format compatible with Pyomo inputs for abstract models.
                The Pyomo disctionary has to match the format in: https://pyomo.readthedocs.io/en/stable/working_abstractmodels/data/raw_dicts.html
==========================================================================================================================================================================="""


def carriers(self):
    """
    This method adds the sets and parameters related to the carriers
    """    
    
    # define the sets according to system
    carrierSubsets = ['setInputCarriers', 'setOutputCarriers']
    for carrierSubset in carrierSubsets:
        
        self.pyoDict[None][carrierSubset] = {None:self.system[carrierSubset]}
    
    # define the parameters according to system
    for parameterType in ['demand', 'availability', 'importPrice', 'exportPrice']:
        self.pyoDict[None][parameterType] = {}
        for carrierName in self.system['setOutputCarriers']: 
            for node in self.system['nodes']:
                for time in self.system['times']:
                    for scenario in self.system['scenarios']: 
                        key = (carrierName, node, time, scenario)
                        value = self.data['output_carriers'][carrierName][parameterType].loc[(node,time,scenario)]
                        self.pyoDict[None][parameterType] = {key: value}
        
def technologies(self):
    """
    This method adds the sets and paramters related to the technologies
    """        
    # define the sets according to system    
    self.pyoDict[None]['setProduction'] = {None: self.system['setProduction']}    
    self.pyoDict[None]['setTransport'] = {None: self.system['setTransport']}
    self.pyoDict[None]['setStorage'] = {None: self.system['setStorage']}    
    
    # ... other paramters according to how Alisssa defined them in model.py
    
def nodes(self):
    """
    This method adds the sets and paramters related to the nodes
    """      
    # define the sets according to system      
    self.pyoDict[None]['nodes'] = {None: self.system['nodes']}   
    
def times(self):
    """
    This method adds the sets and paramters related to the time
    """            
    # define the sets according to system
    self.pyoDict[None]['times'] = {None: self.system['times']}
    
def scenarios(self):
    """
    This method adds the sets and paramters related to the scenarios
    """    
    # define the sets according to system        
    self.pyoDict[None]['scenarios'] = {None: self.system['scenarios']}    
    
    
    
    
    
    