"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Methods used in the class FillPyoDict to fill the dictionary in Pyomo format
==========================================================================================================================================================================="""

from pyomo.core.base import param


def add_parameter(dictionary, df, dfIndexNames, dfIndex, dfColumn, key, parameter, element=None):
    
    if df.empty:
        pass
    
    else:
        df = df.set_index(dfIndexNames)
        
        value = df.loc[dfIndex, dfColumn] 

        if element:
            # if no element specified
            if parameter not in dictionary:
                # create a new dictionary for the parameter
                dictionary[parameter] = {element:{key:value}}
            else:
                if element not in dictionary[parameter]:
                    # create a new dictionary for the element in the parameter
                    dictionary[parameter][element] = {key:value}
                else:
                    # add the indexes to the dictionary
                    dictionary[parameter][element][key] = value
        else:
            if parameter not in dictionary:       
                # create a new dictionary for the parameter
                dictionary[parameter] = {key: value}
            else:
                # add the indexes to the dictionary
                dictionary[parameter][key] = value    
            
            
def add_function(dictionary, function, key, parameterName):
    
    if parameterName not in dictionary:       
        # create a new dictionary for the function
        dictionary[parameterName] = {key: function}
    else:
        # add the indexes to the dictionary
        dictionary[parameterName][key] = function    


    
    
    
    