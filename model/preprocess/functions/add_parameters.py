"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Methods used in the class FillPyoDict to fill the dictionary in Pyomo format
==========================================================================================================================================================================="""

def add_parameter(dictionary, df, dfIndexNames, dfIndex, dfColumn, key, parameterName):
    
    if df.empty:
        pass
    
    else:
        df = df.set_index(dfIndexNames)
                                     
        value = df.loc[dfIndex, dfColumn] 

        if parameterName not in dictionary.keys():       
            # create a new dictionary for the parameter
            dictionary[parameterName] = {key: value}
        else:
            # add the indexes to the dictionary
            dictionary[parameterName][key] = value    