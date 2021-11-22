"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:    Auxiliary functions to the class Model
==========================================================================================================================================================================="""

def hassattr(model, attributeName):
    """
    Method to check if the object "model" has the attribute ""attributeName
    :param model: object
    :param attributeName: string
    :return: boolean
    """
    
    if attributeName in model.__dict__.keys():
        return True
    else:
        return False