"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:    Class to create new data from the input datasets based on default values
==========================================================================================================================================================================="""

import pandas as pd
import numpy as np

class Create:
    
    def __init__(self):
        pass
    
    def conversionMatrix(self):
        """
        Create the efficiency matrix with the input data for each technology
        """

        numberCarriers = len(self.system['setCarriers'])
        technologySubset = 'setProductionTechnologies'
        inputFileName = 'conversionBalanceConstant'
        newFileName = 'conversionMatrix'
        
        for technologyName in self.system[technologySubset]:
            
            # dataframe stored in data
            dfConversionBalance = self.data[technologySubset][technologyName][inputFileName]
            dfConversionBalance = dfConversionBalance.set_index(self.analysis['dataInputs']['nameCarrier'])
            
            # list of carrier taken as reference
            mapCarriersReference = (dfConversionBalance[self.analysis['dataInputs']['nameConversionBalance']] == 1.0)
            carriersReference = list(dfConversionBalance[mapCarriersReference].index)
            # collect the remaining carriers in the input dataframe and remove the reference carriers
            carriersConverted = list(dfConversionBalance.index)
            for carrierReference in carriersReference:
                carriersConverted.remove(carrierReference)
            
            # Create a matrix containing the parameters of the technology efficiency given in the input data 
            dfCarriers = pd.DataFrame(
                np.zeros([len(carriersReference), len(carriersConverted)]),
                columns=carriersConverted,
                index=carriersReference
                )
            
            # assign the conversion balance constants according to the input data
            for carrierReference in carriersReference:
                for carrierConverted in carriersConverted:
                    dfCarriers.loc[carrierReference, carrierConverted] = dfConversionBalance.loc[carrierConverted, self.analysis['dataInputs']['nameConversionBalance']]
        
            self.data[technologySubset][technologyName][newFileName] = dfCarriers