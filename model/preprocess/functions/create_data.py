"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class to create new data from the input datasets based on default values
==========================================================================================================================================================================="""

# IMPORT AND SETUP
import pandas as pd
import numpy  as np


#%% CLASS DEFINITION AND METHODS
class Create:
    
    def conversionMatrices(self):
        """
        This method creates: (i)  the efficiency matrix with the input data for each technology,
                             (ii) the availability matrix which defines which carriers can be converted
        """
        technologySubset        = 'setProductionTechnologies'
        inputFileName           = 'conversionBalanceConstant'
        newFileNameEfficiency   = 'conversionMatrix'
        newFileNameAvailability = 'availabilityMatrix'
        numberCarriers          = len(self.system['setCarriers'])        
        
        for technologyName in self.system[technologySubset]:
            
            # create empty dataframes having all the setCarriers as index and columns
            dfConversionBalance = pd.DataFrame(np.zeros([numberCarriers, numberCarriers]),
                                                index   = self.system['setCarriers'],
                                                columns = self.system['setCarriers'],
                                                dtype   = np.int)

            dfAvailabilityMatrix = pd.DataFrame(np.zeros([numberCarriers, numberCarriers]),
                                                index   = self.system['setCarriers'],
                                                columns = self.system['setCarriers'],
                                                dtype   = np.int)
            
            # dataframe stored in data
            dataConversionBalance = self.data[technologySubset][technologyName][inputFileName]
            dataConversionBalance = dataConversionBalance.set_index(self.analysis['dataInputs']['nameCarrier'])  
                 
            # list of input and output carriers
            mapOutputCarriers = (dataConversionBalance[self.analysis['dataInputs']['nameConversionBalance']] == 1.0)
            outputCarriers    = list(dataConversionBalance[mapOutputCarriers].index)

            # collect the remaining carriers in the input dataframe and remove the output carriers
            inputCarriers = list(dataConversionBalance[~mapOutputCarriers].index)            
            
            for inputCarrier in inputCarriers:
                for outputCarrier in outputCarriers:                  
                    if ((inputCarrier not in self.system['setCarriers']) or (outputCarrier not in self.system['setCarriers'])):
                        raise ValueError("Carriers in technology {technologyName} not matching setCarriers")
                        
                    # assign the conversion balance constants according to the input data
                    dfConversionBalance.loc[inputCarrier, outputCarrier] = dataConversionBalance.loc[inputCarrier, self.analysis['dataInputs']['nameConversionBalance']]   

                    # assign 0 or 1 depending if the carrier is converted
                    dfAvailabilityMatrix.loc[inputCarrier, outputCarrier] = 1
            
            # change the indexing and rename it as column
            self.data[technologySubset][technologyName][newFileNameEfficiency]   = dfConversionBalance.reset_index()
            self.data[technologySubset][technologyName][newFileNameAvailability] = dfAvailabilityMatrix.reset_index()            
            self.data[technologySubset][technologyName][newFileNameEfficiency].rename(columns={"index": self.analysis['dataInputs']['nameCarrier']}, inplace=True)
            self.data[technologySubset][technologyName][newFileNameAvailability].rename(columns={"index": self.analysis['dataInputs']['nameCarrier']}, inplace=True)
            