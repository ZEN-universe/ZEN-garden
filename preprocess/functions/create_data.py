"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Davide Tonelli (davidetonelli@outlook.com)
Organization: Labratory of Risk and Reliability Engineering, ETH Zurich

Description:  Class to create new data from the input datasets based on default values
==========================================================================================================================================================================="""

import pandas as pd
import numpy  as np

class Create:
    
    def conversionMatrices(self):
        """
        Create the efficiency matrix with the input data for each technology and the availability matrix which defines which carriers can are converted
        """
        technologySubset        = 'setConversionTechnologies'
        inputFileName           = 'conversionBalanceConstant'
        newFileNameEfficiency   = 'converEfficiency'
        newFileNameAvailability = 'converAvailability'
        
        dfShape = [len(self.system['setImportCarriers'])  , len(self.system['setExportCarriers'])]
        
        for technologyName in self.system[technologySubset]:
            
            # create empty dataframes having all the setCarriers as index and columns
            dfConversionBalance = pd.DataFrame(np.zeros(dfShape),
                                                index = self.system['setImportCarriers'],
                                                columns = self.system['setExportCarriers'],
                                                dtype=np.int)
            dfAvailabilityMatrix = pd.DataFrame(np.zeros(dfShape),
                                                index = self.system['setImportCarriers'],
                                                columns = self.system['setExportCarriers'],
                                                dtype=np.int)
            
            # dataframe stored in data
            dataConversionBalance = self.data[technologySubset][technologyName][inputFileName]
            dataConversionBalance = dataConversionBalance.set_index(self.analysis['dataInputs']['nameCarrier'])  
                 
            # list of import and export carriers
            mapExportCarriers = (dataConversionBalance[self.analysis['dataInputs']['nameConversionBalance']] == 1.0)
            ExportCarriers = list(dataConversionBalance[mapExportCarriers].index)

            # collect the remaining carriers in the import dataframe and remove the export carriers
            ImportCarriers = list(dataConversionBalance[~mapExportCarriers].index)            

            for importCarrier in ImportCarriers:
                for exportCarrier in ExportCarriers:
                    
                    if ((importCarrier not in self.system['setImportCarriers']) or (exportCarrier not in self.system['setExportCarriers'])):
                        raise ValueError(f"Carriers in technology {technologyName} not matching setCarriers")
                        
                    # assign the conversion balance constants according to the input data
                    dfConversionBalance.loc[importCarrier, exportCarrier] = dataConversionBalance.loc[importCarrier, self.analysis['dataInputs']['nameConversionBalance']]    

                    # assign 0 or 1 depending if the carrier is converted
                    dfAvailabilityMatrix.loc[importCarrier, exportCarrier] = 1

            # change the indexing and rename it as column
            self.data[technologySubset][technologyName][newFileNameEfficiency]   = dfConversionBalance.reset_index()
            self.data[technologySubset][technologyName][newFileNameAvailability] = dfAvailabilityMatrix.reset_index()      # TODO is availability matrix used?      
            self.data[technologySubset][technologyName][newFileNameEfficiency].rename(columns={"index": self.analysis['dataInputs']['nameCarrier']}, inplace=True)
            self.data[technologySubset][technologyName][newFileNameAvailability].rename(columns={"index": self.analysis['dataInputs']['nameCarrier']}, inplace=True)
