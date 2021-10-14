# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================
#                                    PREPARATION: DEFINE PATHS AND REARRANGE DATA
# ======================================================================================================================

import os
import pandas as pd
import preprocess.auxiliary.initialise as init
import preprocess.auxiliary.read_data as read
import preprocess.auxiliary.compute_from_data as compute

class Prepare:
    
    def __init__(self, analysis, system):
        """
        create dictionary with input data
        :param analysis: dictionary defining the analysis framework
        :param system: dictionary defining the system
        """
        
        # analysis structure
        self.analysis = analysis
        # system structure
        self.system = system 
        
        self.paths = {
            'Carriers':'.//data//carriers//',
            'Network':'.//data//network//',
            'Technologies':'.//data//technologies//'
            }        
        
        # create dictionary with keys collecting all input data of the model 
        # to be passed to the core for solution of the specific model
        self.initDict()
        
        # read the data given the initialised dictionary
        self.readData()
        
        # derive data from input data
        self.computeData()

    def initDict(self):
        
        self.input = dict()
        
        init.Carriers(self)
        init.Network(self)
        init.Technologies(self)
        
    def readData(self):
        
        read.Carriers(self)
        read.Network(self)
        read.Technologies(self)
        
    def computeData(self):
        
        compute.DistanceMtx(self)
        
    
    
    
        