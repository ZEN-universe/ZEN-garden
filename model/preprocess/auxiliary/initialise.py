# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================

import os
    
def Carriers(self):
    
    self.input['Carriers'] = dict()
    
    # read all the folders in the Carriers directory
    path = self.paths['Carriers']
    for carrier in next(os.walk(path))[1]:
        self.input['Carriers'][carrier] = dict() 
        
def Network(self):

    self.input['Network'] = dict()

def Technologies(self):
    
    self.input['Technologies'] = dict()
    
    # read all the folders in the Techologies directory
    path = self.paths['Technologies']
    for technology in next(os.walk(path))[1]:
        self.input['Technologies'][technology] = dict()