# =====================================================================================================================
#                                   ENERGY-CARBON OPTIMIZATION PLATFORM
# =====================================================================================================================

#                                Institute of Energy and Process Engineering
#                               Labratory of Risk and Reliability Engineering
#                                         ETH Zurich, September 2021

# ======================================================================================================================
#                                                   MASTER FILE
# ======================================================================================================================

import os
import logging
import numpy as np
import pandas as pd
#import config
from model.preprocess.prepare import Prepare
#from core.model import Model
# import core.model as ValueChain

# SETUP LOGGER
log_format = '%(asctime)s %(filename)s: %(message)s'
if not os.path.exists('logs'):
    os.mkdir('logs')
logging.basicConfig(filename='logs/valueChain.log', level=logging.CRITICAL, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
logging.propagate = False #prevent double printing

# CREATE INPUT FILE
resolution = 'NUTS0'
prepare = Prepare({'case':'NUTS0'}, {})

# FORMULATE AND SOLVE THE OPTIMIZATION PROBLEM
#model = Model(config.analyis, config.system)
# results = valueChain.solve(config.solver)

# EVALUATE RESULTS
# postprocess(results)