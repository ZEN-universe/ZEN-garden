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
from preprocess.prepare import Prepare
from model_instance.model import Model

# SETUP LOGGER
log_format = '%(asctime)s %(filename)s: %(message)s'
if not os.path.exists('logs'):
    os.mkdir('logs')
logging.basicConfig(filename='logs/valueChain.log', level=logging.CRITICAL, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
logging.propagate = False #prevent double printing

# CREATE INPUT FILE
#prepare = Prepare(0, 0)

# FORMULATE AND SOLVE THE OPTIMIZATION PROBLEM
model = Model(config.analyis, config.system)
# results = valueChain.solve(config.solver)

# EVALUATE RESULTS
# postprocess(results)