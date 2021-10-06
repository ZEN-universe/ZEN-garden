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
import modelSettings
import core.model as ValueChain

# SETUP LOGGER
log_format = '%(asctime)s %(filename)s: %(message)s'
if not os.path.esits('logs'):
    os.mkdir('logs')
logging.basicConfig(filename='logs/valueChain.log', level=logging.CRITICAL, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
logging.propagate = False #prevent double printing

# CREATE INPUT DICTIONARY
#prepare(config.analyis, config.system)

# FORMULATE AND SOLVE THE OPTIMIZATION PROBLEM
valueChain = ValueChain(config.analyis, config.system)
valueChain.solve(config.solver)
