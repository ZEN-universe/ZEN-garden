# """===========================================================================================================================================================================
# Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
# Created:      October-2021
# Authors:      Davide Tonelli (davidetonelli@outlook.com)
#               Alissa Ganter (aganter@ethz.ch)
# Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich
#
# Description:  Compilation  of the optimization problem.
# ==========================================================================================================================================================================="""

# IMPORT AND SETUP
import os, logging
import config
import numpy  as np
import pandas as pd

from model.preprocess.prepare     import Prepare
from model.model_instance.model   import Model
from model.postprocess.evaluation import Evaluation

# SETUP LOGGER
log_format = '%(asctime)s %(filename)s: %(message)s'
if not os.path.exists('outputs/logs'):
    os.mkdir('outputs/logs')
logging.basicConfig(filename='outputs/logs/valueChain.log', level=logging.CRITICAL, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')

# prevent double printing
logging.propagate = False


#%% OPTIMIZATION PROBLEM
# CREATE INPUT FILE
prepare = Prepare(config.analysis, config.system)
print(' Model preparation completed.')

# FORMULATE AND SOLVE THE OPTIMIZATION PROBLEM
model = Model(config.analysis, config.system)
model.solve(config.solver, prepare.pyoDict)

# EVALUATE RESULTS
evaluation = Evaluation(config.system, model)
