"""===========================================================================================================================================================================
Title:        ENERGY-CARBON OPTIMIZATION PLATFORM
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
==========================================================================================================================================================================="""

import os
import logging
import config
from model.preprocess.prepare import Prepare
from model.model_instance.model import Model
from model.postprocess.results import Postprocess

# SETUP LOGGER
log_format = '%(asctime)s %(filename)s: %(message)s'
if not os.path.exists('outputs/logs'):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    os.mkdir('outputs/logs')
logging.basicConfig(filename='outputs/logs/valueChain.log', level=logging.CRITICAL, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
# prevent double printing
logging.propagate = False

# CREATE INPUT FILE
prepare = Prepare(config.analysis, config.system)
# FORMULATE AND SOLVE THE OPTIMIZATION PROBLEM
model = Model(config.analysis, config.system)
model.solve(config.solver, prepare.pyoDict)

# EVALUATE RESULTS
evaluation = Postprocess(model, prepare.pyoDict, modelName = 'test')
print(evaluation)