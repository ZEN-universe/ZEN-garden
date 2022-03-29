"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
==========================================================================================================================================================================="""

import os
import logging
import sys

import config                       as config
from datetime                       import datetime
from preprocess.prepare             import Prepare
from model.optimization_setup       import OptimizationSetup
from model.metaheuristic.algorithm  import Metaheuristic
from postprocess.results            import Postprocess

# SETUP LOGGER
log_format = '%(asctime)s %(filename)s: %(message)s'
if not os.path.exists('outputs/logs'):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    os.mkdir('outputs/logs')
logging.basicConfig(filename='outputs/logs/valueChain.log', level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
logging.captureWarnings(True)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
# handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)
# prevent double printing
logging.propagate = False

# create a dictionary with the paths to access the model inputs and check if input data exists
prepare = Prepare(config)
paths   = prepare.createPaths()
system  = prepare.checkExistingInputData()

# formulate the optimization problem
optimizationSetup = OptimizationSetup(config.analysis, system, paths, prepare.solver)

# solve the optimization problem
if config.solver['model'] == 'MILP':
    # using the MILP solver
    optimizationSetup.solve(config.solver)

elif config.solver['model'] == 'MINLP':
    # using the MINLP solver
    master = Metaheuristic(optimizationSetup, prepare.nlpDict)
    master.solveMINLP(config.solver)

# EVALUATE RESULTS
today      = datetime.now()
modelName  = "model_" + today.strftime("%Y-%m-%d")
evaluation = Postprocess(optimizationSetup, modelName = modelName)
