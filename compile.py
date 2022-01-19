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
from preprocess.prepare import Prepare
from model.model import Model
from model.metaheuristic.algorithm import Metaheuristic
from postprocess.results import Postprocess

# SETUP LOGGER
log_format = '%(asctime)s %(filename)s: %(message)s'
if not os.path.exists('outputs/logs'):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    os.mkdir('outputs/logs')
logging.basicConfig(filename='outputs/logs/valueChain.log', level=logging.DEBUG, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
# prevent double printing
logging.propagate = False

# CREATE INPUT FILE
prepare = Prepare(config)

# FORMULATE THE OPTIMIZATION PROBLEM
model = Model(config.analysis, config.system, prepare.paths)

# SOLVE THE OPTIMIZATION PROBLEM
if config.solver['model'] == 'MILP':
    # BASED ON MILP SOLVER
    model.solve(config.solver)
elif config.solver['model'] == 'MINLP':
    # BASED ON HYBRID SOLVER - MASTER METAHEURISTIC AND SLAVE MILP SOLVER
    master = Metaheuristic(model, prepare.nlpDict)
    # master.solveMINLP(prepare.pyoDict)
a=1
# EVALUATE RESULTS
# evaluation = Postprocess(model, prepare.pyoDict, modelName = 'test')
# print(evaluation)
