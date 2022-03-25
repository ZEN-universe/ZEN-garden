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
import data.config as config
from preprocess.prepare            import Prepare
from model.optimization_setup      import OptimizationSetup
from model.metaheuristic.algorithm import Metaheuristic
from postprocess.results import Postprocess

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
logging.getLogger().addHandler(handler)

# prevent double printing
logging.propagate = False

# CREATE INPUT FILE
prepare = Prepare(config)

# check if all data inputs exist and remove non-existent
system = prepare.checkExistingInputData()

# FORMULATE THE OPTIMIZATION PROBLEM
optimizationSetup = OptimizationSetup(config.analysis, system, prepare.paths, prepare.solver)

# SOLVE THE OPTIMIZATION PROBLEM
if config.solver['model'] == 'MILP':
    # MILP solver
    optimizationSetup.solve(config.solver)

elif config.solver['model'] == 'MINLP':
    # Hybrid solver - Master metaheuristic and slave MILP
    master = Metaheuristic(optimizationSetup, prepare.nlpDict)
    master.solveMINLP(config.solver)

# EVALUATE RESULTS
evaluation = Postprocess(optimizationSetup, modelName = config.system["modelName"])
a = 1
