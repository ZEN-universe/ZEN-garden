"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
              Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Risk and Reliability Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
==========================================================================================================================================================================="""

import os
import logging
import sys
from datetime import datetime
import data.NUTS0_electricity.config as config
from preprocess.prepare import Prepare
from model.optimization_setup import OptimizationSetup
from model.metaheuristic.algorithm import Metaheuristic
from postprocess.results import Postprocess

# SETUP LOGGER
log_format = '%(asctime)s %(filename)s: %(message)s'
if not os.path.exists('outputs/logs'):
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    os.mkdir('outputs/logs')
logging.basicConfig(filename='outputs/logs/valueChain.log', level=logging.INFO, format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S')
logging.captureWarnings(True)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
# handler.setFormatter(formatter)
logging.getLogger().addHandler(handler)

# prevent double printing
logging.propagate = False

# create a dictionary with the paths to access the model inputs and check if input data exists
prepare = Prepare(config)
# check if all data inputs exist and remove non-existent
system = prepare.checkExistingInputData()

# FORMULATE THE OPTIMIZATION PROBLEM
# add the elements and read input data
optimizationSetup           = OptimizationSetup(config.analysis, prepare)
# get rolling horizon years
stepsOptimizationHorizon    = optimizationSetup.getOptimizationHorizon()
# iterate through horizon steps
for stepHorizon in stepsOptimizationHorizon:
    if len(stepsOptimizationHorizon) == 1:
        logging.info("\n--- Conduct optimization for perfect foresight --- \n")
    else:
        logging.info(f"\n--- Conduct optimization for rolling horizon step {stepHorizon} of {max(stepsOptimizationHorizon)}--- \n")
    # overwrite time indices
    optimizationSetup.overwriteTimeIndices(stepHorizon)
    # create optimization problem
    optimizationSetup.constructOptimizationProblem()
    # SOLVE THE OPTIMIZATION PROBLEM
    if config.solver['model'] == 'MILP':
        # BASED ON MILP SOLVER
        optimizationSetup.solve(config.solver)

    elif config.solver['model'] == 'MINLP':
        # BASED ON HYBRID SOLVER - MASTER METAHEURISTIC AND SLAVE MILP SOLVER
        master = Metaheuristic(optimizationSetup, prepare.nlpDict)
        master.solveMINLP(config.solver)

    # add newly builtCapacity of first year to existing capacity
    optimizationSetup.addNewlyBuiltCapacity(stepHorizon)
    # EVALUATE RESULTS
    today      = datetime.now()
    modelName  = "model_" + today.strftime("%Y-%m-%d")
    if len(stepsOptimizationHorizon)>1:
        modelName += f"_rollingHorizon{stepHorizon}"
    else:
        modelName += "_perfectForesight"
    evaluation = Postprocess(optimizationSetup, modelName = modelName)
