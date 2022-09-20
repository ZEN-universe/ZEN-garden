"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      October-2021
Authors:      Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
              Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
==========================================================================================================================================================================="""
import os
import logging
import sys
import pytest
from   config                                    import config
from   zen_garden.preprocess.prepare             import Prepare
from   zen_garden.model.optimization_setup       import OptimizationSetup
from   zen_garden.postprocess.results            import Postprocess
from   zen_garden.model.objects.energy_system    import EnergySystem

# wrap in function for pytest
@pytest.mark.forked
def test_4f():

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
    
    # reset the energy system
    EnergySystem.reset_system()

    # create a dictionary with the paths to access the model inputs and check if input data exists
    prepare = Prepare(config)
    # check if all data inputs exist and remove non-existent
    prepare.checkExistingInputData()
    
    # FORMULATE THE OPTIMIZATION PROBLEM
    # add the elements and read input data
    optimizationSetup           = OptimizationSetup(config.analysis, prepare)
    # get rolling horizon years
    stepsOptimizationHorizon    = optimizationSetup.getOptimizationHorizon()
    
    # update input data
    for scenario, elements in config.scenarios.items():
        optimizationSetup.restoreBaseConfiguration(scenario, elements)  # per default scenario="" is used as base configuration. Use setBaseConfiguration(scenario, elements) if you want to change that
        optimizationSetup.overwriteParams(scenario, elements)
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
            optimizationSetup.solve(config.solver)
            # add newly builtCapacity of first year to existing capacity
            optimizationSetup.addNewlyBuiltCapacity(stepHorizon)
            # add cumulative carbon emissions to previous carbon emissions
            optimizationSetup.addCarbonEmissionsCumulative(stepHorizon)
            # EVALUATE RESULTS
            modelName = config.analysis["dataset"]
            if len(stepsOptimizationHorizon) > 1:
                modelName += f"_MF{stepHorizon}"
            if config.system["conductScenarioAnalysis"]:
                modelName += f"_{scenario}"
            evaluation = Postprocess(optimizationSetup, modelName=modelName)

if __name__ == "__main__":
    test_4f()

