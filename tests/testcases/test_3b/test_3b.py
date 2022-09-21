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
from   config_3b                                 import config
from   zen_garden                                import restore_default_state
from   zen_garden.preprocess.prepare             import Prepare
from   zen_garden.model.optimization_setup       import OptimizationSetup
from   zen_garden.postprocess.results            import Postprocess
from   zen_garden.model.objects.energy_system    import EnergySystem

def test_3b():

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
    
    # restore defaults
    restore_default_state()

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
            nameDir = os.path.join(config.analysis["dataset"], "outputs")
            if len(stepsOptimizationHorizon) > 1:
                nameDir += f"_MF{stepHorizon}"
            if config.system["conductScenarioAnalysis"]:
                nameDir += f"_{scenario}"
            evaluation = Postprocess(optimizationSetup, nameDir=nameDir)

if __name__ == "__main__":
    test_3b()

