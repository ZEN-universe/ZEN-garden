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
import sys
import logging
import importlib.util
import pkg_resources

from shutil import rmtree

from   .preprocess.prepare             import Prepare
from   .model.optimization_setup       import OptimizationSetup
from   .postprocess.results            import Postprocess


def main(config, dataset_path=None):
    """
    This function runs the compile.py script that was used in ZEN-Garden prior to the package build, it is executed
    in the __main__.py script
    :param config: A config instance used for the run
    :param dataset_path: If not None, used to overwrite the config.analysis["dataset"]
    """
    # SETUP LOGGER
    log_format = '%(asctime)s %(filename)s: %(message)s'
    log_path = os.path.join('outputs', 'logs')
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_path, 'valueChain.log'), level=logging.INFO,
                        format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
    logging.captureWarnings(True)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)

    # print the version
    version = pkg_resources.require("zen_garden")[0].version
    logging.info(f"Running ZEN-Garden version: {version}")

    # prevent double printing
    logging.propagate = False

    # overwrite the path if necessary
    if dataset_path is not None:
        logging.info(f"Overwriting dataset to: {dataset_path}")
        config.analysis["dataset"] = dataset_path
    # get the abs path to avoid working dir stuff
    config.analysis["dataset"] = os.path.abspath(config.analysis['dataset'])
    config.system["folderOutput"] = os.path.abspath(config.system['folderOutput'])

    ### System - load system configurations
    system_path = os.path.join(config.analysis['dataset'], "system.py")
    if not os.path.exists(system_path):
        raise FileNotFoundError(f"system.py not found in dataset: {config.analysis['dataset']}")
    spec    = importlib.util.spec_from_file_location("module", system_path)
    module  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    system  = module.system
    config.system.update(system)

    ### overwrite default system and scenario dictionaries
    if config.system["conductScenarioAnalysis"]:
        scenarios_path = os.path.abspath(os.path.join(config.analysis['dataset'], "scenarios.py"))
        if not os.path.exists(scenarios_path):
            raise FileNotFoundError(f"scenarios.py not found in dataset: {config.analysis['dataset']}")
        spec        = importlib.util.spec_from_file_location("module", scenarios_path)
        module      = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        scenarios   = module.scenarios
        config.scenarios.update(scenarios)

    # create a dictionary with the paths to access the model inputs and check if input data exists
    prepare = Prepare(config)
    # check if all data inputs exist and remove non-existent
    prepare.checkExistingInputData()

    # FORMULATE THE OPTIMIZATION PROBLEM
    # add the elements and read input data
    optimizationSetup           = OptimizationSetup(config.analysis, prepare)
    # get rolling horizon years
    stepsOptimizationHorizon    = optimizationSetup.getOptimizationHorizon()

    # get the name of the dataset
    modelName = os.path.basename(config.analysis["dataset"])
    if os.path.exists(out_folder := os.path.join(config.system["folderOutput"], modelName)):
        if config.system["overwriteOutput"]:
            logging.info(f"Removing existing output folder: {out_folder}")
            rmtree(out_folder)
        else:
            logging.warning(f"The outputfolder {out_folder} already exists")

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
            subfolder = ""
            if config.system["conductScenarioAnalysis"]:
                # handle scenarios
                subfolder += f"scenario_{scenario}"
            # handle myopic foresight
            if len(stepsOptimizationHorizon) > 1:
                if subfolder != "":
                    subfolder += f"_"
                subfolder += f"MF_{stepHorizon}"
            # write results
            evaluation = Postprocess(optimizationSetup, scenarios=config.scenarios, subfolder=subfolder,
                                     modelName=modelName)

    return optimizationSetup
