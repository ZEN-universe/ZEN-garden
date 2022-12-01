"""===========================================================================================================================================================================
Title:        ZEN-GARDEN
Created:      September-2022
Authors:      Janis Fluri (janis.fluri@id.ethz.ch)
              Alissa Ganter (aganter@ethz.ch)
              Davide Tonelli (davidetonelli@outlook.com)
              Jacob Mannhardt (jmannhardt@ethz.ch)
Organization: Laboratory of Reliability and Risk Engineering, ETH Zurich

Description:  Compilation  of the optimization problem.
==========================================================================================================================================================================="""
from zen_garden.restore_default_state import restore_default_state
from zen_garden._internal import main

import pytest
from copy import deepcopy

import os

# fixture to get the default config (always as new instance)
@pytest.fixture
def config():
    from config import config
    return deepcopy(config)

#adaption LK
import pandas as pd

def compareVariables(testModel,optimizationSetup):
    # import csv file containing selected variable values of test model collection
    testVariables = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\\test_variables_readable.csv',
                                header=0, index_col=None)
    # dictionary to store variable names, indices, values and test values of variables which don't match the test values
    failedVariables = {}
    #iterate through dataframe rows
    for dataRow in testVariables.values:
        #skip row if data doesn't correspond to selected test model
        if dataRow[0] != testModel:
            continue
        #get variable attribute of optimizationSetup object by using string of the variable's name (e.g. optimizationSetup.model.importCarrierFLow)
        variableAttribute = getattr(optimizationSetup.model,dataRow[1])
        #iterate through indices of current variable
        for variableIndex in variableAttribute.extract_values():
            #ensure equality of dataRow index and variable index
            if str(variableIndex) == dataRow[2]:
                #check if variable value at current index differs from zero such that relative error can be computed
                if variableAttribute.extract_values()[variableIndex] != 0:
                    #check if relative error exceeds limit of 10^-3, i.e. value differs from test value
                    if abs(variableAttribute.extract_values()[variableIndex] - dataRow[3]) / variableAttribute.extract_values()[variableIndex] > 10**(-3):
                        if dataRow[1] in failedVariables:
                            failedVariables[dataRow[1]][dataRow[2]] = {'computedValue' : variableAttribute.extract_values()[variableIndex]}
                        else:
                            failedVariables[dataRow[1]] = {dataRow[2] : {'computedValue' : variableAttribute.extract_values()[variableIndex]} }
                        failedVariables[dataRow[1]][dataRow[2]]['testValue'] = dataRow[3]
                else:
                    #check if absolute error exceeds specified limit
                    if abs(variableAttribute.extract_values()[variableIndex] - dataRow[3]) > 10**(-3):
                        if dataRow[1] in failedVariables:
                            failedVariables[dataRow[1]][dataRow[2]] = {'computedValue' : variableAttribute.extract_values()[variableIndex]}
                        else:
                            failedVariables[dataRow[1]] = {dataRow[2] : {'computedValue' : variableAttribute.extract_values()[variableIndex]} }
                        failedVariables[dataRow[1]][dataRow[2]]['testValue'] = dataRow[3]
    assertionString = str()
    for failedVar in failedVariables:
        assertionString += f"\n{failedVar}{failedVariables[failedVar]}"

    return failedVariables, assertionString


# All the tests
###############

def test_1a(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_1a"))

    failedVariables, assertionString = compareVariables('test_1a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_1b(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_1b"))

    failedVariables, assertionString = compareVariables('test_1b', optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_2a(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_2a"))

    failedVariables, assertionString = compareVariables('test_2a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_2b(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_2b"))

    failedVariables, assertionString = compareVariables('test_2b',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_2c(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_2c"))

    failedVariables, assertionString = compareVariables('test_2c',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_2d(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_2d"))

    failedVariables, assertionString = compareVariables('test_2d',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_3a(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_3a"))

    failedVariables, assertionString = compareVariables('test_3a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_3b(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_3b"))

    failedVariables, assertionString = compareVariables('test_3b',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"
"""
def test_4a(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4a"))

    failedVariables, assertionString = compareVariables('test_4a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_4b(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4b"))

    failedVariables, assertionString = compareVariables('test_4b',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_4c(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4c"))

    failedVariables, assertionString = compareVariables('test_4c', optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_4d(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4d"))

    failedVariables, assertionString = compareVariables('test_4d',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_4e(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4e"))

    failedVariables, assertionString = compareVariables('test_4e',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_4f(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4f"))

    failedVariables, assertionString = compareVariables('test_4f',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_4g(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4g"))

    failedVariables, assertionString = compareVariables('test_4g',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"
"""
def test_5a(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_5a"))

    failedVariables, assertionString = compareVariables('test_5a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_6a(config):
    # run the test
    restore_default_state()

    # Change the config according to Alissa's settings
    # Slack message 30/09/22
    config.solver["analyzeNumerics"]                           = False

    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_6a"))

    failedVariables, assertionString = compareVariables('test_6a', optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"

def test_7a(config):
    # run the test
    restore_default_state()
    optimizationSetup = main(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_7a"))

    failedVariables, assertionString = compareVariables('test_7a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {assertionString} don't match their test values"