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
from zen_garden._internal import compile

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
    testVariables = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + '\\test_variables_readable.csv', header=0,index_col=None)
    # list to store variable names and indices of variables which don't match the test values
    failedVariables = []
    for i in range(testVariables.shape[0]):
        # skip line if data doesn't correspond to selected test model
        if testVariables['test'][i] != testModel:
            continue
        variableName = testVariables['variableName'][i]
        index = testVariables['index'][i]
        class_method = getattr(optimizationSetup.model, variableName)
        # iterate over indices of current variable
        for x in class_method.extract_values():
            if str(x) == index:
                if class_method.extract_values()[x] != 0:
                    # check if relative error exceeds limit of 10^-3
                    if abs(class_method.extract_values()[x] - testVariables['value'][i]) / \
                            class_method.extract_values()[x] > 10 ** (-3):
                        failedVariables.append(testVariables['variableName'][i] + ' ' + testVariables['index'][i])
                else:
                    # check if variable and test variable aren't equal
                    if class_method.extract_values()[x] != testVariables['value'][i]:
                        failedVariables.append(testVariables['variableName'][i] + testVariables['index'][i])
    return  failedVariables

# All the tests
###############

def test_1a(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_1a"))

    failedVariables = compareVariables('test_1a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_1b(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_1b"))

    failedVariables = compareVariables('test_1b',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"


def test_2a(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_2a"))

    failedVariables = compareVariables('test_2a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_2b(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_2b"))

    failedVariables = compareVariables('test_2b',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_2c(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_2c"))

    failedVariables = compareVariables('test_2c',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_2d(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_2d"))

    failedVariables = compareVariables('test_2d',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_3a(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_3a"))

    failedVariables = compareVariables('test_3a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_3b(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_3b"))

    failedVariables = compareVariables('test_3b',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_4a(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4a"))

    failedVariables = compareVariables('test_4a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_4b(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4b"))

    failedVariables = compareVariables('test_4b',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_4c(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4c"))

    failedVariables = compareVariables('test_4c',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"


def test_4d(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4d"))

    failedVariables = compareVariables('test_4d',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_4e(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4e"))

    failedVariables = compareVariables('test_4e',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_4f(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4f"))

    failedVariables = compareVariables('test_4f',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_4g(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4g"))

    failedVariables = compareVariables('test_4g',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_5a(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_5a"))

    failedVariables = compareVariables('test_5a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_6a(config):
    # run the test
    restore_default_state()

    # Change the config according to Alissa's settings
    # Slack message 30/09/22
    config.solver["analyzeNumerics"]                           = False

    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_6a"))

    failedVariables = compareVariables('test_6a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"

def test_7a(config):
    # run the test
    restore_default_state()
    optimizationSetup = compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_7a"))

    failedVariables = compareVariables('test_7a',optimizationSetup)
    assert len(failedVariables) == 0, f"The variables {failedVariables} don't match their test values"