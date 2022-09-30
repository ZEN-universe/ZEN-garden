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

# All the tests
###############

def test_1a(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_1a"))


def test_1b(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_1b"))


def test_2c(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_2c"))


def test_2d(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_2d"))


def test_3a(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_3a"))


def test_3b(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_3b"))


def test_4a(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4a"))


def test_4b(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4b"))


def test_4c(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4c"))


def test_4d(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4d"))


def test_4e(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4e"))


def test_4f(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4f"))


def test_4g(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_4g"))


def test_5a(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_5a"))


def test_6a(config):
    # run the test
    restore_default_state()

    # Change the config according to Alissa's settings
    # Slack message 30/09/22
    config.solver["analyzeNumerics"]                           = False

    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_6a"))


def test_7a(config):
    # run the test
    restore_default_state()
    compile(config=config, dataset_path=os.path.join(os.path.dirname(__file__), "test_7a"))
