#!/bin/bash

# activate the right modules
# gcc 12.2 stack
module load stack/2024-06
module load gcc/12.2.0
# python 3.11.2
module load python/3.11.6
# backend for the solvers
module load glpk
module load gurobi/12.0.1

# create the env
python -m venv zen_garden_env

# activate the env
source zen_garden_env/bin/activate

# install the requirements
pip install -U pip
pip install -e ..[dev,vis,docs]
pip install gurobipy==12.0.1