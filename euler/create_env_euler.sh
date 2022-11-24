#!/bin/bash

# swap to the right module /env2lmod is an alias for this
# space between the dot is because it is an absolute path
. /cluster/apps/local/env2lmod.sh

# actiavte the right modules
# gcc 8.2 stack
module load gcc/8.2.0
# python 3.9.9
module load python/3.9.9
# backend for the solver
module load glpk

# create the env
python -m venv zen_garden_env

# activate the env
source zen_garden_env/bin/activate

# install the requirements
pip install -U pip
pip install -r requirements.txt
pip install -e ..
