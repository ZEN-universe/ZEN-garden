#!/bin/bash

# swap to the right module /env2lmod is an alias for this
# space between the dot is because it is an absolute path
. /cluster/apps/local/env2lmod.sh

# actiavte the right modules
# gcc 8.2 stack
module load gcc/8.2.0
# python 3.11.2
module load python/3.11.2

# backend for the solvers
module load glpk
module load gurobi/10.0.1

# activate the env from anywhere
# $BASH_SOURCE contains the path that was sourced 
source $(realpath $BASH_SOURCE | xargs dirname)/zen_garden_env/bin/activate
