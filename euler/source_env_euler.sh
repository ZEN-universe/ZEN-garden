#!/bin/bash


# actiavte the right modules
# gcc 12.2 stack
module load stack/2024-06
module load gcc/12.2.0

# python 3.11.6
module load python/3.11.6

# backend for the solvers
module load glpk
module load gurobi/10.0.3

# activate the env from anywhere
# $BASH_SOURCE contains the path that was sourced
source $(realpath $BASH_SOURCE | xargs dirname)/zen_garden_env/bin/activate
