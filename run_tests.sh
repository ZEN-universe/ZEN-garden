#!/bin/bash

# exit on error and trace to trap
set -eE 
# print report in any case
trap "coverage report -m" EXIT

# setup 
#######
# move everything to a tmp dir to not mess with whatever is in data, delete in case already there
rsync -a --delete --exclude=".git/" --exclude="data/" $(pwd) /tmp
# move to tmp and create the data
cd /tmp/$(basename $(pwd))
rm -rf data
mkdir data
mv ./tests/testcases/* ./data

# erase current coverage
coverage erase

# cycle through all test cases
for d in ./data/test_*/
  do
  echo "Running test: $d"
  echo "===================================="
  current_test=$(basename $d)
  sed -e "s/analysis\[\"dataset\"\].*/analysis\[\"dataset\"\] \= \"${current_test}\"/g" -i "./data/config.py"
  coverage run --source="./model,./preprocess,./postprocess" -a compile.py
done
