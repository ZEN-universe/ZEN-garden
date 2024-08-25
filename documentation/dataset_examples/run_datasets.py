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
import os
import re
from pathlib import Path
import pytest
from zen_garden.__main__ import run_module

def find_dataset_files():
    def extract_number(entry):
        number_str = entry.split('_', 1)[0]  # Split only at the first underscore
        return int(number_str)
    folder_path = os.path.dirname(__file__)
    all_entries = os.listdir(folder_path)
    pattern = re.compile(r'^\d+_.+$')
    filtered_entries = [entry for entry in all_entries if pattern.match(entry) and not entry.endswith("expectedfailure")]
    filtered_entries = sorted(filtered_entries, key=extract_number)
    dataset_dirs = []
    for entry in filtered_entries:
        path = os.path.join(folder_path, entry)
        if os.path.isdir(path) and ('system.py' in os.listdir(path) or 'system.json' in os.listdir(path)):
            dataset_dirs.append(path)

    return dataset_dirs

# All the tests
###############

@pytest.mark.parametrize("dataset_name",find_dataset_files())
def test_dataset(dataset_name):
    dataset_main_path = Path(dataset_name).parent.absolute()
    config = str(dataset_main_path / "config.json")
    run_module(["--config",config,"--dataset", dataset_name])

if __name__ == "__main__":
    find_dataset_files()
