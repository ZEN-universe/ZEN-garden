# ZEN-garden
![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FZEN-universe%2FZEN-garden%2Fmain%2Fpyproject.toml)

[![GitHub Release](https://img.shields.io/github/v/release/ZEN-universe/ZEN-garden)](https://github.com/ZEN-universe/ZEN-garden/releases)
[![PyPI - Version](https://img.shields.io/pypi/v/zen-garden)](https://pypi.org/project/zen-garden/)

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/ZEN-universe/ZEN-garden/pytest_with_conda.yml)](https://github.com/ZEN-universe/ZEN-garden/actions)
[![Endpoint Badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jacob-mannhardt/30d479a5b4c591a63b7b0f41abbce6a0/raw/zen_garden_coverage.json)](https://github.com/ZEN-universe/ZEN-garden/actions)
[![Read the Docs](https://img.shields.io/readthedocs/zen-garden?logo=readthedocs)](https://zen-garden.readthedocs.io/en/latest/index.html)

[![GitHub forks](https://img.shields.io/github/forks/ZEN-universe/ZEN-garden)](https://github.com/ZEN-universe/ZEN-garden/forks)

<img src="https://github.com/ZEN-universe/ZEN-garden/assets/114185605/d6a9aca9-74b0-4a82-8295-43e6a78b8450" alt="drawing" width="200"/>

Welcome to the ZEN-garden! ZEN-garden is an optimization model of energy systems and value chains. 
It is currently used to model the electricity system, hydrogen value chains, and carbon capture, storage and utilization (CCUS) value chains. 
However, it is designed to be modular and flexible, and can be extended to model other types of energy systems, value chains or other network-based systems. 

ZEN-garden is developed by the [Reliability and Risk Engineering Laboratory](https://www.rre.ethz.ch/) at ETH Zurich.
<hr style="height: 5px; background-color: black;">

## Quick Start
To get started with ZEN-garden, you can follow the instructions in the [installation guide](https://zen-garden.readthedocs.io/en/latest/files/user_guide/installation.html).

If you want to use ZEN-garden without working on the codebase, run the following command:
```bash
pip install zen-garden
```
If you want to work on the codebase, fork and clone the repository and install the package in editable mode. More information on how to install the package in editable mode can be found in the [installation guide](https://zen-garden.readthedocs.io/en/latest/files/user_guide/installation.html).

## Documentation
Please refer to the documentation of the ZEN-garden framework [on Read-the-Docs](https://zen-garden.readthedocs.io/en/latest/). 

In the file `documentation/how_to_ZEN-garden.md`, you can find additional information on how to use the framework. 
The `documentation/dataset_creation_tutorial.md` file contains a tutorial on how to create a simple dataset for the framework. 
Additionally, example datasets are available in the `dataset_examples` folder.

More in-depth manuals are available in the [discussions forum](https://github.com/ZEN-universe/ZEN-garden/discussions) of our repo.

## News
Review recent modifications outlined in the [changelog](https://github.com/ZEN-universe/ZEN-garden/blob/main/CHANGELOG.md).

## Citing ZEN-garden
If you use ZEN-garden for research, please cite

Jacob Mannhardt, Alissa Ganter, Johannes Burger, Francesco De Marco, Lukas Kunz, Lukas Schmidt-Engelbertz, Paolo Gabrielli, Giovanni Sansavini (2025).
ZEN-garden: Optimizing energy transition pathways with user-oriented data handling. https://www.sciencedirect.com/science/article/pii/S2352711025000263

and use the following BibTeX:
```
@article{ZENgarden2025,
title = {ZEN-garden: Optimizing Energy Transition Pathways with User-Oriented Data Handling},
author = {Mannhardt, Jacob and Ganter, Alissa and Burger, Johannes and De Marco, Francesco and Kunz, Lukas and {Schmidt-Engelbertz}, Lukas and Gabrielli, Paolo and Sansavini, Giovanni},
year = {2025},
journal = {SoftwareX},
volume = {29},
pages = {102059},
issn = {2352-7110},
doi = {10.1016/j.softx.2025.102059},
}
```
