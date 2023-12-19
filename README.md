<table><tr><td valign="center"> 
  <img align="left" height="25px" src="https://github.com/RRE-ETH/ZEN-garden/actions/workflows/pytest_with_conda.yml/badge.svg?branch=development"> 
  <img align="left" height="25px" src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jafluri/5d0d79e86182cd9ccd785d824b1f9ac7/raw/zen_coverage.json">
  <b> (Development Branch) </b>
</td></tr></table>

# ZEN-garden

Welcome to the ZEN-garden! ZEN-garden is an optimization model of energy systems and value chains. 
It is currently used to model the electricity system, hydrogen value chains, and carbon capture, storage and utilization (CCUS) value chains. 
However, it is designed to be modular and flexible, and can be extended to model other types of energy systems, value chains or other network-based systems. 

ZEN-garden is developed by the [Reliability and Risk Engineering Labratory](https://www.rre.ethz.ch/) at ETH Zurich.

## Installation
ZEN-garden is written in Python and is available as a package. To install the package, run the following command in the terminal while in the root directory of the repository:

```
pip install -e
```

## Documentation
The documentation of the ZEN-garden framework is available [here](https://github.com/ZEN-universe/ZEN-garden/tree/3fa27aebe1b8ec463d9470de5ff5df36e3f2e372/documentation). 
In the file `how_to_ZEN-garden.pdf`, you can find a step-by-step guide on how to use the framework. 
The `dataset_creation_tutorial.pdf` file contains a tutorial on how to create a simple dataset for the framework. 
Additionally, example datasets are available in the `dataset_examples` folder.

More in-depth manuals are available in the [discussions forum](https://github.com/ZEN-universe/ZEN-garden/discussions) of our repo.

## Modifications to the code
Multiple developers are working in parallel. To allow a smooth cooperation, the following procedures should be followed:
* any modification should be done in the "development" branch. If it doesn't exist, it should be created from the "main" branch
* before any action, esure that the remote branch of "development" is updated with the latest version: 
```
git pull
```
* for any modification, every developer should create a new branch of "development" to modify the code
* this can be done from the "development" branch with: 
```
git branch new_branch_name
```
Once your changes are done, you may want to push them to the development branch. To prevent accidental changes, we have protected the "main" and "development" branches. 
Therefore, you will need to push your changes to your branch and create a pull request to merge your changes into the development branch. 
Please provide details regarding the pull request, including a potential issue number that the pull request is addressing. 
One of the admins will take care of the pull request afterwards.

## Creating issues
Apart from the first stages of development, very specific issues should be opened when the need of improvements occur.
When an issue is opened, it should be assigned to one or more developers and it should be classified according to the typology of issue (e.g. documentation, improvement).

## Coding rules
We follow the [PEP-8](https://peps.python.org/pep-0008/) coding style.

### Classes
* the name of the classes should always be with the first capital letter
* classes must all have a short description of what they do (right beneath the class name) and a second docstring describing the constructor along with its parameters (blank line between description and parameters is mandatory), e.g.:

    ```python
    class Results(object):
      """
      This class reads in the results after the pipeline has run
      """

      def __init__(self, path, scenarios=None, load_opt=False):
          """
          Initializes the Results class with a given path
  
          :param path: Path to the output of the optimization problem
          :param scenarios: A None, str or tuple of scenarios to load, defaults to all scenarios
          :param load_opt: Optionally load the opt dictionary as well
          """

### Methods
* the name of the methods should always be in lower case letters
* the name can be composed by multiple words, seprated by underscores
* main methods should all have a short desciption of what they do (again, the blank line is mandatory), e.g.:

        """
        This method creates a dictionary with the paths of the data split
        by carriers, networks, technologies

        :param analysis: dictionary defining the analysis framework
        :return: dictionary all the paths for reading data
        """

### Comments
* comments are located above the line of code they refer to

### File header
* all files contain a header which contains the following information: 


        """
        :Title:        ZEN-GARDEN
        :Created:      month-20yy
        :Authors:      Jane Doe (jdoe@ethz.ch)
        :Organization: Labratory of Reliability and Risk Engineering, ETH Zurich

        Class defining...; the class takes as inputs...; the class returns ...
        """

### Variables name
* the variable name should always be lower case
* the name can be composed by multiple words, separated by underscores

### Files name
* the files name should always be lower case
* the name can be composed by multiple words, separated by underscores

### Folders name
* the name of the folders should always be lower case
* the name can be composed by multiple words, separated by underscores
