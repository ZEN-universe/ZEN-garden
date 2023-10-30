<table><tr><td valign="center"> 
  <img align="left" height="25px" src="https://github.com/RRE-ETH/ZEN-garden/actions/workflows/pytest_with_conda.yml/badge.svg?branch=development"> 
  <img align="left" height="25px" src="https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/jafluri/5d0d79e86182cd9ccd785d824b1f9ac7/raw/zen_coverage.json">
  <b> (Development Branch) </b>
</td></tr></table>

# ZEN-garden

Optimization model of European energy systems and value chains. It includes the electricity system, hydrogen value chains, and carbon capture, storage and utilization (CCUS) value chains. 

## Modifications to the code
Multiple developers are working in parallel. To allow a smooth cooperation, the following procedures should be followed:
* any modification should be done in the "development" branch. If it doesn't exist, it should be created from the "master" branch
* before any action, esure that the remote branch of "development" is updated with the latest version: 
```
git pull
```
* for any modification, every developer should create a new branch of "development" to modify the code
* this can be done from the "development" branch with: 
```
git branch new_branch_name
```
* once the commitment is done done, the branch should be pushed into development: 
```
git push origin new_branch_name
```
* a pull request is automatically generated in the online github repo. The branch can be modified and merged into development
* at the stage of merging, it's possible to reference to an existing issue with '#issue_number'
* the reference of the merging to the issue can also be done in a second stage, with the generated '#commit_number'

## Discussion forum
Please refer to the [Discussion Forum](https://github.com/RRE-ETH/ZEN-garden/discussions) for the description of several functionalities of the ZEN-garden framework.

## Creating issues
Apart from the first stages of development, very specific issues should be opened when the need of improvements occur.
When an issue is opened, it should be assigned to one or more developers and it should be classified according to the typology of issue (e.g. documentation, improvement).

## Coding rules
We follow the [PEP-8](https://peps.python.org/pep-0008/)
 coding style.
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
