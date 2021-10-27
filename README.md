# Energy-Carbon-Optimization-Platform

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

## Creation of issues
Apart from the first stages of development, very specific issues should be opened when the need of improvements occur.
When an issue is opened, it should be assigned to one or more developers and it should be classified according to the typology of issue (e.g. documentation, improvement).

## Coding rules
### Classes
* the name of the classes should always be with the first capital letter
* classes should all have a short description of what they do, e.g.:

        """
        This class creates the dictionary containing all the input data
        :param analysis: dictionary defining the analysis framework
        :return: dictionary containing all the input data
        """

### Methods
* the name of the methods should always be in lower case letters
* the name can be composed by two words, in that case the second word should be capitalised e.g. addVariables()
* main methods should all have a short desciption of what they do, e.g.:

        """
        This method creates a dictionary with the paths of the data split
        by carriers, networks, technologies
        :param analysis: dictionary defining the analysis framework
        :return: dictionary all the paths for reading data
        """

### Comments
* comments are located above the line of code they refer to

### File header
* the file header is the same in the files
* files containing main components of the tool, e.g. preprocess, core, postprocess have an additional heading line in capital letters followd by a separator "# ====="

### Variables name
* the name of the variables should always be in lower case letters
* the name can be composed by two words, in that case the second word should be capitalised e.g. myVariable

### Files name
* the files name should always be lower case
* the name can be composed by multiple words, seprated by underscores

### Folders name
* the name of the folders should always be lower case
* the name can be composed by multiple words, seprated by underscores
