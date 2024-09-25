################
How to contribute 
################
You can also execute the tests by running::

  coverage run --source="zen_garden" -m pytest -v tests/testcases/run_test.py

The coverage report is also generated in a similar way by running::

  coverage report -m

**Pycharm configuration**

To run the tests, add another Python configuration. The important settings are:

- Change "Script Path" to "Module name" and set it to "coverage"
- Set the "Parameters" to: ``´run --source="zen_garden" -m pytest -v run_test.py``
- Set the python interpreter to the Conda environment that was used to install the requirements and also has the package installed. **Important**: This setup will only work for Conda environments that were also declared as such in PyCharm; if you set the path to the Python executable yourself, you should create a new proper PyCharm interpreter.
- Set the "Working directory" to the directory ``tests/testcases`` of the repo.

In the end, your configuration to run the tests should look similar to this:

.. image:: ../images/pycharm_run_tests.png
    :alt: run tests

To run the test and also get the coverage report, we use the pipeline settings of the configuration. Add another Python configuration and use the following settings:

- Change "Script Path" to "Module name" and set it to "coverage"
- Set the "Parameters" to ``report -m``
- Set the python interpreter to the Conda environment that was used to install the requirements and also has the package installed. *Important*: This setup will only work for Conda environments that were also declared as such in PyCharm; if you set the path to the Python executable yourself, you should create a new proper PyCharm interpreter.
- Set the "Working directory" to the base directory of the repo.
- Click on "Modify options", go to the section "Before launch", and select "Add run before launch" where you can now add the "Run Tests" configuration from above.

In the end, your configuration to run the coverage should look similar to this:

.. image:: ../images/pycharm_coverage.png
    :alt: run coverage

Coding rules
=================
We follow the `PEP-8 <https://peps.python.org/pep-0008/)>`_ coding style.

**Classes**

* the name of the classes should always be with the first capital letter
* classes must all have a short description of what they do (right beneath the class name) and a second docstring describing the constructor along with its parameters (blank line between description and parameters is mandatory), e.g.:

.. code-block::

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

**Methods**

* the name of the methods should always be in lower case letters
* the name can be composed by multiple words, seprated by underscores
* main methods should all have a short desciption of what they do (again, the blank line is mandatory), e.g.:

.. code-block::

    """
    This method creates a dictionary with the paths of the data split
    by carriers, networks, technologies

    :param analysis: dictionary defining the analysis framework
    :return: dictionary all the paths for reading data
    """

**Comments**

* comments are located above the line of code they refer to

**File header**

* all files contain a header which contains the following information:

.. code-block::

    """
    :Title:        ZEN-GARDEN
    :Created:      month-20yy
    :Authors:      Jane Doe (jdoe@ethz.ch)
    :Organization: Labratory of Reliability and Risk Engineering, ETH Zurich

    Class defining...; the class takes as inputs...; the class returns ...
    """

**Variables name**

* the variable name should always be lower case
* the name can be composed by multiple words, separated by underscores

**Files name**

* the files name should always be lower case
* the name can be composed by multiple words, separated by underscores

**Folders name**

* the name of the folders should always be lower case
* the name can be composed by multiple words, separated by underscores

Creating Issues
=================
#TODO
Apart from the first stages of development, very specific issues should be opened when the need of improvements occur.
When an issue is opened, it should be assigned to one or more developers and it should be classified according to the typology of issue (e.g. documentation, improvement).



