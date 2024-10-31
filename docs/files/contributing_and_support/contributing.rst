################
How to contribute 
################
We welcome any contribution to the ZEN-garden package. Many thanks for contributing to the project and helping to maintain our ZEN-garden!
This document provides guidelines on how to contribute to the project.

**How to contribute**

* Reporting bugs
* Suggesting new features
* Improving the documentation
* Writing tests
* Fixing bugs/solving issues
* Implementing new features

**Contribution workflow**

1. Fork and clone the repository
2. Get started with ZEN-garden
3. Write your code
4. Add and run tests
5. Push your changes to your fork and create a pull request on GitHub

Creating Issues
=================
If you find a bug, have a feature request or want to suggest an improvement, please create an issue in the `GitHub repository <https://github.com/ZEN-universe/ZEN-garden/issues>`_.

When creating an issue, please follow these guidelines:

* The title should be short and descriptive.
* The description should contain all the necessary information to understand the issue.
* If it is a bug, please provide a minimal working example that reproduces the bug.
* Classify the issue according to the typology of issue (e.g. documentation, enhancement, bug).

Running tests
=================

After implementing a new feature or fixing a bug, it is important to run the tests to ensure that the changes do not break the existing code.
The tests are located in the ``tests`` folder and are written using the `pytest <https://docs.pytest.org/en/stable/>`_ framework.
If you add new functionalities, make sure to add a new test that covers the new code.

You can execute the tests by running::

  coverage run --source="zen_garden" -m pytest -v tests/testcases/run_test.py

The coverage report is also generated in a similar way by running::

  coverage report -m

**Pycharm configuration**

To run the tests, add another Python configuration. The important settings are:

- Change "Script Path" to "Module name" and set it to "coverage"
- Set the "Parameters" to: ``run --source="zen_garden" -m pytest -v run_test.py``
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

We follow the `PEP-8 <https://peps.python.org/pep-0008/>`_ coding style.

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


Defining the unit dimensions when adding a new parameter/variable to the framework
====================================================================================

Parameters
----------
The argument ``unit_category`` specifies the unit dimensions of the parameter and must be passed to the ``extrect_input_data`` function, e.g., for _capacity_addition_min_ the ``unit_category`` is defined as ``{"energy_quantity": 1, "time": -1}`` since a technology capacity is per definition given as energy_quantity (e.g. MWh) per time (hour), i.e. e.g. MW
``self.capacity_addition_min = self.data_input.extract_input_data("capacity_addition_min", index_sets=[], unit_category={"energy_quantity": 1, "time": -1})``

Variables
---------
Since the units of variables are not defined by the user but are a consequence of the parameter units as explained above, their unit dimensions are specified in the ``add_variable`` functions of the class ``Variable``. Again, the argument ``unit_category`` is used to define the unit dimensionality.
``variables.add_variable(model, name="capacity", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup), bounds=capacity_bounds, doc='size of installed technology at location l and time t', unit_category={"energy_quantity": 1, "time": -1})``
