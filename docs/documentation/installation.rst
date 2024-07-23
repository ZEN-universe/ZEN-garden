################
Installation
################

ZEN-garden is written in Python and is available as a package. 


Installing python
==============

If it is your first time using Python, we recommend using `conda <https://docs.conda.io/en/latest/miniconda.html>`_ as the package manager and setting up ZEN-garden as a conda environment. Furthermore, it is helpful to install an integrated development environment (IDE), such as `PyCharm <https://www.jetbrains.com/pycharm/download/>`_. Most users of ZEN-garden use PyCharm, but other IDEs such as `VS Code <https://code.visualstudio.com/>`_ work as well. 


Installing ZEN-garden 
==============

If it's your first time using GitHub, register at `<https://github.com/>`_. Login to Github and create a fork of the ZEN-garden repository. 

Navigate to ``ZEN-garden`` on Github and click on the "Fork" button at the top right corner of the page to create a copy of the repository under your account and select yourself as the owner.

.. image:: ../images/create_fork.png
    :alt: creating a fork

**Clone your forked repository:**

Clone your forked repository by running the following lines in Git-Bash::

    git clone git@github.com:your-username/ZEN-garden.git
    cd ZEN-garden

.. note::
    If you get the permissions error "Permission denied (publickey)", you will need to create the SSH key. Follow the instructions on `how to generate an SSH key <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key>`_ and then how to add it to your account. You will not need to add the SSH key to the Agent, so only follow the first website until before `Adding your SSH key to the ssh-agent <https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#adding-your-ssh-key-to-the-ssh-agent>`_

**Track the upstream repository:**

Track the upstream repository by running the following lines in Git-Bash::

    git remote add upstream git@github.com:ZEN-universe/ZEN-garden.git
    git fetch upstream

**Create the ZEN-garden conda environment:**

Open the Anaconda prompt. Change the directory to the root directory of your local ZEN-garden repository where the file ``zen_garden_env.yml`` is located::

  cd <path_to_zen_garden_repo>

Now you can install the conda environment for zen-garden with the following command::

  conda env create -f zen_garden_env.yml

The installation may take a couple of minutes. If the installation was successful, you can see the environment at ``C:\Users\username \anaconda3\envs`` or wherever Anaconda is installed

.. note::
    We strongly recommend working with conda environments. When installing the zen-garden conda environment via the ``zen_garden_env.yml``, the zen-garden package, as well as all other dependencies, are installed automatically. 
    You can also install the zen-garden package directly by running the following command in the root directory of your repository: ``pip install -e``

Solver options
==============

ZEN-garden passes the optimization problem to an external solver, per default, the open source solver `HiGHS <https://highs.dev/>`_ is selected. Alternatively, the commercial solver `Gurobi <https://www.gurobi.com/>`_ can be used. Academic licenses are available for free and allow you to access all of Gurobi's functionalities. You can get your Gurobi license `here <https://www.gurobi.com/features/academic-named-user-license/>`_. Follow the instructions to retrieve your Gurobi license key and activate the license for your computer.

Create PyCharm Configurations
==============

To execute ZEN-garden with the PyCharm IDE you can use the configuration setup which can be found next to the run button, and click on "Edit configurations.." to edit or add a configuration. There are three configurations that are easy to setup and run the model, the tests, and get the coverage reports.

.. image:: ../images/pycharm_configuration.png
    :alt: creating zen-garden configurations in pycharm

**Running the module**

Add a new configuration by clicking on the "+" button on the top left corner of the window. Choose ´´Python´´ as a type. You can name the configuration however you like. The important settings are:

- Change "Script Path" to "Module name" and set it to "zen_garden"
- Set the Python interpreter to the Conda environment that was used to install the requirements and ZEN-garden as a package. Per default, the environment will be named "zen-garden-env". **Important**: This setup will only work for Conda environments that were also declared as such in PyCharm; if you set the path to the Python executable, you will have to create a new PyCharm interpreter first.
- Set the "Working directory" to the path that contains the ``config.json``, this directory will also be used to save the results.

In the end, your configuration to run ZEN-garden as a module should look similar to this:

.. image:: ../images/pycharm_run_module.png
    :alt: run module

**Running the tests**

To run the tests, add another Python configuration. The important settings are:

- Change "Script Path" to "Module name" and set it to "coverage"
- Set the "Parameters" to: ``´run --source="zen_garden" -m pytest -v run_test.py``
- Set the python interpreter to the Conda environment that was used to install the requirements and also has the package installed. **Important**: This setup will only work for Conda environments that were also declared as such in PyCharm; if you set the path to the Python executable yourself, you should create a new proper PyCharm interpreter.
- Set the "Working directory" to the directory ``tests/testcases`` of the repo.

In the end, your configuration to run the tests should look similar to this:

.. image:: ../images/pycharm_run_tests.png
    :alt: run tests

**Getting the coverage report**

To run the test and also get the coverage report, we use the pipeline settings of the configuration. Add another Python configuration and use the following settings:

- Change "Script Path" to "Module name" and set it to "coverage"
- Set the "Parameters" to ``report -m``
- Set the python interpreter to the Conda environment that was used to install the requirements and also has the package installed. *Important*: This setup will only work for Conda environments that were also declared as such in PyCharm; if you set the path to the Python executable yourself, you should create a new proper PyCharm interpreter.
- Set the "Working directory" to the base directory of the repo.
- Click on "Modify options", go to the section "Before launch", and select "Add run before launch" where you can now add the "Run Tests" configuration from above.

In the end, your configuration to run the coverage should look similar to this:

.. image:: ../images/pycharm_coverage.png
    :alt: run coverage

