.. _installation.installation:

############
Installation
############

.. note::
    This section is written for ZEN-garden users who exclusively use the model 
    as a black-box. Users have no access to the code base and must work within 
    existing functionalities. Alternatively,  see the :ref:`installation guide 
    for developers <dev_install.dev_install>` for instructions on how to fork 
    the ZEN-garden repository in a way that allows you to modify and contribute 
    to the code.

ZEN-garden is written in Python and is available as a package. You can install 
the package via `pip <https://pypi.org/project/zen-garden/>`_ in a terminal or 
command prompt.

We recommend working from a conda environment for the installation. If you have 
not installed Anaconda, you can download it from the 
`Anaconda website <https://docs.anaconda.com/anaconda/install/>`_. You can check 
if you have Anaconda installed by running the following command in  a terminal 
(MacOS)/command prompt (Windows)::

    conda --version

You can quickly create an environment with the following command::

  conda create -n <your_env_name> python==3.12

Replace ``<your_env_name>`` with the name of your environment.

.. warning::
    Gurobi currently does not support Python version 3.13. We therefore 
    recommend using Python 3.12.

Activate the environment with the following command::

  conda activate <your_env_name>

Now you can install the zen-garden package with the following command::

    pip install zen-garden

To test whether the installation was successful, type:

.. code::

    conda list
    
into the command prompt. This will print a list of all installed packages. You 
should see ``zen_garden`` in the list.


.. _installation.activate:

Activate Conda environment
==========================

After installing ZEN-garden, you need to activate the ZEN-garden environment 
each time you open a new terminal. To activate the environment, type

.. code::

    conda activate <your_environment_name>  

into the terminal in which you would like to run ZEN-garden. At any time, you 
can deactivate the environment by typing: 

.. code::

    conda deactivate


.. _installation.solver:

Install a Solver
================

To run ZEN-garden, you also need to install an optimization solver. Two options
are:

1. `HiGHS <https://highs.dev/>`_
2. `Gurobi <https://www.gurobi.com/>`_

Per default, ZEN-garden selects the open source solver HiGHS. Academic licenses 
for Gurobi are available for free and allow you to access all of Gurobi's 
functionalities. You can get your Gurobi license from the 
`Gurobi webpage <https://www.gurobi.com/features/academic-named-user-license/>`_. 
Follow the instructions to retrieve your Gurobi license key and activate the 
license for your computer.

.. important::
    If you are planning to use Gurobi, make sure that the version of your Gurobi 
    solver license and your Gurobi installation align.