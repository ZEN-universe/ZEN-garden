.. _Installation:

############
Installation
############

.. note::
    This section is written for ZEN-garden users who exclusively use the model as a black-box. Users have no access to the code base and must work within existing functionalities. Alternatively,  see :ref:`Install for developers` for instructions on how to fork the ZEN-garden repository in a way that allows you to modify and contribute to the code.

..  _install for users:
Installation for Users
=======================

ZEN-garden is written in Python and is available as a package. You can install the package via `pip <https://pypi.org/project/zen-garden/>`_ in a terminal or command prompt.

We recommend working from a conda environment for the installation. If you have not installed Anaconda, you can download it `here <https://docs.anaconda.com/anaconda/install/>`_. You can check if you have Anaconda installed by running the following command in a terminal (MacOS)/command prompt (Windows)::

    conda --version

You can quickly create an environment with the following command::

  conda create -n <your_env_name> python==3.12

Replace ``<your_env_name>`` with the name of your environment and replace ``<python_version>`` with the necessary python version (see `ZEN-garden README <https://github.com/ZEN-universe/ZEN-garden/blob/main/README.md>`_).

.. warning::
    Gurobi currently does not support Python version 3.13. We therefore recommend using Python 3.12.

Activate the environment with the following command::

  conda activate <your_env_name>

Now you can install the zen-garden package with the following command::

    pip install zen-garden

.. note::

    To get started with small example models, check out :ref:`Running ZEN-garden as a Model-User`.

Solver options
==============

ZEN-garden passes the optimization problem to an external solver, per default, the open source solver `HiGHS <https://highs.dev/>`_ is selected. Alternatively, the commercial solver `Gurobi <https://www.gurobi.com/>`_ can be used. Academic licenses are available for free and allow you to access all of Gurobi's functionalities. You can get your Gurobi license `here <https://www.gurobi.com/features/academic-named-user-license/>`_. Follow the instructions to retrieve your Gurobi license key and activate the license for your computer.

.. warning::
    If you are planning to use Gurobi, make sure that the version of your Gurobi solver license and your Gurobi installation align.