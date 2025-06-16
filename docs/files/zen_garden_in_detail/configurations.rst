.. _configuration.configuration:

################
Configurations
################


.. _configuration.system:

System Configurations
=====================

The ``system.json`` defines the structure of the energy system. The following 
table summarizes the available system settings and their default values.

.. csv-table:: System Settings
    :header-rows: 1
    :file: tables/system_settings.csv
    :widths: 10 10 10 20
    :delim: ;


Per default, the technology selection is empty. You must define the set of 
technologies you want to investigate in your system. Only technologies selected 
in ``system.json`` are added to the optimization problem. You can flexibly 
select any subset of the technologies available in your ``set_technologies`` 
folder. Selecting a technology that is not defined in your input data will raise 
an error.

Per default, all nodes defined in ``energy_system/set_nodes.csv`` are added.
You can reduce the number of nodes by selecting a subset of nodes in your 
``system.json``. In addition, you can specify the starting year 
(``reference_year``), the time horizon (``optimized_years``), and the 
interyearly resolution (``interval_between_years``) in the ``system.json``.

Per default, each year is represented by 8760 timesteps of length 1h.
You can change the interyearly resolution by modifying the 
``unaggregated_time_steps_per_year``. To reduce the complexity, timeseries 
aggregation can be used (``conduct_time_series_aggregation``) to reduce the 
number of time steps. Per default, the number of timesteps is reduced to 10 
(``aggregated_time_steps_per_year``). :ref:`tsa.tsa` and :ref:`tsa.time_parameters` provide a detailed description of 
the time representation and the time parameters.


.. _configuration.config:

Config Configurations
=====================

This section describes all configurations which can be set in the 
``config.json`` file. The ``config.json`` file generally contains a list 
of two dictionaries: ``analysis`` and ``solver``. Each of these dictionaries
contains configurations which the user can specify. The lists below contain
a complete list of configurations for each dictionary.

.. _configuration.analysis:

Analysis
--------

The dataset, the objective function and the solver are selected in the 
``analysis`` dictionary of ``config.json``. The following table summarizes the 
available ``analysis`` settings and their default values:

.. csv-table:: Analysis Settings
    :header-rows: 1
    :file: tables/analysis_settings.csv
    :widths: 10 10 10 20
    :delim: ;

The settings of the timeseries aggregation algorithm are also specified in the 
``analysis`` section. The following table summarizes the available timeseries 
aggregation settings and their default values. For further information on how to 
use the timeseries aggregation, see :ref:`tsa.using_the_tsa`. In addition, 
:ref:`tsa.tsa` and :ref:`tsa.time_parameters` 
provide helpful information on the time representation and the time parameters 
in ZEN-garden.

.. csv-table:: Timeseries Aggregation Settings
    :header-rows: 1
    :file: tables/tsa_settings.csv
    :widths: 10 10 10 20
    :delim: ;


.. _configuration.solver:

Solver
------

Solver settings are also specified in the ``config.json``. The following table 
summarizes the available solver settings and their default values.

.. csv-table:: Solver Settings
    :header-rows: 1
    :file: tables/solver_settings.csv
    :widths: 10 10 10 20
    :delim: ;

Per default the open-source solver `HiGHS <https://highs.dev/>`_ is used. You 
can change the solver by modifying the ``solver`` key. Solver-specific settings 
are passed via the ``solver_settings``. Please refer to the solver documentation 
for the available solver settings for the solver that you are using.

For linear optimization problems, the dual variables can be computed and saved 
by selecting ``save_duals=True``. Saving the duals helps understand the 
optimality of the solution, but it also strongly increases the file size of the 
output files. The parameters of the optimization problem can be saved by 
selecting ``save_parameters=True``. If you only want to save specific 
parameters, you can specify them in the ``selected_saved_parameters`` list. The 
same applies to the variables and duals, which can be specified in the
``selected_saved_variables`` and ``selected_saved_duals`` lists, respectively. The name of the duals corresponds to the name of the constraints.

.. note::

    Non-selected parameters, variables, and duals are not saved. We recommend to only
    use the option to skip saving parameters, variables, and duals if you are sure that
    you do not need them. The visualization platform may not work properly if 
    you do not save the parameters and variables.

You can analyze the numerics of your optimization problem via 
``analyze_numerics``. In addition, a scaling algorithm is available. Per 
default, four iterations of the scaling algorithm are conducted without 
including the values of the right-hand-side. :ref:`input_handling.scaling` provides a detailed 
description of the scaling algorithm.
