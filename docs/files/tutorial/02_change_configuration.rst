.. _t_configuration.t_configuration:

#################################
Tutorial 2: Change Configurations
#################################


This tutorial describes how to change the default configurations of ZEN-garden.
Configurations allow the user to control various aspects of model behavior.
For example, they include options for: (a) which solver to use; (b) what
variables to save; (c) how aggregate time steps; (d) whether to scale the model
coefficients for enhanced numerical stability; (d) what technologies and regions
to include; (e) what objective function to use; and (f) whether to use a 
rolling foresight horizon. A complete list of all configurations can be found 
in the :ref:`configurations <configuration.configuration>`.


This tutorial assumes that you have installed and run the example dataset 
``5_multiple_time_steps_per_year`` as described in the tutorial :ref:`setup 
instructions <tutorials_intro.setup>`. 


Configuration files
===================

The configurations are set in two input files: ``config.json`` and
``system.json``. The ``config.json`` file sets all configurations which
relate to how the model should be processed, solved, and saved. For example,
it allows users to select: which solver to use, what variables to save, how
to aggregate time steps, and whether to scale the model. In contrast, the 
``system.json`` file includes all configurations which relate to the physical
energy system being simulated. These include: what technologies and regions to
include, what objective function to use, and whether to use a rolling foresight
horizon. The location of these files in the ZEN-garden input data is shown below:


.. code-block:: text

    <data>/
    |--<dataset>/
    |   |--energy_system/...
    |   |--set_carriers/...
    |   |--set_technologies/...
    |   `--system.json
    |
    `--config.json


The configuration files are formatted in JavaScript Object Notation (JSON),
an easy-to-read-and-interpret file format for representing objects and data 
structures. The following `video <https://www.youtube.com/watch?v=iiADhChRriM>`_
provides a introduction to JSON files. Users are recommended to familiarize 
themselves with the JSON files structure before continuing with this tutorial.

.. tip::

    A common mistake when writing JSON files is to put a comma at the end of a 
    list in json. For example, `"list": [1, 2, 3,]`` is wrong, it should be 
    ``"list": [1, 2, 3]``. The cryptic error message which results is:
    ``json.decoder.JSONDecodeError: Expecting value: [...]``. If you recieve
    this message, check the ''`system.json``, ``config.json``, and 
    ``attributes.json`` for commas at the end of lists. When you scroll up in 
    the error message, it will tell you which file caused the error.


Modifying config.json
=====================

The ``config.json`` file includes creates a dictionary with two entries:
``analysis`` and ``solver``. Each of these is, in turn, a dictionary containing
configurations and their values. A full set of options which can be specified in
the ``config.json`` file are found in :ref:`analysis settings 
<configuration.analysis>` and :ref:`solver settings <configuration.solver>`.


The following steps can be used to change the ``config.json`` file:

1. Identify which configurations you would like to change. To do so, see
   the complete list of :ref:`config.json configurations 
   <configuration.config>`.
2. Set the desired configuration in the ``conf.json`` file. An example 
   ``conf.json`` file, in which multiple configurations are specified, is shown 
   below. The example file is not exhaustive of all of the available 
   configurations. Instead, it is intended to give users an intuition for how 
   configurations can be specified. To add configurations which are not 
   already listed, simply add the desired ``<configuration_name>: <value>`` to 
   the appropriate dictionary (``analysis`` or ``solver``).


.. code:: JSON

    {
      "analysis": {
        "dataset": "5_multiple_time_steps_per_year"
      },
      "solver": {
        "name": "gurobi",
        "solver_options": {
          "Method": 2,
          "NodeMethod": 2,
          "BarHomogeneous": 1,
          "DualReductions": 0,
          "Threads": 128,
          "Crossover": 0,
          "ScaleFlag": 2,
          "BarOrder": 0
        },
        "save_duals": false,
        "use_scaling": false,
        "run_diagnostics": true,
        "scaling_include_rhs": true
      }
    }


Example Exercise
----------------


1. **Modify the default ``conf.py`` file from the dataset example
   ``5_multiple_time_steps_per_year`` in order to save the dual variables
   to the outputs. Note: by default, dual variables are not saved to reduce
   the memory requirement of the solution**


   a. Identify the appropriate setting which needs to be changed by reading
      through the options in the :ref:`configurations 
      <configuration.configuration>`. The option for saving duals is located in
      the solver settings and called ``save_duals``. It takes a boolean value 
      as input.

   b. Add the ``save_duals`` to the ``config.json`` file. The new file should
      look like this:

      .. code:: JSON

         {
           "analysis": {
             "dataset": "5_multiple_time_steps_per_year"
           },
           "solver": {
             "save_duals": true
           }
         }

   c. You can verify that the dual variables were saved running the model and
      using the results codebase described in the tutorial on :ref:`analyzing 
      outputs <t_analyze.t_analyze>`. The list of components should now
      include duals variables, whose name begins with ``constraint_<...>``.

Modifying system.json
=====================

The ``system.json`` file contains a single dictionary of all the system 
configurations of ZEN-garden. Similar to the ``conf.json``, these configurations 
can be adjusted to match user preferences. Importantly, the ``system.json`` file 
lists which technologies and regions are to be included in the model. It also 
controls the temporal resolution of the model and sets parameters for spatial 
aggregation. The ``system.json`` file which comes with the dataset example
``5_multiple_time_steps_per_year`` is shown below:


.. code:: JSON
    
    {
        "set_conversion_technologies": [
            "natural_gas_boiler",
            "photovoltaics",
            "heat_pump"
        ],
        "set_storage_technologies": [
            "natural_gas_storage"
        ],
        "set_transport_technologies": [
            "natural_gas_pipeline"
        ],
        "set_nodes": [
            "DE",
            "CH"
        ],
        "reference_year": 2023,
        "unaggregated_time_steps_per_year": 96,
        "aggregated_time_steps_per_year": 96,
        "conduct_time_series_aggregation": false,
        "optimized_years": 3,
        "interval_between_years": 1,
        "use_rolling_horizon": false,
        "years_in_rolling_horizon": 1
    }


To modify the system configurations, use the following steps:

1. Identify which configurations you would like to change. To do so, see
   the complete list of :ref:`system.json configurations 
   <configuration.system>`.

2. Set the desired configuration in the ``system.json`` file. To add 
   configurations which are not already listed, simply add the desired 
   ``<configuration_name>: <value>`` to the JSON file.

Example Exercise
----------------

1. **Remove the natural gas boiler from the system. What heat pump capacity
   is installed in Switzerland in 2023 to meet the heat demand?**

   a. Open the ``system.json`` file for the ``5_multiple_time_steps_per_year``
      dataset. Under the option ``set_conversion_technologies``, delete the 
      line containing the ``natural_gas_boiler``. Save the file.

   b. Run ZEN-garden by following the instructions on :ref:`running a model 
      <running.running>`
   c. View the heat pump capacity using the ZEN-garden visualization platform,
      as described in the tutorial on :ref:`analyzing outputs 
      <t_analyze.t_analyze>`.

   `Solution: 31.0 GW`

1. **Using the above model (without natural gas boilers), invoke time-series 
   aggregation to represent the system in only 10 representative hours. What
   is the new heat pump capacity installed in Switzerland in 2023? How did 
   the heat demand profile change?**

   a. Open the ``system.json`` file for the ``5_multiple_time_steps_per_year``
      dataset. Change the configuration of ``conduct_time_series_aggregation``
      to ``true``. This tells ZEN-garden to use time series aggregation. Then,
      change the configuration of ``aggregated_time_steps_per_year`` to 10. This
      specifies the number of representative hours used. Save the file. For 
      more detailed information on time series aggregation and available options,
      see the documentation on :ref:`time series aggregation <tsa.tsa>`.

   b. Run ZEN-garden by following the instructions on :ref:`running a model 
      <running.running>`

   c. View the heat pump capacity using the ZEN-garden visualization platform,
      as described in the tutorial on :ref:`analyzing outputs 
      <t_analyze.t_analyze>`. Similarly, you can also view the heat demand 
      profile by looking at the hourly energy balance.

   `Solution: 30.0 GW. The new heat demand profile is less smooth. Blocks of
   multiple hours often have the same heat demand. This is because the entire 
   demand profile can now only be represented by ten different demand steps.`
