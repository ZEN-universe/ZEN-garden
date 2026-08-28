.. _t_configuration.t_configuration:

######################
Change configurations
######################

.. admonition:: At a glance
   :class: note

   | **You will** change ZEN-garden's behaviour by editing ``config.json`` and ``system.json``.
   | **You need** the setup from :ref:`tutorials_intro.setup`.

Configurations control how the model is solved and what it represents. They
live in two files:

* ``config.json`` — how the model is **processed, solved and saved**: which
  solver to use, what to save, how to aggregate time steps, whether to scale.
* ``system.json`` — the **physical energy system**: which technologies and
  nodes to include, how many years, the objective function, whether to use a
  rolling foresight horizon.

.. code-block:: text

    <data>/
    |--<dataset>/
    |   |--energy_system/...
    |   |--set_carriers/...
    |   |--set_technologies/...
    |   `--system.yaml
    |
    `--config.yaml

Note that ``config.json`` sits next to the dataset folder, not inside it: one
``config.json`` can serve several datasets.

The complete list of available settings, with types and defaults, is in
:ref:`configuration.configuration`. This tutorial shows how to change them.

.. warning::

    A common mistake is a trailing comma at the end of a JSON list:
    ``"list": [1, 2, 3,]`` is invalid, it should be ``"list": [1, 2, 3]``.
    JSON also does not allow comments. Both produce the error
    ``json.decoder.JSONDecodeError: Expecting value: [...]``. See
    :ref:`troubleshooting.troubleshooting`.


.. _t_configuration.config:

Modifying config.yaml
=====================

``config.json`` contains two dictionaries, ``analysis`` and ``solver``. To
change a setting, add ``"<configuration_name>": <value>`` to the appropriate
dictionary. Anything you do not specify keeps its default.

The example below is not exhaustive; it shows the shape of the file.

.. code:: json

    {
      "analysis": {
        "dataset": "5_multiple_time_steps_per_year"
      },
      "solver": {
        "name": "gurobi",
        "solver_options": {
          "Method": 2,
          "BarHomogeneous": 1,
          "DualReductions": 0,
          "Threads": 128,
          "Crossover": 0
        },
        "save_duals": false,
        "use_scaling": true,
        "run_diagnostics": true,
        "scaling_include_rhs": true
      }
    }

The available settings are listed in :ref:`configuration.analysis` and
:ref:`configuration.solver`.

.. warning::
    Settings are validated against a schema. A misspelled key is rejected with
    a validation error rather than silently ignored — which is useful, but it
    means you cannot leave notes to yourself as unused keys.


Exercise
--------

1. **Save the dual variables to the outputs.** By default, duals are not saved,
   which keeps the result files small.

   a. Find the setting. Duals are a solver concern, so look in
      :ref:`configuration.solver`. The setting is ``save_duals`` and it takes
      a boolean.

   b. Add it to ``config.json``:

      .. code:: json

         {
           "analysis": {
             "dataset": "5_multiple_time_steps_per_year"
           },
           "solver": {
             "save_duals": true
           }
         }

   c. Re-run the model and load the results as in
      :ref:`t_analyze.t_analyze`. ``r.get_component_names('dual')`` now returns
      a non-empty list, and every name begins with ``constraint_``.

   *Expected result: the dual of the nodal energy balance,
   ``constraint_nodal_energy_balance``, is now available. It is the marginal
   price of a carrier at a node and time step — the value the visualization
   platform shows alongside the energy balance.*

   To save duals for selected constraints only, see :ref:`t_output.t_output`.


.. _t_configuration.system:

Modifying system.json
=====================

``system.json`` is a single dictionary describing the energy system. It lists
the technologies and nodes to include, and controls the temporal resolution.
The file shipped with ``5_multiple_time_steps_per_year`` is:

.. code:: json

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

Only technologies listed here enter the optimization, even if more are defined
in ``set_technologies``. The same is true for nodes. The full list of settings
is in :ref:`configuration.system`.


Exercises
---------

The two exercises below are cumulative: the second continues from the dataset
you changed in the first.

1. **Remove the natural gas boiler from the system. What heat pump capacity is
   then installed in Switzerland in 2023?**

   a. In ``system.json``, delete ``"natural_gas_boiler"`` from
      ``set_conversion_technologies``. Save the file.
   b. Run the model (:ref:`running.run_model`).
   c. Read the heat pump capacity, either in the visualization platform or with
      the ``Results`` class (:ref:`t_analyze.t_analyze`).

   *Solution: 31.0 GW. With the boiler gone, the heat pump is the only way to
   supply heat, and it takes over exactly the capacity the boiler had.*

2. **Continuing from exercise 1, represent the system with only 10
   representative time steps. What is the new heat pump capacity in
   Switzerland in 2023, and how did the heat demand profile change?**

   a. In ``system.json``, set ``"conduct_time_series_aggregation": true`` and
      ``"aggregated_time_steps_per_year": 10``. Save the file.
   b. Run the model.
   c. Read the heat pump capacity, and look at the hourly energy balance to see
      the demand profile.

   *Solution: 30.0 GW. The demand profile is less smooth — blocks of hours now
   share the same demand value, because the whole profile is represented by ten
   distinct steps. The capacity is slightly lower because clustering has
   smoothed away part of the peak.*

   This is the central trade-off of time series aggregation, and
   :ref:`t_tsa.t_tsa` explores it properly.
