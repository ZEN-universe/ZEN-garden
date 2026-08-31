.. _t_scenario_tutorial.t_scenario_tutorial:

#################
Scenario analysis
#################

.. admonition:: At a glance
   :class: note

   | **You will** define several variants of one dataset, run them in a single command,
       and read the results back scenario by scenario.
   | **You need** the setup from :ref:`tutorials_intro.setup`.

Studies are rarely one model run. You want the same system under a high gas
price, under a tighter carbon budget, without a particular technology.

Copying the dataset once per variant works until the variants multiply, and
then it stops working: twelve near-identical folders drift apart, and you can
no longer say what distinguished run 7 from run 8.

The **scenario tool** keeps one dataset and describes the variations in a
single file.

.. seealso::
    This tutorial covers the common cases. The complete syntax, i.e., set expansion,
    list expansion, partial files, naming keys, is documented in
    :ref:`t_scenario.t_scenario`.


.. _t_scenario_tutorial.setup:

Setting up
==========

Two steps switch the scenario tool on.

1. Add a ``scenarios.json`` file to the dataset folder:

   .. code-block:: text

       <data>/
       |--<dataset>/
       |   |--energy_system/...
       |   |--set_carriers/...
       |   |--set_technologies/...
       |   |--scenarios.json
       |   `--system.json
       |
       `--config.json

2. Set ``"conduct_scenario_analysis": true`` in ``system.json``.

Run the model exactly as before; every scenario is run in turn:

.. code-block:: shell

    zen-garden --dataset="<dataset>"

Results appear under ``outputs/<dataset>/scenario_<scenario_name>/``: every
scenario folder, and every ``scenario_name`` you pass to the ``Results`` class,
carries a ``scenario_`` prefix.

.. note::
    By default the unmodified dataset is also run, as the default scenario,
    under the name ``scenario_`` (the prefix with an empty suffix). 
    Set ``"run_default_scenario": false`` in ``system.json`` if
    you only want the scenarios you defined. Check
    ``r.scenarios.keys()`` if you are ever unsure of the exact name.


.. _t_scenario_tutorial.four_ways:

Four ways to vary a parameter
=============================

A scenario names an element, a parameter, and how to change it:

.. code-block:: json

    {
      "high_gas_price": {
        "natural_gas": {
          "price_import": {
            "default_op": 1.5
          }
        }
      }
    }

The four you will use most often are:

``default_op``
    Multiply the default value from ``attributes.json`` by a factor. Best for
    "what if this were 50% higher".

``default``
    Read the default value from a different attributes file. ``"attributes_low"``
    reads ``attributes_low.json`` from the same folder. Best when a variant
    changes several parameters of one element together.

``file``
    Read the values from a different ``.csv``. ``"price_import_high"`` reads
    ``price_import_high.csv``. Best when the variation differs by node or year.

``file_op``
    Multiply the values **after** reading them from a ``.csv`` by a factor.

You can also vary ``system`` and ``analysis`` settings, which is how you change
the technology set or the objective:

.. code-block:: json

    {
      "no_gas_boiler": {
        "system": {
          "set_conversion_technologies": ["photovoltaics", "heat_pump"]
        }
      }
    }

.. warning::
    System and analysis values are type-checked against the existing setting. A
    list must stay a list, an integer an integer.


.. _t_scenario_tutorial.reading:

Reading scenario results
========================

One ``Results`` object holds all scenarios. Select one with ``scenario_name``:

.. code:: python

    from zen_garden import Results
    r = Results(path='<data>/outputs/<dataset>')

    capacity_base = r.get_total('capacity', scenario_name='scenario_')
    capacity_high = r.get_total('capacity', scenario_name='scenario_high_gas_price')

.. important::
    Every scenario name carries a ``scenario_`` prefix, including the default
    scenario, whose name is exactly ``scenario_``. A
    scenario you named ``"high_gas_price"`` in ``scenarios.json`` becomes
    ``"scenario_high_gas_price"`` here, and the folder it is written to is
    named the same way. Use ``r.scenarios.keys()`` to see the exact names
    rather than guessing them.

Exercises
=========

The exercises are cumulative. Work on a copy of
``5_multiple_time_steps_per_year``.

1. **Run three gas price scenarios.** Create ``scenarios.json``:

   .. code-block:: json

       {
         "gas_price_low": {
           "natural_gas": {
             "price_import": {"default_op": 0.5}
           }
         },
         "gas_price_high": {
           "natural_gas": {
             "price_import": {"default_op": 2.0}
           }
         }
       }

   Set ``"conduct_scenario_analysis": true`` and run.

   *Expected result: there are three result folders (*``scenario_``\ *,*
   ``scenario_gas_price_low``\ *,* ``scenario_gas_price_high``\ *): the
   default plus the two scenarios. Heat pump capacity increases as the gas price rises.* 

   *On this dataset: With the default gas price, only 0.00044 GW of heat pumps are 
   built. Halving the price changes nothing, reduces it to 0 GW. 
   Doubling the price, instead, increases it to 156 GW.*

2. **Add a scenario that changes the system rather than a parameter.** Add a
   scenario removing the gas boiler from
   ``set_conversion_technologies``.

   .. code-block:: json

       {
         "no_gas_boiler": {
           "system": {
             "set_conversion_technologies": ["photovoltaics", "heat_pump"]
           }
         }
       }

   *Expected result: four scenarios now run. The no-boiler scenario now produces the 
   entire heat through heat pumps (218 GW).*

4. **Generate a price sweep without writing four scenarios.** Replace the two
   price scenarios (you can remove the no-boiler scenario) with one that uses a list:

   .. code-block:: json

       {
         "gas_price_sweep": {
           "natural_gas": {
             "price_import": {
               "default_op": [0.5, 1.0, 1.5, 2.0],
               "default_op_fmt": "gas_price_{}"
             }
           }
         }
       }

   *Expected result: four sub-scenarios named* ``scenario_gas_price_0.5``
   *through* ``scenario_gas_price_2.0``\ *, in
   * ``outputs/<dataset>/scenario_gas_price_sweep/`` *alongside a*
   ``param_map.yml``. Watch out when combining lists: multiple lists in one scenario
   produce the cartesian product, and the count grows fast.*
   
   *On this dataset: The heat pump capacity grows from 0 GW when halving the price to 
   156 GW when doubling it, with intermediate values of 137 GW at a 50% higher price.*

.. seealso::
    :ref:`t_scenario.t_scenario` for the full syntax.
    :ref:`t_scenario.running_the_analysis` to run the scenarios in parallel
    on a cluster instead of one after another.
