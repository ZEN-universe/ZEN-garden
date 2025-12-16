.. _t_operation_only.t_operation_only:

########################################
Tutorial 8: Operation Scenarios
########################################

It is often helpful to test the results of a capacity-planning optimization
in various operational scenarios. Such tests allow users to: (1) evaluate
the robustness of their model results to various contingencies, (2) ensure that 
results obtained through time-series aggregation are feasibly on the full
hourly time-series, (3) and refine estimates of marginal carrier prices that 
are obtained through the dual variables. 

The ``zen-operation`` wrapper allows users to seamlessly run operational 
scenarios on the results of a previously model. 


.. tip:: 
   
   Fully customized operation scenarios can be created using the 
   ``allow_investment`` configuration in ``system.json``. When `false`, this
   configuration prohibits new capacity additions. All capacities, including 
   future capacities, must then be pre-specified in the model via 
   ``capacity_existing``.
    

.. _t_operation_only.setup:

Setup 
-----

The ``zen-operations`` requires users to specify a set of operational 
scenarios in a ``scenarios_op.json`` file. This new file uses the same 
syntax as the standard ZEN-garden scenario tool (refer to the 
:ref: `tutorial on scenario analysis<t_scenario.t_scenario>`). It should 
be placed into the `<dataset>`  folder of the model being run:

.. _t_operation.file_structure_operation:

.. code-block::

    <data_folder>/
    |--<dataset>/
    |   |--energy_system/...
    |   |--set_carriers/...
    |   |--set_technologies/...
    |   |--scenarios.json
    |   |--scenarios_op.json
    |   `--system.json
    |
    `--config.json


For example, the ``scenarios_op.json`` file tells ZEN-garden
to run an operational scenario with all 8760 hours per year. Such a scenario 
is useful for testing whether the results from time-series aggregation are 
feasible with a full hourly resolution:



.. code:: json

  {
    "full_year": {
      "system": {
        "conduct_time_series_aggregation": false,
        "aggregated_time_steps_per_year": 8760
      }
    }
  }

.. _t_operation_only.results:


Running Operational Scenarios
-----------------------------

The `zen-operation` wrapper requires that you first run the original model 
and have the model results available. This can be done using the standard 
zen-garden commands, as described in the section on 
:ref:`running a model:<running.running>`:

.. code:: shell

  cd "<data>"
  zen-garden --dataset="<dataset>"


Next, run then ``zen-operation`` wrapper by entering the command:

.. code:: shell

  zen-operation --dataset="<dataset>" --scenarios_op="scenarios_op.json"

The wrapper first creates a copy of the original model to use for the 
operational optimizations. The ``capacity_addition`` results of the original 
model are copied to ``capacity_existing`` in the new operational model. This 
ensures that the technology capacities in the operational model match the 
optimal capacities of the original model. Finally, the operational model is run
without capacity additions. Any scenarios specified in the ``scenarios_op`` file
are included in the operational simulations. For a detailed description of 
all options in the wrapper, see 
:func:`zen_garden.cli.zen_operation.cli_zen_operation`.

If the original capacity-planning model has multiple scenario, then the 
``zen-operations`` wrapper applies the operational scenarios to all of the
capacity planning scenarios. If the capacity planning model has 5 scenarios, 
and there are two operational scenarios, then the ``zen-operations`` wrapper 
will run :math:`5 \times 2 = 10` scenarios in total.

The ``zen-operation`` wrapper can also be run directly from a python script.
The code below replicates the command line commands presented previously:


.. code:: python

  import zen_garden
  import os 

  os.chdir("<data>")
  zen_garden.run(dataset="<dataset>")
  zen_garden.operation_scenarios(dataset="<dataset>", scenarios_op="scenarios_op.json")


Results
-------

The results will be available in the ``outputs`` folder of the current 
working directory. The results will be stored under the following dataset 
name: ``"<dataset>_<scenario>__operation"``. 


.. _t_operation_only.example:

Example
-------

Download the run the example dataset ``5_multiple_time_steps_per_year`` as described 
in the :ref:`tutorial setup <tutorials_intro.setup>`.  Use the ``zen-operation``
wrapper to answer the following question:

1. **Run an operation-only version of the example-dataset. Do the operational
   results match that of the original capacity-planning solution?**


   a. To answer this question, run the following code in a terminal window. The 
      ZEN-garden environment must be activated for these commands to work:

      .. code:: shell

         zen-garden --dataset="5_multiple_time_steps_per_year"
         zen-operation --dataset="5_multiple_time_steps_per_year"
         zen-visualization

      These commands do not specify a value of the ``--scenarios_op`` flag of
      ``zen-operation``. In this case, ZEN-garden will run only one operational 
      scenario with all parameter values equal to that of the original dataset.
      The results of the operational simulations will be located under the 
      name ``5_multiple_time_steps_per_year_none__operation``.

      The last command opens the ZEN-garden visualization platform. Explore the 
      results of the visualization platform to see the differences between the 
      two solutions.
    

   *Solution: The operational variables (e.g. production, emissions) have 
   the same values in the capacity-planning problem and the operation 
   scenario. This makes sense since the planning problem includes operational
   optimization. Similarly, the total installed capacity in each year is 
   the same in each problem. That said, the operation problem has no 
   capacity addition in any model year since all technology capacities 
   were specified ahead of time.* 


2. **Define a new operational scenario in which the system is operated to 
   minimize emissions instead of cost. Do the results change? From the dual 
   values of the nodal energy balance constraint, what marginal are associated
   with supplying one extra unit of electricity demand?**



   a. In the ``config.json`` file, set the solver configuration ``save_duals``
      to true. This ensures that the dual variables are saved in the results.
      
      .. code:: json

        {
          "analysis": {
            // Additional analysis fields go here
          },
          "solver": {
            // Additional solver fields go here
            "save_duals": true,
          }
        }

   b. Create a ``scenario_op`` file in the ``5_multiple_time_steps_per_year`` 
      folder. In the file, add a scenario that minimizes emissions instead 
      of costs. 

      .. code:: json

        {
          "min_emissions": {
            "analysis": {
              "objective": "total_carbon_emissions"
            }
          }
        }

   c. Run the operational scenario using the following code:

      .. code:: shell

      zen-garden --dataset="5_multiple_time_steps_per_year"
      zen-operation --dataset="5_multiple_time_steps_per_year" --scenarios_op="scenarios_op.json"
      zen-visualization



   *Solution: A price spike in the capacity planning model occurs in the 
   hour where electricity demand is highest. In this hour, adding
   additional capacity would require installing an additional unit of
   electricity generating capacity. These capacity costs are reflected
   in the price. In contrast, the operation-only model shows electricity 
   prices equal to the marginal cost of electricity production in 
   all hours* 
    
.. note::
  With some solvers, the operation-only price in the 
  highest-load hour might equal the price of lost load. 

.. tip:: 

  The units of the dual variables can be identified using the definition
  :math:`\lambda = \frac{\partial f}{\partial g}`. The units of the dual
  are simply the units of the objective function divided by the units of 
  the constraint. The units of both can be discovered using the results 
  functions ``r.get_unit('cost_total')`` and 
  ``r.get_unit('flow_conversion_output')``, respectively.









  



