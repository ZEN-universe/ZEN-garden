.. _t_output.t_output:

####################
Managing output size
####################

.. admonition:: At a glance
   :class: note

   | **You will** control what ZEN-garden writes to disk, and save duals and reduced costs without producing unusable result files.
   | **You need** the setup from :ref:`tutorials_intro.setup`.

On the tutorial dataset, output size is irrelevant. On a European model with
hourly resolution, thirty years and a dozen scenarios, it becomes the thing
that fills your disk and makes results slow to load.

Everything here is set in the ``solver`` section of ``config.json``.


.. _t_output.what_is_saved:

What gets saved by default
==========================

.. list-table::
   :widths: 34 16 50
   :header-rows: 1

   * - Setting
     - Default
     - Effect
   * - ``save_parameters``
     - ``true``
     - Writes the model parameters alongside the results.
   * - ``save_duals``
     - ``false``
     - Dual variables. Strongly increases file size.
   * - ``save_reduced_costs``
     - ``false``
     - Reduced costs. Also a significant increase.

Variables are always saved. Duals and reduced costs are off by default
precisely because they are large.


.. _t_output.selecting:

Saving only what you need
=========================

Four list settings narrow what is written. An **empty list means "save
everything"** of that type, which is the default:

* ``selected_saved_parameters``
* ``selected_saved_variables``
* ``selected_saved_duals``
* ``selected_saved_reduced_costs``

So this saves the duals of one constraint and nothing else:

.. code-block:: json

    {
      "solver": {
        "save_duals": true,
        "selected_saved_duals": ["constraint_nodal_energy_balance"]
      }
    }

This is usually what you want. The nodal energy balance dual is the marginal
price of a carrier at a node and time step, the reason most people enable
duals at all, and it is a small fraction of the duals of the whole model.

.. warning::
    Non-selected parameters, variables, duals and reduced costs are **not
    saved**, and cannot be recovered without re-running the model. Narrow the
    lists only when you are confident about what you need. In particular, the
    visualization platform may not work correctly if the parameters and
    variables it expects are missing.

.. note::
    The name of a dual is the name of its constraint. Use
    ``r.get_component_names('dual')`` on a small run first to find the exact
    names, then narrow the list on the big run.


.. _t_output.other:

Related settings
================

``folder_output``
    Where results are written. Defaults to ``./outputs/``. Also settable per
    run with ``zen-garden --folder_output <path>``, which is convenient on a
    cluster where results belong on scratch rather than in the home directory.

``overwrite_output``
    Whether a re-run replaces the previous results. Default ``true``.

``keep_files``
    Whether the solver's working files are kept after solving. Default
    ``false``; turning it on is a debugging aid, not something to leave on.

``clean_sub_scenarios``
    Whether sub-scenario folders are removed between runs. Default ``false``.
    Relevant when list expansion generates many sub-scenarios; see
    :ref:`t_scenario_tutorial.t_scenario_tutorial`.


Exercises
=========

The exercises are cumulative. Work on a copy of
``5_multiple_time_steps_per_year``.

1. **Measure the baseline.** Run the dataset as shipped and record the size of
   the output folder on disk.

   *Expected result: a small folder, since this model has 96 time steps and
   three years. Note the number so you can compare against it.*

2. **Turn on duals and reduced costs, and measure again.** Set both
   ``save_duals`` and ``save_reduced_costs`` to ``true`` and re-run.

   *Expected result: the output folder grows substantially. Even on a model
   this small the difference is obvious, and it scales with time steps, nodes
   and technologies, which is why it becomes a problem on a real model rather
   than this one.*

3. **Keep the dual you actually want and drop the rest.** Add:

   .. code-block:: json

       "selected_saved_duals": ["constraint_nodal_energy_balance"]

   and set ``save_reduced_costs`` back to ``false``.

   *Expected result: the folder shrinks back close to the baseline while
   ``r.get_dual('constraint_nodal_energy_balance')`` still works. Confirm that
   another dual is now absent: Pick any other name from
   ``r.get_component_names('dual')`` on the exercise 2 run.
   That absence is the trade-off: you have the marginal prices and nothing
   else.*

4. **Check what you broke.** Open the results of exercise 3 in the
   visualization platform.

   *Expected result: the platform still works, because variables and parameters
   were untouched. Now try narrowing* ``selected_saved_variables`` *to a single
   variable and re-run: the platform loses most of its plots.*

.. seealso::
    :ref:`configuration.solver` lists every solver setting.
    :ref:`t_analyze.t_analyze` and :ref:`results_api.results_api` cover reading
    what you saved.
