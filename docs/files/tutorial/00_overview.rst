.. _tutorials_intro.intro:

#########################
Overview and setup
#########################


.. _tutorials_intro.overview:

Overview
========

The ZEN-garden tutorials are hands-on: each one asks you to change something,
run the model, and check the answer. They are grouped so that each group builds
on the ones before it, but you do not have to work through them in order.

**First steps** — get results out of a model that already runs.

1. :ref:`Analyze and compare results <t_analyze.t_analyze>`
2. :ref:`Change configurations <t_configuration.t_configuration>`

**Build your own model** — create and extend input data.

3. :ref:`Build a dataset from scratch <t_build.t_build>`
4. :ref:`Add technologies and carriers <t_add_techs_carriers.t_add_techs_carriers>`
5. :ref:`Varying input data <t_yearly.t_yearly>`

**Represent your system** — features that change what the model can do. These
five are independent of each other; take the ones you need.

6. :ref:`Existing capacities <t_brownfield.t_brownfield>`
7. :ref:`Carbon emission targets, budgets and prices <t_emissions.t_emissions>`
8. :ref:`Storage <t_storage.t_storage>`
9. :ref:`Retrofitting and fuel switching <t_retrofit.t_retrofit>`
10. :ref:`Technology expansion constraints <t_expansion.t_expansion>`

**Model size and foresight** — make a large model tractable.

11. :ref:`Time series aggregation <t_tsa.t_tsa>`
12. :ref:`Myopic foresight <t_foresight.t_foresight>`

**Run studies** — go from one model run to many.

13. :ref:`Scenario analysis <t_scenario_tutorial.t_scenario_tutorial>`
14. :ref:`Managing output size <t_output.t_output>`

**When things go wrong**

15. :ref:`Infeasibilities <t_infeasibilities.t_infeasibilities>`

New users should start with
:ref:`Analyze and compare results <t_analyze.t_analyze>`, which shows how to
read the outputs that every other tutorial asks you to inspect.

.. note::
    Each tutorial is independent of the others. **Within** a tutorial, however,
    the exercises are cumulative: exercise 2 usually builds on the dataset you
    changed in exercise 1. Work through the exercises of a single tutorial in
    order, and start from a fresh copy of the dataset when you move to a new
    tutorial.

.. seealso::
    Tutorials show you how to *do* things. For what a setting means or which
    files a parameter lives in, see :ref:`configuration.configuration`,
    :ref:`input_structure.input_structure` and
    :ref:`notation.notation`. If something breaks, see
    :ref:`troubleshooting.troubleshooting`.


.. _tutorials_intro.setup:

Setup
=====

Unless a tutorial says otherwise, every tutorial starts from the example
dataset ``4_multiple_time_steps_per_year``. To prepare it:

1. Install ZEN-garden by following the :ref:`installation guide
   <installation.installation>`.

2. Create a folder to hold your data, and open a terminal in it. This folder is
   referred to as ``<data>`` throughout the tutorials.

3. Activate the ZEN-garden environment (see :ref:`instructions
   <installation.activate>`), then download the example dataset:

   .. code-block:: shell

       zen-example --dataset="4_multiple_time_steps_per_year"

   The full list of example datasets is given in
   :ref:`dataset_examples.dataset_examples`.

4. Run the model:

   .. code-block:: shell

       zen-garden --dataset="4_multiple_time_steps_per_year"

ZEN-garden prints its progress to the terminal. On success, the last line is:

.. code-block:: text

   --- Optimization finished ---

A new directory ``outputs`` is created in ``<data>``, containing the results.

.. tip::
    Results are self-contained. You can copy the
    ``outputs/<dataset_name>`` folder somewhere else and still read it.


.. _tutorials_intro.dataset:

The tutorial dataset
====================

``4_multiple_time_steps_per_year`` optimizes electricity and heat supply for a
two-node system. The two nodes are Germany (``DE``) and Switzerland (``CH``).
Electricity is supplied by photovoltaics, heat by a natural gas boiler and a
heat pump. Natural gas can be imported freely at each node, stored, and
transported between nodes by pipeline. The model covers three years (2023,
2024 and 2025) and 96 hours (4 days) per year.

Because it is small, it solves in seconds — which is what makes it useful for
tutorials, and why some of its results are not physically realistic.

A more detailed description of this and all other example datasets is given in
:ref:`dataset_examples.dataset_examples`.
