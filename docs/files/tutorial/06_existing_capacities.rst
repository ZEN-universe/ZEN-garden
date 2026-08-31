.. _t_brownfield.t_brownfield:

###################
Existing capacities
###################

.. admonition:: At a glance
   :class: note

   | **You will** start the optimization from an existing technology fleet instead of an 
       empty system, and account for capacity that is already committed.
   | **You need** the setup from :ref:`tutorials_intro.setup`.

The tutorials so far have assumed **greenfield** expansion: nothing exists and
the model builds the entire system from scratch. Real studies almost always
start from an existing fleet — this is **brownfield** expansion.

ZEN-garden supports two kinds of pre-existing capacity:

1. Capacity **built in the past**, which is available immediately and has a
   reduced remaining lifetime.
2. Capacity that will be built **within the optimization horizon**, for
   example a plant whose investment decision has already been made but whose
   construction has not started.


.. _t_brownfield.capacity_existing:

Declaring existing capacity
===========================

Existing capacity is given per technology in a ``capacity_existing.csv`` file
in that technology's folder. It is indexed by node **and construction year**:

.. code-block:: text

    node,year_construction,capacity_existing
    CH,2010,0.037
    CH,2015,0.092
    CH,2018,0.000
    DE,2010,14.76
    DE,2019,3.043

The construction year matters because it determines how much lifetime is left.
A technology with a ``lifetime`` of 25 years, built in 2010, retires in 2035, and
the model will have to replace it if demand persists.

For transport technologies, the first index is the edge instead of the node, and for
storage technologies the energy-rated capacity is declared separately in
``capacity_existing_energy.csv`` (see :ref:`t_storage.t_storage`).


Future committed capacity
-------------------------

A construction year **inside** the optimization horizon describes capacity that
is already committed but not yet built:

.. code-block:: text

    node,year_construction,capacity_existing
    DE,2030,2.5

The 2.5 GW appears in 2030 whether or not the optimizer would have chosen it,
and is unavailable before then.

.. tip::
    Use this for projects under construction or contractually fixed. Modelling
    them as free investment decisions lets the optimizer "cancel" commitments
    that in reality cannot be cancelled.

The related parameter ``capacity_investment_existing`` records capacity whose
investment decision has been made but whose construction has not started. The time
lag between investment and construction is the construction time.


.. _t_brownfield.switch:

Switching brownfield off
========================

``use_capacities_existing`` in ``system.json`` controls whether existing
capacities are used at all:

.. code-block:: json

    "use_capacities_existing": false

The default is ``true``. Setting it to ``false`` ignores every
``capacity_existing`` entry and runs a greenfield optimization on the same
input data — which is the cleanest way to ask "how much of this system is
locked in by what we already have?"

.. seealso::
    ``allow_investment`` is the complementary setting: it forbids *new*
    capacity, so only the existing fleet may be operated. See
    :ref:`configuration.system`.


Exercises
=========

The exercises are cumulative and use the dedicated example dataset.

Download it alongside your tutorial dataset:

.. code-block:: shell

    zen-example --dataset="10_brown_field"
    zen-garden --dataset="10_brown_field"

This dataset extends the yearly-variation example with an existing
photovoltaics fleet in ``CH`` and ``DE``, given per construction year from the
late 1990s onwards.

1. **How much photovoltaic capacity exists before the model builds anything?**
   Read ``capacity_existing`` for photovoltaics and sum it per node.

   .. code:: python

       from zen_garden import Results
       r = Results(path='<data>/outputs/10_brown_field')
       print(r.get_total('capacity_existing', index="photovoltaics"))

   *Expected result: a non-zero existing fleet in both nodes, considerably
   larger in DE than in CH — and one DE vintage with a construction year
   inside the optimization horizon, which is the committed-capacity case above.*

   *On this dataset: 0.607 GW existing in CH versus 60.223 GW in DE (about
   100x larger), and a 2.5 GW DE vintage with* ``year_construction,2024`` *—
   inside the 2023-2025 horizon.*

2. **Compare capacity additions against a greenfield run.** Copy the dataset,
   set ``"use_capacities_existing": false`` in ``system.json``, and run it.
   Then compare the two:

   .. code:: python

       from zen_garden import Results, compare_model_values
       r_brown = Results(path='<data>/outputs/10_brown_field')
       r_green = Results(path='<data>/outputs/10_brown_field_greenfield')
       compare_model_values([r_brown, r_green], component_type='variable')

   *Expected result: the greenfield run adds more photovoltaic capacity in the
   first year, because it cannot draw on the existing fleet. Total capacity in
   the first year should be similar between the two runs, the system needs
   roughly the same amount of PV either way; what differs is how much of it has
   to be paid for inside the horizon.*

   *On this dataset (2023 photovoltaic capacity): total capacity is close in
   both runs (CH identical at 9.54 GW; DE 85.58 GW brownfield vs. 86.17 GW
   greenfield, +0.7%). Capacity* **additions**\ *, however, differ sharply — CH
   adds 8.93 GW brownfield vs. 9.54 GW greenfield (the full amount, since
   nothing exists to draw on); DE adds 27.88 GW vs. 86.17 GW. Total cost is
   close too: 2023 is marginally cheaper greenfield (65,606 vs. 65,629 MEuro,
   -0.03%), and 2024-2025 are identical between the two runs down to the last
   digit: once the horizon moves past the existing fleet's influence, the two
   scenarios converge.*

3. **Make the fleet retire and watch the model replace it.** In the copied
   dataset, reduce the ``lifetime`` of photovoltaics in ``attributes.json`` to
   a value short enough that the older installations retire during the horizon —
   for example 20 years.

   *Expected result: capacity additions appear in the years when the large
   early-2010s DE installations reach end of life. The pattern of additions follows
   the construction years in* ``capacity_existing.csv`` *shifted by the
   lifetime, which is the clearest demonstration of why the file is indexed by
   the construction year.*

.. seealso::
    :ref:`dataset_examples.dataset_examples` describes ``10_brown_field`` in
    context, and :ref:`additional_features.construction_times` covers the
    related delay between an investment decision and capacity becoming
    available.
