.. _t_emissions.t_emissions:

###########################################
Carbon emission targets, budgets and prices
###########################################

.. admonition:: At a glance
   :class: note

   | **You will** constrain emissions three different ways and see how each one shapes 
       the transition pathway.
   | **You need** the setup from :ref:`tutorials_intro.setup`.

Decarbonization usually drives the energy transition, and ZEN-garden offers three
instruments for it. They can be used alone or together:

.. list-table::
   :widths: 26 40 34
   :header-rows: 1

   * - Instrument
     - Parameter
     - What it does
   * - Annual limit
     - ``carbon_emissions_annual_limit``
     - Caps annual emissions in **each** year separately.
   * - Cumulative budget
     - ``carbon_emissions_budget``
     - Caps cumulative emissions over the **whole** horizon.
   * - Carbon price
     - ``price_carbon_emissions``
     - Penalizes every tonne in the objective.

All three are parameters of the energy system, so they live in
``energy_system/attributes.yaml`` and, if they vary by year, in a matching
``.csv`` in the ``energy_system`` folder.

.. note::
    The dataset we are using in this example, ``6_time_series_aggregation``,
    already has a carbon budget of about
    4.27 gigatons and a budget overshoot price of 400 Euro/tons. Look at
    ``energy_system/attributes.yaml`` before you start, so you know what you
    are changing.


.. _t_emissions.annual:

Annual limits
=============

``carbon_emissions_annual_limit`` is indexed by year. Set the default in
``attributes.yaml`` and the yearly path in
``energy_system/carbon_emissions_annual_limit.csv``:

.. code-block:: text

    year,carbon_emissions_annual_limit
    2023,0.6
    2024,0.4
    2025,0.2

Annual limits force a trajectory: the model must comply in every single year
and cannot borrow headroom from a later one.

.. warning::
    Missing years are interpolated linearly by default. If your target steps
    down rather than ramping, switch interpolation off for this parameter —
    see :ref:`t_yearly.interpolation`.


.. _t_emissions.budget:

Cumulative budget
=================

``carbon_emissions_budget`` applies to the entire planning horizon, so a single
value in ``energy_system/attributes.yaml`` is enough:

.. code-block:: yaml

    carbon_emissions_budget:
      default_value: 2.0
      unit: gigatons

A budget lets the optimizer choose *when* to abate. Given a discount rate and the 
existing capacity, it will generally emit more early and abate later.
cheaper in present-value terms.


.. _t_emissions.overshoot:

Relaxing the constraints
========================

Both limits can be made soft by setting an overshoot price below infinity:

* ``price_carbon_emissions_annual_overshoot`` for the annual limits
* ``price_carbon_emissions_budget_overshoot`` for the budget

With a finite overshoot price the limit becomes a penalty rather than a hard
constraint, and the model may exceed it if compliance costs more than the
penalty.

.. tip::
    A finite overshoot price is a good diagnostic. If a model with hard limits
    is infeasible, set an overshoot price, re-run, and read how much the model
    overshoots and when — that tells you which years are actually binding. See
    :ref:`t_infeasibilities.t_infeasibilities`.


.. _t_emissions.price:

Carbon price
============

``price_carbon_emissions`` is indexed by year and penalizes all emissions in
the objective:

.. code-block:: text

    year,price_carbon_emissions
    2023,50
    2024,100
    2025,150

A price changes the relative economics of technologies but guarantees no
particular emission level. It is the right instrument for asking "what would
this price achieve?" and the wrong one for asking "how do we hit this target?"


Exercises
=========

The exercises are cumulative. Work on a copy of
``7_yearly_variation``.

1. **Start from an unconstrained baseline.** Set
   ``carbon_emissions_budget`` and ``price_carbon_emissions_budget_overshoot`` to
   ``inf`` in ``energy_system/attributes.yaml``, run the model, and record total
   emissions and total cost.

   .. code:: python

       from zen_garden import Results
       r = Results(path='<data>/outputs/<your_dataset>')
       print(r.get_total('carbon_emissions_annual'))
       print(r.get_total('cost_total'))

   *Expected result: the cheapest system, which meets heat demand with the gas
   boiler. This is your reference point — every instrument below should raise
   cost and lower emissions relative to it.*

   *On* ``7_yearly_variation``\ *: Around 33 Mton/year (334 Mton over
   the three years) and yearly costs at around 72000 MEuro/year.*

2. **Impose a cumulative budget of roughly half the baseline emissions.** Set a carbon
   budget (``carbon_emissions_budget``) of 100 Mtons. Make sure that the unit
   is in megatons, not gigatons.

   *Expected result: emissions fall to the budget and cost rises.*

   *With a budget of 100 Mt: emissions hit the budget
   exactly and annual emissions drop to 10 Mton/year. Costs rise to around 75000 
   MEuro/year (+4%).*

3. **Replace the budget with annual limits that sum to the same total.** Set
   ``carbon_emissions_budget`` back to ``inf`` and add a
   ``carbon_emissions_annual_limit.csv``
   (**Make sure that the unit in attributes.yaml is in megatons**) whose values sum to the budget
   from exercise 2:

   .. code-block:: text

    year,carbon_emissions_annual_limit
    2023,25
    2024,20
    2025,15
    2026,10
    2027,10
    2028,10
    2029,5
    2030,3
    2031,2
    2032,0

   *Expected result: a higher total cost than the budget. On this dataset, 
   actual cumulative emissions collapse to 70 Mton total. In the first years, the
   limit is not binding anymore (the maximum emissions are 10 Mton/year).
   The cost increases to 92370 MEuro/year in 2032.*

4. **Try a carbon price instead of a limit.** Remove the annual limits and set
   ``price_carbon_emissions`` to a value per year, 
   
   .. code-block:: text

    year,price_carbon_emissions
    2023,0
    2032,150

    The price ramps linearly from 0 to 150 Euro/ton over the ten years. 

   *Expected result: emissions fall somewhere between the baseline and the
   capped runs, but you cannot control where.*

   *The cumulative emissions in 2032 are around 210 Mton, so halfway between the 
   baseline and the budget. Annual costs increase from 70000 MEuro/year to 
   around 76000 MEuro/year, since now the carbon price is applied. The cost paid for
   carbon emissions ("cost_carbon_emissions_total") increases from 0 MEuro/year to 
   around 3000 MEuro/year.*

.. seealso::
    :ref:`additional_features.modeling_carbon_emissions` documents the
    parameters, and :ref:`notation.notation` lists their units and dimensions.
    Emission accounting itself — carrier and technology carbon intensities — is
    described in :ref:`features.emissions`.
