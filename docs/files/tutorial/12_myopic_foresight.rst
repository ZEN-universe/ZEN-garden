.. _t_foresight.t_foresight:

################
Myopic foresight
################

.. admonition:: At a glance
   :class: note

   | **You will** limit how far ahead the optimization can see, and compare the resulting pathway against perfect foresight.
   | **You need** the setup from :ref:`tutorials_intro.setup`.

By default ZEN-garden optimizes every year at once. This is **perfect
foresight**: the 2023 investment decision is made in full knowledge of the 2050
carbon budget, gas price and demand.

That is a strong assumption, and often the wrong one. Real investors do not
know future policy. **Myopic foresight** optimizes a limited window of years,
keeps the decisions from the start of that window, then rolls forward and
optimizes again, each time seeing only a few years ahead.

.. _t_foresight.horizons:

The two horizons
================

Myopic foresight is configured by three settings in ``system.yaml``:

.. code-block:: yaml

    use_rolling_horizon: true
    years_in_rolling_horizon: 2
    years_in_decision_horizon: 1

``use_rolling_horizon``
    Switches the rolling horizon on. Default ``false``, i.e., perfect foresight.

``years_in_rolling_horizon``
    The **foresight horizon**: how many years the optimization can see at once.
    Setting it to 1 means the model optimizes each year in complete isolation.

``years_in_decision_horizon``
    The **decision horizon**: how many years of decisions are kept from each
    window before rolling forward. Default 1. It must not exceed the foresight
    horizon.

With ``reference_year`` 2020, four optimized years, a ten-year interval, a
foresight horizon of 2 and a decision horizon of 1, the sequence of
optimizations is:

.. code-block:: text

    [2020, 2030]
    [2030, 2040]
    [2040, 2050]
    [2050]

Each run keeps only its first year's decisions and passes the resulting
capacities forward as existing capacity for the next.

.. figure:: ../figures/zen_garden_in_detail/rolling_horizon.png
    :align: center
    :figwidth: 550 pt

    Decision and foresight horizons. Under perfect foresight the two coincide
    and span the whole horizon; under myopic foresight the foresight horizon is
    a moving window and only the decision horizon is kept.

.. note::
    A longer decision horizon means the model commits to more years at a time
    and rolls forward in bigger steps. It does not give the model more
    information; that is what the foresight horizon does.


.. _t_foresight.consequences:

What changes, and why
=====================

Myopic foresight systematically under-invests in anything whose payoff falls
outside the foresight window:

* **Long-lived, capital-intensive assets**: look worse, because the model cannot
  see the years over which they pay back.
* **Future policy**: a carbon budget tightening in 2040 is invisible until
  the window reaches it, so the model builds emitting capacity that it later
  has to strand.
* **Lock-in becomes visible.**: perfect foresight cannot
  produce a stranded asset, because it never makes a decision it will regret.

.. warning::
    A cumulative carbon budget and a short foresight horizon interact strongly. 
    If the model cannot see the budget, it will spend it early and then find the
    later years challenging or infeasible. Consider this critical behavior when
    configuring the model. See
    :ref:`t_emissions.t_emissions`.

.. tip::
    The rolling horizon chops the optimization into smaller pieces, which can reduce 
    runtime. But it also changes the decision-making paradigm, so be sure how you 
    configure the model. Combining a short foresight horizon with a cumulative
    carbon budget strongly changes the model's behavior. If this is not what you want, 
    use a longer foresight horizon or a different carbon policy.

Exercises
=========

The exercises are cumulative. Use the dedicated example dataset, which is
already configured for a rolling horizon over ten years:

.. code-block:: shell

    zen-example --dataset="8_myopic_foresight"
    zen-garden --dataset="8_myopic_foresight"

Its ``system.yaml`` sets ``use_rolling_horizon: true`` with
``years_in_rolling_horizon: 1`` — single-step foresight, the most extreme
case.

1. **Build the perfect-foresight counterpart.** Copy the dataset, set
   ``use_rolling_horizon: false``, and run it.

2. **Compare the two capacity pathways.**

   .. code:: python

       from zen_garden import Results
       r_myopic  = Results(path='<data>/outputs/8_myopic_foresight')
       r_perfect = Results(path='<data>/outputs/8_myopic_foresight_perfect')
       cv = compare_model_values([r_myopic, r_perfect], component_type="variable")

   *Expected result: In 2023, the myopic run does not invest in heat pumps at all,
   whereas the perfect-foresight run builds 112 GW of heat pumps. Instead the myopic
   run builds 193 GW of natural gas boilers (82 GW in the perfect-foresight run)*

3. **Widen the foresight horizon and watch the gap close.** Set
   ``years_in_rolling_horizon`` to 5, keeping
   ``years_in_decision_horizon`` at 1.

   *Expected result: Under a 5-year foresight horizon, the gap between the myopic and 
   perfect-foresight results narrows, showing a more informed investment strategy.
   In 2023, the myopic run builds 106 GW of heat pumps (112 GW in the perfect-foresight 
   run), and reduces its natural gas boiler investments.*

