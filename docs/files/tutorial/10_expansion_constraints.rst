.. _t_expansion.t_expansion:

################################
Technology expansion constraints
################################

.. admonition:: At a glance
   :class: note

   | **You will** limit how fast a technology can grow, reproduce an S-curve, and 
        see how the same constraint can accidentally prevent a sector from starting at all.
   | **You need** the setup from :ref:`tutorials_intro.setup`.

Left alone, a cost-optimal model may build a hundred gigawatts of a technology
in a single year if that is cheapest. Real supply chains, workforces and permit
queues do not work that way.

ZEN-garden constrains annual capacity additions as a function of **past**
capacity additions. Past additions are depreciated over time to represent
knowledge decay — skilled staff and engineering firms are lost when a
technology is not deployed continuously.

.. warning::
    This feature makes both the solution and its interpretation harder. It
    couples years to each other, and with spillover it couples nodes as well,
    which increases solve time. It can also reduce the feasible space sharply
    and produce effects you did not intend, including preventing an entire
    sector from being taken up. Use it when you need it, and check the results
    carefully. Exercise 3 below reproduces exactly this failure.


.. _t_expansion.parameters:

The five parameters
===================

Four are technology parameters, in the technology's ``attributes.json``; the
fifth is a system-wide parameter in ``energy_system/attributes.json``.

``max_diffusion_rate``
    The maximum annual capacity addition as a fraction of the existing
    knowledge stock. Because the limit is proportional to what already exists,
    the constraint is linear but produces **exponential** growth — it describes
    the exponential part of an S-curve. The default is ``inf``,
    which switches the constraint off.

``knowledge_spillover_rate``
    How much knowledge from other nodes counts towards the local stock. ``0.05``
    means 5% of other nodes' knowledge contribute to the local knowledge stock. 
    Setting it to ``inf`` assumes perfect spillover, so only overall additions are 
    constrained by the overall knowledge stock.

``market_share_unbounded``
    A small fraction of the existing capacity of **all other technologies with the
    same reference carrier** that may be added regardless of the technology's
    own knowledge stock. Values of 1–2% are realistic. Without it, a technology
    with zero existing capacity has zero knowledge and can therefore never be
    built.

``capacity_addition_unbounded``
    A fixed amount of capacity that may be added each year regardless of any
    knowledge stock. Use it only when nothing in the sector has existing
    capacity: an emerging sector such as carbon capture, where otherwise no
    technology could ever start.

``knowledge_depreciation_rate``
    How fast knowledge decays, default ``0.1``. At 10% per year, 1 GW added in
    2020 leaves :math:`0.9^{10} = 0.349` GW of knowledge by 2030.


.. _t_expansion.bootstrap:

The bootstrapping problem
=========================

The constraint limits additions in proportion to what exists. If nothing
exists, nothing may be added, and nothing ever will.

``market_share_unbounded`` and ``capacity_addition_unbounded`` are the two ways
out:

* Use ``market_share_unbounded`` when other technologies in the same sector
  already exist: a new heat technology can borrow from the installed heating
  fleet.
* Use ``capacity_addition_unbounded`` when the whole sector is new and there is
  nothing to borrow from.


Exercises
=========

The exercises are cumulative. Work on a copy of
``5_multiple_time_steps_per_year`` and extend it to ten years so that growth
has room to show:

.. code-block:: json

    "optimized_years": 10,
    "interval_between_years": 1

.. warning::
    On the unmodified dataset, the heat pump is never competitive with the gas
    boiler at all — its capacity stays at exactly zero in every one of these
    exercises regardless of ``max_diffusion_rate``, because there is nothing
    pushing the system away from gas. Add a cost pressure that makes the heat
    pump worth building at some point in the ten years, for example a rising
    carbon price in ``energy_system/price_carbon_emissions.csv`` (see
    :ref:`t_emissions.t_emissions`) ramping from 0 to a few hundred Euro/ton
    over the horizon. Everything below assumes you have done this — otherwise
    every exercise "succeeds" by producing the same all-zero result and
    nothing will look different between them.

1. **Establish the unconstrained baseline.** Run with the default
   ``max_diffusion_rate`` of ``"inf"`` and plot heat pump capacity by year.

   *Expected result: capacity jumps to its final level almost immediately,
   in the first year where it becomes economic. There is no ramp, because
   nothing prevents one.*

   *With a carbon price ramping 0 -> 360 Euro/ton over ten years: heat pump
   capacity in CH jumps straight to 23.11 GW in 2023 (the very first year) and
   stays exactly there through 2031, with a small further step to 23.59 GW in
   2032 as the price keeps rising — DE follows the same pattern at 142.21 ->
   147.56 GW. No ramp, exactly as expected.*

2. **Constrain the growth rate.** In the heat pump's ``attributes.json``, set:

   .. code-block:: json

       "max_diffusion_rate": {
         "default_value": 0.3,
         "unit": "1"
       }

   *Expected result: capacity now grows gradually year over year rather than
   jumping. The growth is roughly exponential at first and flattens as demand
   is met — the S-curve. Total cost rises, because the system must meet demand
   with something more expensive while the heat pump fleet grows.*

   *Same carbon price, with the 0.3 diffusion cap: CH heat pump capacity goes
   0 -> 0.94 -> 4.42 -> 9.42 -> 16.50 -> 21.16 -> 22.11 -> 22.53 -> 23.11 ->
   23.59 GW over the ten years — roughly quadrupling in the first two active
   years (0.94 -> 4.42), then the year-on-year ratio drops steadily towards 1
   as it approaches the unconstrained level from exercise 1. Total cost over
   the ten years rises from 9720.0 to 11855.9 (+22%).*

3. **Reproduce the trap.** Set the heat pump's ``capacity_existing`` to ``0``
   everywhere, and set ``market_share_unbounded`` to ``0`` in
   ``energy_system/attributes.json``, keeping ``max_diffusion_rate`` at 0.3.

   *Expected result: heat pump capacity stays at zero in every year. Its
   knowledge stock starts at zero, 30% of zero is zero, and there is no
   unbounded term to seed it — so the technology can never be built, at any
   price. The model does not report an error; it just silently never adopts the
   technology. This is why the warning at the top of the page exists, and why
   you should always check that a constrained technology is capable of
   starting.*

   *Confirmed even with the carbon price still active and heat demand fully
   met (no shedding): capacity is exactly zero in all ten years, at both
   nodes — the diffusion constraint dominates entirely, regardless of how
   attractive the technology would otherwise be.*

4. **Fix it two ways and compare.** First restore ``market_share_unbounded`` to
   ``0.01``; then instead set ``capacity_addition_unbounded`` to a small value
   such as ``0.5 GW``.

   *Expected result: both let the technology start, but they ramp differently.
   The market-share route grows in proportion to the rest of the heating
   fleet, so it starts slowly and accelerates. The unbounded-addition route
   adds the same fixed amount every year regardless, so it starts faster and
   does not accelerate. Pick the one that matches the story you are telling
   about the sector.*

.. seealso::
    :ref:`additional_features.technology_diffusion` documents the parameters
    and the equations behind them. If solve times rise sharply after enabling
    this, see :ref:`troubleshooting.solution_times`.
