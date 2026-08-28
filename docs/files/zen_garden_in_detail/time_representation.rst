.. _time_representation.time_representation:

###################
Time representation
###################

ZEN-garden optimizes both the investment in and the operation of an energy
system, so it carries several time indices at once. This page explains how
those indices relate to each other, which settings control them, and what time
series aggregation does to them.

To switch aggregation on and see its effect, see the tutorial on
:ref:`time series aggregation <t_tsa.t_tsa>`. To limit how far the optimization
looks ahead, see the tutorial on :ref:`myopic foresight <t_foresight.t_foresight>`.


.. _time_representation.time_steps:

Time steps in ZEN-garden
========================

There are three different time indices:

1. ``set_hours`` is the highest resolution in the model. Each hour is a base
   time index. When time series aggregation is used, the base time steps are
   aggregated to representative time steps, where each representative time step
   stands for multiple base time steps. The order in which the representative
   time steps appear is the ``sequence_time_steps``, and the number of
   occurrences of each representative time step is the ``time_steps_duration``.

2. ``set_years`` is the yearly resolution. Some components are resolved per
   year, for example the yearly carbon emission limit
   (``carbon_emissions_annual_limit``) or the yearly costs (``cost_total``).
   These are generally not associated with a specific element.

3. ``set_time_steps_operation`` resolves the operation of built capacities at a
   higher resolution than years. For technologies and carriers, this is the
   operational index.


.. _t_tsa.time_parameters:
.. _time_representation.time_parameters:

The time parameters
===================

The parameters below are set in ``system.json``
(see :ref:`configuration.system`).

* ``reference_year``: first year of the optimization. Used to calculate the
  remaining lifetime of existing capacities and the following years of the
  optimization.
* ``unaggregated_time_steps_per_year``: number of base time steps per
  optimization year. Must be <= 8760 (total number of hours per year).
* ``aggregated_time_steps_per_year``: number of representative time steps per
  year used when time series aggregation is active. All operational components
  are aggregated to this many time steps.
* ``optimized_years``: number of investigated years.
* ``interval_between_years``: interval between two optimization years.
* ``use_rolling_horizon``: if ``true``, the years are not all optimized
  simultaneously. Instead a subset of years is optimized, the optimization
  window moves forward, and the optimization is repeated. See
  :ref:`t_foresight.t_foresight`, and for background, e.g.
  `Poncelet et al. 2016 <https://www.sciencedirect.com/science/article/abs/pii/S0306261915013276>`_.
* ``years_in_rolling_horizon``: number of optimization periods in the foresight
  horizon. Only relevant if ``use_rolling_horizon`` is ``true``.
* ``years_in_decision_horizon``: number of optimization periods for which the
  decisions of each rolling horizon are kept. Must not be longer than
  ``years_in_rolling_horizon``; the default is 1. For an example of varying
  decision horizon lengths, see e.g. `Keppo et al. 2010
  <https://www.sciencedirect.com/science/article/abs/pii/S0360544210000216>`_.
  Only relevant if ``use_rolling_horizon`` is ``true``.

Example I, no rolling horizon:

.. code-block:: json

    "reference_year": 2020,
    "optimized_years": 4,
    "interval_between_years": 10

The resulting investigated years are:

.. code-block:: text

    [2020, 2030, 2040, 2050]

Example II, rolling horizon:

.. code-block:: json

    "reference_year": 2020,
    "optimized_years": 4,
    "interval_between_years": 10,
    "use_rolling_horizon": true,
    "years_in_rolling_horizon": 2,
    "years_in_decision_horizon": 1

The resulting sequence of investigated years is:

.. code-block:: text

    [2020, 2030]
    [2030, 2040]
    [2040, 2050]
    [2050]


.. _time_representation.tsa_idea:

What time series aggregation does
=================================

Full time series with 8760 time steps per year are often so large that the
optimization takes too long or cannot be solved at all in reasonable time. Time
series aggregation (TSA) reduces the number of time steps by clustering time
steps with similar input values into a single representative time step. The
full time series of 8760 base time steps is then represented by a smaller
number of representative time steps, for example 200.

Because each aggregated time step stands for multiple base time steps, the
operational costs and operational carbon emissions of each aggregated time step
are multiplied by the ``time_steps_operation_duration`` of that time step.

The clustering algorithm itself is configured in the ``analysis`` section of
``config.json``; see the timeseries aggregation settings in
:ref:`configuration.analysis`. Most importantly, ``clusterMethod`` selects the
clustering algorithm. Note that ``kmeans`` averages the input data over the
representative time steps, which smooths peaks and reduces extreme-period
behaviour, whereas ``kmedoids`` and the default ``hierarchical`` pick
representative periods from the data.

For an in-depth introduction to TSA, see `Hoffmann et al. 2020
<https://www.mdpi.com/1996-1073/13/3/641>`_. The authors at FZ Jülich are also
the developers of the TSA package `tsam
<https://tsam.readthedocs.io/en/latest/>`_ that ZEN-garden uses.


.. _time_representation.storage_tsa:

Short- and long-term storage under aggregation
==============================================

Modeling storage technologies with TSA is difficult because storages couple
time steps (see :ref:`input_structure.storage_technologies`). The sequence of
time steps therefore matters for the storage level. The two most common
approaches are `Gabrielli et al. 2018
<https://www.sciencedirect.com/science/article/pii/S0306261917310139>`_ and
`Kotzur et al. <https://www.sciencedirect.com/science/article/pii/S0306261918300242>`_.

ZEN-garden extends the approach by Gabrielli et al. 2018, as detailed in
`Mannhardt et al. 2023
<https://www.sciencedirect.com/science/article/pii/S2589004223008271>`_. In
short, every time the sequence of operational time steps changes, another
storage time step is added. This increases the number of variables but
explicitly enables both short- and long-term storage. In particular, this
storage level representation needs fewer time steps than the full time series
without losing information.
