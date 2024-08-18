################
Configurations
################
.. _System, analysis, solver settings:
System, analysis, solver settings
========

.. _system:
System
------

.. _analysis:
Analysis
-------

.. _solver:
Solver
------

.. _Time series aggregation and representation:
Time series aggregation and representation
========
Time steps in ZEN-garden
------------------------
ZEN-garden is a temporally resolved investment and operation optimization model. That means that in general we have three different time indices:

1. ``set_base_time_steps``: is the highest resolution in the model. It is not necessarily used to index any component, but merely as a common "beat" or "rhythm" to all other time indices. We consider each hour as the base time index. Thus, each time index can be converted to the base time index, which is then a sequence of the time steps with the length of the base time index. This sequence is called ``sequence_time_steps``. The number of occurrences of each time step is called ``time_steps_duration``.
2. ``set_time_steps_yearly``: Some components have a yearly resolution. These include for example the yearly carbon emission limit (``carbon_emissions_limit``) or the yearly costs (``cost_total``). Note that these are in general not associated with any specific element (technology or carrier).
3. ``set_time_steps_operation``: The operation of built capacities is resolved on a higher resolution than the yearly time steps. For the technologies and the carriers, this is the index ``set_time_steps_operation``.

The time parameters in ZEN-garden
---------------------------------

* ``reference_year``: First year of the optimization. Used to calculate the remaining lifetime of the existing capacities and the following years of the optimization.
* ``unaggregated_time_steps_per_year``: number of base time steps per optimization year. Must be <= 8760 (total number of hours per year)
* ``aggregated_time_steps_per_year``: number of representative periods per year to aggregate the time series. Thus, all operational components are aggregated to ``aggregated_time_steps_per_year`` time steps. For further information on time series aggregation, see below.
* ``optimized_years``: number of investigated years.
* ``interval_between_years``: interval between two optimization years.
* ``use_rolling_horizon``: if True, we do not optimize all years simultaneously but optimize for a subset of years and afterward move the optimization window to the next year and optimize again. For further information on rolling horizon and myopic foresight versus perfect foresight refer to, e.g., `Poncelet et al. 2016 <10.1109/EEM.2016.7521261>`_.
* ``years_in_rolling_horizon``: number of optimization periods in the subset of the optimization horizon as mentioned above. Only relevant if ``use_rolling_horizon`` is True.
* ``interval_between_optimizations``: number of optimization periods for which the decisions of each rolling horizon are saved. Must be shorter than ``years_in_rolling_horizon``; default is 1. For an example for varying decision horizon lengths, refer to `Keppo et al. 2010 <10.1016/J.ENERGY.2010.01.019>`_. Only relevant if ``use_rolling_horizon`` is True.

Example I, no rolling horizon:

.. code-block::

    "reference_year": 2020,
    "optimized_years": 4,
    "interval_between_years": 10

The resulting investigated years are

.. code-block::

    [2020,2030,2040,2050]

Example II, rolling horizon:

.. code-block::

    "reference_year": 2020,
    "optimized_years": 4,
    "interval_between_years": 10,
    "use_rolling_horizon": True,
    "years_in_rolling_horizon": 2,
    "interval_between_optimizations": 1

The resulting sequence of investigated years are:

.. code-block::

    [2020,2030]
    [2030,2040]
    [2040,2050]
    [2050]

What is the idea of time series aggregation (TSA)?
---------------------------------------------------

Full time series with 8760 time steps per year are often too large so that the optimization takes too long or cannot be solved at all in feasible times.
Thus, we apply a time series aggregation (TSA) which reduces the number of time steps by aggregating time steps with similar input values to a single time step.
By doing so, we can represent our full time series (8760 base time steps) by representative time steps, e.g., 200.

I don't investigate hourly behavior or I want to investigate a full time series. What do I do?
-------------------------------------------------------------------------------------------

Open the ``system.json`` file and set ``"conduct_time_series_aggregation"=False``. This disables the time series aggregation. If you do not want to investigate a full year, set ``"unaggregated_time_steps_per_year"<8760``

I want to use the time series aggregation. What do I do?
-------------------------------------------------------

Open the ``system.json`` file and set ``"aggregated_time_steps_per_year"`` smaller than ``"unaggregated_time_steps_per_year"``. You are then aggregating ``"unaggregated_time_steps_per_year"`` (e.g., 8760 base time steps) to ``"aggregated_time_steps_per_year"`` (e.g., 200 representative time steps).
If you mistakingly set ``"aggregated_time_steps_per_year">"unaggregated_time_steps_per_year"``, don't worry, the TSA is disabled and it behaves as if ``"aggregated_time_steps_per_year"="unaggregated_time_steps_per_year"``.

For an in-depth introduction to TSA, refer to `Hoffmann et al. 2020 <https://www.mdpi.com/1996-1073/13/3/641>`_. The authors at FZ Jülich are also the developers of the TSA package `tsam <https://tsam.readthedocs.io/en/latest/>`_ that we are using in ZEN-garden.

How are short-term and long-term storages modeled?
--------------------------------------------------

The modeling of storage technologies with TSA is challenging because storages couple time steps (see :ref:`Storage Technologies`).
Hence, the sequence of time steps is important for the operation of the storage level.
There are different approaches to model storages with TSA, with the approaches by `Gabrielli et al. 2018 <https://www.sciencedirect.com/science/article/pii/S0306261917310139>`_ and `Kotzur et al. <https://www.sciencedirect.com/science/article/pii/S0306261918300242>`_ being the most common.
In ZEN-garden, we extend the approach by Gabrielli et al. 2018 to model storages with TSA. The approach is detailed in `Mannhardt et al. 2023 <https://www.sciencedirect.com/science/article/pii/S2589004223008271>`_.
In short, every time that the sequence of operational time steps changes, the another storage time step is added. This increases the number of variables, but explicitly enables short- and long-term storages.
In particular, this storage level representation leads to fewer time steps than the full time series without loss of information.

Great, the TSA works. But I want more information!
----------------------------------------------------

1. In the ``default_config.py``, you find the class ``TimeSeriesAggregation`` where you can set the ``clusterMethod``, ``solver``, ``extremePeriodMethod`` and ``representationMethod``. Most importantly, the ``clusterMethod`` selects which algorithm is used to determine the clusters of representative time steps. Probably, the most common ones are `k_means <https://en.wikipedia.org/wiki/K-means_clustering>`_ and `k_medoids <https://en.wikipedia.org/wiki/K-medoids>`_. While it is probably not necessary at this point to understand the difference of k-means and k-medoids in detail, it is important to know that k-means averages the input data over the representative time steps, which reduces the extreme period behavior, thus, peaks are smoothened.
2. As said before, each aggregated time step represents multiple base time steps. Thus, the behavior in each aggregated time step accounts for more than one time step. Thus, the operational costs and operational carbon emissions of each aggregated time step are multiplied with the ``time_steps_operation_duration`` of the respective time step.
3. What is this strange ``sequence_time_steps`` floating around everywhere in the code? The substitution of the base time steps by the aggregated time steps yields a sequence of time steps, which is ``len(set_base_time_steps)`` entries long and encapsulates the order in which the aggregated time steps appear in the representation of the base time steps. We use the sequence of time steps to convert one time step into another. For example we can use the order to get the yearly time step associated with a certain operational time step, or the year of a certain operational time step.
