.. _t_tsa.t_tsa:

#######################
Time series aggregation
#######################

.. admonition:: At a glance
   :class: note

   | **You will** reduce the number of time steps in a model, measure what that costs you in accuracy, and exclude a time series from the clustering.
   | **You need** the setup from :ref:`tutorials_intro.setup`.

A full year at hourly resolution is 8760 time steps. Multiplied by years,
nodes, technologies and carriers, that is often too large to solve. Time series
aggregation (TSA) clusters similar time steps into a smaller number of
representative ones.

This tutorial is about using it. For what aggregation does to the model's time
indices, and how storage is handled when the sequence of hours is no longer the
calendar, see :ref:`time_representation.time_representation`.


.. _t_tsa.using_the_tsa:

Switching aggregation on
========================

Two settings in ``system.json`` control it:

.. code-block:: json

    "conduct_time_series_aggregation": true,
    "aggregated_time_steps_per_year": 10

``unaggregated_time_steps_per_year`` is the number of base time steps, i.e., the
resolution of your input data. ``aggregated_time_steps_per_year`` is how many
representative time steps you want. Aggregation only happens when the second is
smaller than the first.

.. note::
    If you set ``aggregated_time_steps_per_year`` larger than
    ``unaggregated_time_steps_per_year`` by mistake, nothing breaks:
    aggregation is disabled and the model behaves as if the two were equal.

To switch aggregation off entirely, set
``"conduct_time_series_aggregation": false``. If you want a shorter period
rather than a coarser one, reduce ``unaggregated_time_steps_per_year`` instead
— that models the first *N* hours of the year at full resolution, which is a
different thing from clustering the whole year.

The clustering algorithm itself, e.g., with the settings ``clusterMethod``, 
``representationMethod``, ``extremePeriodMethod``, is configured in the ``analysis`` 
section of ``config.json``. The available options and defaults are listed under the
timeseries aggregation settings in :ref:`configuration.analysis`.

.. note::
    The default ``clusterMethod`` is ``hierarchical``. Be aware that by default
    we use mean representation, which averages the input data within each cluster.
    An important property is that it conserves the total value of the time series, 
    but it does not conserve the peak value. This is important if your original
    model is sized by extreme events, that choice will systematically
    under-build.


.. _t_tsa.exclude:

Excluding a time series from clustering
=======================================

Not every time series should influence which hours get grouped together. A
helper series used to shape a technology's availability, for example, carries
no information about what the system has to cope with, and letting it drive the
clustering wastes representative time steps.

Create ``energy_system/exclude_parameter_from_TSA.yaml``, keyed by element
name:

.. code-block:: yaml

    natural_gas:
      - availability_import

To exclude a parameter for every element of a class, use the class name:

.. code-block:: yaml

    set_technologies:
      - max_load

To exclude every parameter of one element, set the value to ``null``:

.. code-block:: yaml

    natural_gas_boiler: null

All three can be combined in the same file, and an element can list several
parameters:

.. code-block:: yaml

    natural_gas:
      - availability_import
      - price_import
    set_technologies:
      - max_load
    natural_gas_boiler: null

Excluded parameters are still used by the model; they simply do not take part
in deciding the clusters.


Exercises
=========

The exercises are cumulative. Work on a copy of
``5_multiple_time_steps_per_year``, which has 96 base time steps and
aggregation switched off.

1. **Establish the exact answer.** Run the dataset as shipped, with
   ``"conduct_time_series_aggregation": false`` and 96 time steps. Record total
   cost and the installed capacity of each technology.

2. **Aggregate to 10 representative time steps and measure the error.** Set
   ``"conduct_time_series_aggregation": true`` and
   ``"aggregated_time_steps_per_year": 5``, then compare against exercise 1.

   .. code:: python

       from zen_garden import Results, compare_model_values
       r_full = Results(path='<data>/outputs/<full_resolution_run>')
       r_agg  = Results(path='<data>/outputs/<aggregated_run>')
       cv = compare_model_values([r_full, r_agg], component_type='variable')
       capacity_comp = cv["capacity"].loc[:,(slice(None),2023)].round(4)

   *Expected result: The PV capacity is reduced by 4%, and gas boiler capacity is 
   reduced by 12%. The total annual cost is reduced from 833 MEuro to 783 MEuro,
   a 6% drop.*

3. **Sweep the number of representative time steps.** Repeat exercise 2 with 5,
   10, 20 and 50 representative time steps.

   *Expected result: the error shrinks as the number of time steps grows, and
   converges towards the full-resolution answer at 96. The interesting part is
   the shape: error typically falls steeply at first and then flattens, so
   there is a point beyond which extra time steps buy little accuracy. Finding
   that point on a small version of your model is how you choose the setting
   for the full one.*
