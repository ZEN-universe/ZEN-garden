.. _t_yearly.t_yearly:

##################
Varying input data
##################

.. admonition:: At a glance
   :class: note

   | **You will** shape input data to vary only along the dimensions you need, scale a 
         time series across years without rewriting it, and control how
         ZEN-garden fills in years you did not specify.
   | **You need** the setup from :ref:`tutorials_intro.setup`.

Large models need input data for every node, technology and hour. Writing out
every value by hand is neither practical nor readable, so ZEN-garden offers
several shortcuts:

* Overwrite files only need to vary along **the dimensions where the data
  actually changes**; dimensions that stay constant can be dropped entirely.
* **Yearly variation** keeps the shape of an hourly time series and scales it
  by a factor per year.
* **Interpolation** fills in years you did not specify, by default linearly.

All three are worth understanding early, because they change your input data
silently and by design.


.. _t_yearly.dimensions:

Matching the shape of your data
================================

An overwrite ``.csv`` file does not need to repeat every dimension of the
parameter it overwrites. Take a ``demand`` that varies by node and by hour:

.. code-block:: text

    node,time,demand
    CH,0,5
    CH,1,5
    DE,0,2
    DE,1,2

If the value does not change along one of the dimensions, drop that dimension.
A demand that is constant over time but differs by node only needs:

.. code-block:: text

    node,demand
    CH,5
    DE,2

You can also unstack a dimension into columns instead of rows, which is often
shorter when there are many time steps:

.. code-block:: text

    node,0,1
    CH,5,5
    DE,2,2

or 

.. code-block:: text

    time,CH,DE
    0,5,2
    1,5,2

All three files above are valid; use whichever shape is shortest for the data
you have. Dropping a constant dimension is not just less typing, it also
tells the next person who opens the file that the value truly does not vary
along it.

.. seealso::
    :ref:`input_structure.overwrite_defaults` covers every shape in full,
    including how ZEN-garden decides which column refers to which dimension.


.. _t_yearly.variation:

Yearly variation
================

A file named ``<parameter_name>_yearly_variation.csv`` multiplies the
hourly-resolved parameter by a factor for each year. The hourly shape stays the
same; only the scale changes.

Put it in the same folder as the parameter it scales. To grow electricity
demand over the horizon, create
``set_carriers/electricity/demand_yearly_variation.csv``:

.. code-block:: text

    year,demand_yearly_variation
    2023,1
    2024,1.1
    2025,1.2

Every hourly demand value in 2024 is now multiplied by 1.1.

The factor can differ per node. Specify the other dimensions as rows and the
years as columns:

.. code-block:: text

    node,2023,2024,2025
    CH,1,1.05,1.10
    DE,1,1.10,1.20

Missing values default to ``1``, meaning "no scaling".

.. warning::
    Yearly variation only works for parameters that are resolved **hourly**. It
    does nothing for parameters with a single value per year, such as
    ``capacity_limit``. For those, put the yearly values directly in
    ``<parameter_name>.csv``.

.. note::
    When you overwrite a yearly variation with the scenario tool, target
    ``demand_yearly_variation``, not ``demand``. See
    :ref:`t_scenario.t_scenario`.


.. _t_yearly.interpolation:

Interpolation
=============

You rarely have a value for every year. By default ZEN-garden interpolates
linearly between the years you do give, so this:

.. code-block:: text

    year,demand_yearly_variation
    2023,1
    2033,2

is equivalent to writing out a straight line from 1 to 2 across eleven years.

This is convenient and occasionally wrong. A carbon budget that steps down in
2030, or an availability that is zero until a plant opens, must not be smoothed
into a ramp.

To switch interpolation off for specific parameters, create
``energy_system/parameters_interpolation_off.json``:

.. code-block:: json

    {
      "parameter_name": [
        "carbon_emissions_annual_limit",
        "demand_yearly_variation"
      ]
    }

For the parameters listed there, years without a specified value fall back to
the **default value** from ``attributes.json`` instead of being interpolated.

.. important::
    List the name of the *file*, not the underlying parameter. Writing
    ``demand_yearly_variation`` disables interpolation for
    ``demand_yearly_variation.csv``; it does not affect ``demand.csv``.


Exercises
=========

The exercises are cumulative.

1. **Grow electricity demand by 10% per year and confirm the model responds.**
   Add ``set_carriers/electricity/demand_yearly_variation.csv`` to
   ``5_multiple_time_steps_per_year``:

   .. code-block:: text

       year,demand_yearly_variation
       2023,1
       2024,1.1
       2025,1.2

   Run the model, then compare photovoltaic capacity across the three years.

   *Expected result: PV capacity is non-decreasing across the three years,
   because capacity built in one year remains available in the next and demand
   only grows. The hourly demand profile keeps its shape in every year: check
   the nodal energy balance for 2023 and 2025 and confirm the peaks fall at the
   same hours, just scaled.*

   *On this dataset: PV capacity in CH goes 5.99 -> 6.59 -> 7.19 GW, and in DE
   60.27 -> 66.29 -> 72.32 GW, non-decreasing at both nodes as expected.*

2. **Specify only the endpoints and let interpolation do the rest.** Replace
   the file with:

   .. code-block:: text

       year,demand_yearly_variation
       2023,1
       2025,1.2

   *Expected result: identical results to exercise 1. The value for 2024 is
   interpolated to 1.1, which is exactly what you wrote by hand before. Verify
   with* ``r.get_total('demand')`` *that 2024 demand is unchanged between the
   two runs.*

   *On this dataset: PV capacity and 2024 electricity demand (527.98 GWh CH,
   4502.04 GWh DE) come out identical to exercise 1, to the last digit.*

3. **Switch interpolation off and see the difference.** Keep the two-row file
   from exercise 2 and add
   ``energy_system/parameters_interpolation_off.json``:

   .. code-block:: json

       {
         "parameter_name": [
           "demand_yearly_variation"
         ]
       }

   *Expected result: 2024 no longer gets 1.1. It falls back to the default
   value of the yearly variation, which is 1, so 2024 demand drops back to the
   unscaled profile while 2023 and 2025 keep their specified values. This is
   what makes the setting worth knowing: switching interpolation off does not
   mean "leave the year out", it means "use the default".*

   *On this dataset: 2024 demand drops to exactly the unscaled 2023 baseline
   (479.98 GWh CH, 4092.77 GWh DE, versus 527.98 / 4502.04 in exercises 1-2),
   and PV capacity in both nodes goes flat between 2023 and 2024 (CH stays at
   5.99 GW, DE at 60.27 GW) before jumping to the exercise-1 level in 2025.
   Do not assume this stays non-decreasing on your own data; a case where
   later demand is lower than an earlier interpolated value would show
   capacity built for 2025 sitting oversized for the unscaled 2024.*

.. seealso::
    :ref:`input_structure.overwrite_defaults` documents the file layouts in
    full. :ref:`additional_features.year_specific_input_data` covers the
    related case where a single year needs an entirely different hourly time
    series rather than a scaled one.
