.. _t_storage.t_storage:

#######
Storage
#######

.. admonition:: At a glance
   :class: note

   | **You will** size a storage technology, understand why power and energy capacity 
      are optimized separately, and control how the storage level behaves across years.
   | **You need** the setup from :ref:`tutorials_intro.setup`.

.. _t_storage.two_capacities:

Two capacities, not one
=======================

A storage technology in ZEN-garden has **two independent capacities**:

* **Power capacity** — how fast it can charge and discharge, for example, in GW. 
  This is the capacity referred to by ``capacity_limit``, ``capacity_existing``,
  ``capex_specific_storage`` and so on.
* **Energy capacity** — how much it can hold, e.g., in GWh. Every power-rated
  parameter has an energy-rated twin ending in ``_energy``:
  ``capacity_limit_energy``, ``capacity_existing_energy``,
  ``capex_specific_storage_energy``, ``opex_specific_fixed_energy``,
  ``capacity_addition_min_energy``, ``capacity_addition_max_energy``.

ZEN-garden optimizes the two independently, so a battery and a seasonal gas
store are described by the same technology class with different cost ratios.

To fix the ratio between them, e.g., for a battery that is always "four-hour" storage, 
constrain it:

.. code-block:: json

    "energy_to_power_ratio_min": {
      "default_value": 4,
      "unit": "h"
    },
    "energy_to_power_ratio_max": {
      "default_value": 4,
      "unit": "h"
    }

The defaults are ``0`` and ``inf``, which leaves the ratio free.

.. warning::
    There is no single ``energy_to_power_ratio`` parameter. Setting the minimum
    alone lets the optimizer build an arbitrarily large reservoir; setting the
    maximum alone lets it build an arbitrarily fast one. Fixing a ratio means
    setting both.


.. _t_storage.losses:

Losses
======

Three parameters describe how energy is lost:

* ``efficiency_charge`` and ``efficiency_discharge``: round-trip losses,
  applied at each conversion. A round-trip efficiency of 95% is roughly
  ``0.9747`` on each leg.
* ``self_discharge``: the fraction of the stored level lost per time step.
  This is what distinguishes a battery from a gas cavern over long horizons: a
  small per-hour value compounds heavily across a season.

``flow_storage_inflow`` adds an exogenous inflow, which is how a reservoir
hydro plant is modelled.


.. _t_storage.periodicity:

Periodicity
===========

``storage_periodicity`` in ``system.json`` (default ``true``) requires the
storage level at the end of each year to equal the level at the start. Without
it, the optimizer would happily start the year with a full store it never paid
to fill.

``multiyear_periodicity`` (default ``false``) relaxes this to the whole
planning horizon instead of each year: the level at the start of the horizon
must equal the level at the end, but individual years may end fuller or emptier
than they started.

.. code-block:: json

    "storage_periodicity": true,
    "multiyear_periodicity": true

This matters when supply varies between years, e.g., a year of high gas
availability followed by a scarce one, because it lets a store carry energy
across the year boundary.

.. note::
    Multi-year periodicity currently requires ``interval_between_years`` to be
    ``1``.

.. seealso::
    Under time series aggregation the sequence of time steps is no longer each hour, 
    which makes storage modelling harder. ZEN-garden uses a
    formulation that preserves both short- and long-term storage behaviour; see
    :ref:`time_representation.storage_tsa` and :ref:`t_tsa.t_tsa`.


Exercises
=========

The exercises are cumulative and use ``5_multiple_time_steps_per_year``, which
already contains a ``natural_gas_storage``. Work on a copy.

1. **Find out whether the storage is used at all, and why not.** Run the
   dataset and read the storage capacity and level:

   .. code:: python

       from zen_garden import Results
       r = Results(path='<data>/outputs/<your_dataset>')
       print(r.get_total('capacity', index="natural_gas_storage"))
       print(r.get_full_ts('storage_level', index="natural_gas_storage", year=2023))

   *Expected result: Very little storage is built. Natural gas can be imported
   in CH without limit at a constant price, and can then be transported to DE, 
   so there is nothing to shift — a store only makes sense when supply is 
   constrained or prices vary.*


2. **Give the storage a reason to exist.** Add an entry for ``CH`` in
   ``set_carriers/natural_gas/availability_import.csv`` limiting Swiss
   imports to 50 GW:

   .. code-block:: text

       node,availability_import
       CH,50

   *Expected result: the model now builds natural gas storage.*

   *On this dataset: DE builds 822 GWh of storage energy capacity and 50 GW of 
   storage power capacity. CH builds 8.2 GWh of storage energy capacity and 1.7 GW of 
   storage power capacity.*

3. **Turn on multi-year periodicity and compare.** Set
   ``"multiyear_periodicity": true`` in ``system.json`` and add a 
   ``availability_import_yearly_variation.csv`` file in the 
   ``set_carriers/natural_gas/`` directory:

    .. code-block:: text
  
        year,availability_import_yearly_variation
        2023,1
        2024,0.5
        2025,0

   *Expected result: storage energy capacity increases and the storage level
   trace no longer returns to the same value at each year boundary.*

   *On this dataset: DE and CH together show almost 5000 GWh of storage energy capacity 
   and more than 100 GW of storage power capacity. The storage level increases over
   the first year, is stable in the second year, and decreases in the third year.*
