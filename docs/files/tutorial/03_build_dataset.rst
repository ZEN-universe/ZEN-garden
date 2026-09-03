.. _t_build.t_build:

############################
Build a dataset from scratch
############################

.. admonition:: At a glance
   :class: note

   | **You will** build a working ZEN-garden dataset from an empty folder, starting with one node and one technology.
   | **You need** ZEN-garden installed (:ref:`installation.installation`). You do **not** need to have run an example first.

Most models start from an existing dataset. Building one from nothing is more
work, but it is the fastest way to understand why the input folders look the
way they do — and you will need it when your system has no close relative among
the examples.

We will build ``my_first_model``: a single node that has to meet an electricity
demand, supplied by photovoltaics. Then we will grow it.

.. tip::
    In reality, you will rarely build a dataset from scratch.
    We have developed **ZEN-creator**, with which you can easily create ZEN-garden
    datasets. Check out 
    `The ZEN-creator Documentation <https://zen-creator.readthedocs.io/en/latest/>`_ 
    for more information.

.. _t_build.skeleton:

Step 1: create the folder skeleton
==================================

A dataset is a folder of ``.yaml`` and ``.csv`` files. Create this structure:

.. code-block:: text

    <data>/
    |--my_first_model/
    |   |--energy_system/
    |   |--set_carriers/
    |   |--set_technologies/
    |   |   |--set_conversion_technologies/
    |   |   |--set_storage_technologies/
    |   |   `--set_transport_technologies/
    |   `--system.yaml
    |
    `--config.yaml

All three technology subfolders must exist, even when empty.

``config.yaml`` sits **outside** the dataset folder and names the dataset to
run:

.. code-block:: yaml

    analysis:
      dataset: my_first_model


.. _t_build.energy_system:

Step 2: describe the energy system
==================================

The ``energy_system`` folder defines the physical setting: where the nodes are,
what units the model uses, and system-wide economics.

``energy_system/set_nodes.csv`` lists the nodes and their coordinates. The
coordinates are used to compute default edge distances:

.. code-block:: text

    node,x,y
    CH,8.23,46.80

``energy_system/set_edges.csv`` lists the edges. With one node there are none,
but the file must exist with its header:

.. code-block:: text

    edge,node_from,node_to

``energy_system/base_units.yaml`` defines the units everything is converted to:

.. code-block:: yaml

    unit:
      - hour
      - GW
      - km
      - megatons
      - megaEuro

``energy_system/unit_definitions.txt`` adds units Pint does not already know.
At minimum, define the currency:

.. code-block:: text

    Euro = [currency] = EURO = Eur

``energy_system/attributes.yaml`` holds system-wide parameters. Every parameter
of the energy system needs a default value:

.. code-block:: yaml

    carbon_emissions_annual_limit:
      default_value: inf
      unit: megatons
    carbon_emissions_budget:
      default_value: inf
      unit: megatons
    carbon_emissions_cumulative_existing:
      default_value: 0.0
      unit: megatons
    price_carbon_emissions:
      default_value: 0.0
      unit: Euro/tons
    price_carbon_emissions_budget_overshoot:
      default_value: inf
      unit: Euro/tons
    price_carbon_emissions_annual_overshoot:
      default_value: inf
      unit: Euro/tons
    knowledge_depreciation_rate:
      default_value: 0.1
      unit: "1"
    knowledge_spillover_rate:
      default_value: 0.0
      unit: "1"
    market_share_unbounded:
      default_value: 0.1
      unit: "1"
    discount_rate:
      default_value: 0.06
      unit: "1"

Setting the emission limits and prices to ``inf`` and ``0`` switches those
mechanisms off for now; :ref:`t_emissions.t_emissions` turns them on.

.. seealso::
    :ref:`input_structure.energy_system` describes each of these files, and
    :ref:`t_units.t_units` explains how base units are chosen.


.. _t_build.carrier:

Step 3: add a carrier
=====================

Each carrier gets a folder under ``set_carriers`` containing an
``attributes.yaml`` that defines **every** parameter a carrier can have:

``set_carriers/electricity/attributes.yaml``

.. code-block:: yaml

    demand:
      default_value: 10.0
      unit: GW
    availability_import:
      default_value: 0.0
      unit: GW
    availability_export:
      default_value: 0.0
      unit: GW
    availability_import_yearly:
      default_value: inf
      unit: GWh
    availability_export_yearly:
      default_value: inf
      unit: GWh
    price_import:
      default_value: 0.0
      unit: kiloEuro/GWh
    price_export:
      default_value: 0.0
      unit: kiloEuro/GWh
    price_shed_demand:
      default_value: inf
      unit: kiloEuro/GWh
    carbon_intensity_carrier_import:
      default_value: 0.0
      unit: kilotons/GWh
    carbon_intensity_carrier_export:
      default_value: 0.0
      unit: kilotons/GWh

Two choices worth noting. ``availability_import`` is ``0``, so electricity
cannot simply be bought — it has to be generated, which is the point of the
model. ``price_shed_demand`` is ``inf``, so demand may not be dropped; if the
model cannot meet demand it will be infeasible rather than quietly shedding it.

.. note::
    Carriers are **not** listed in ``system.yaml``. ZEN-garden infers them from
    the technologies you select.


.. _t_build.technology:

Step 4: add a technology
========================

``set_technologies/set_conversion_technologies/photovoltaics/attributes.yaml``

.. code-block:: yaml

    reference_carrier:
      default_value:
        - electricity
    input_carrier:
      default_value: []
    output_carrier:
      default_value:
        - electricity
    conversion_factor: []
    capacity_limit:
      default_value: inf
      unit: GW
    capacity_existing:
      default_value: 0.0
      unit: GW
    capacity_investment_existing:
      default_value: 0.0
      unit: GW
    capacity_addition_min:
      default_value: 0.0
      unit: GW
    capacity_addition_max:
      default_value: inf
      unit: GW
    capacity_addition_unbounded:
      default_value: 0.0
      unit: GW
    max_diffusion_rate:
      default_value: inf
      unit: "1"
    min_load:
      default_value: 0.0
      unit: "1"
    max_load:
      default_value: 1.0
      unit: "1"
    lifetime:
      default_value: 25.0
      unit: "1"
    construction_time:
      default_value: 0.0
      unit: "1"
    carbon_intensity_technology:
      default_value: 0.0
      unit: kilotons/GWh
    capex_specific_conversion:
      default_value: 700.0
      unit: Euro/kW
    opex_specific_fixed:
      default_value: 10.0
      unit: Euro/kW
    opex_specific_variable:
      default_value: 0.0
      unit: kiloEuro/GWh

Photovoltaics has no input carrier, so ``input_carrier`` is an empty list and
``conversion_factor`` is empty too. The ``reference_carrier`` is electricity,
which means the capacity is rated in GW of electricity output.

.. warning::
    ``attributes.yaml`` must contain **all** parameters of the element type,
    not just the ones you care about. A missing parameter is an error, not a
    silent default. :ref:`notation.notation` lists the parameters of each
    element type.


.. _t_build.system:

Step 5: write system.yaml
=========================

``system.yaml`` selects what actually enters the optimization:

.. code-block:: yaml

    set_conversion_technologies:
      - photovoltaics
    set_storage_technologies: []
    set_transport_technologies: []
    set_nodes:
      - CH
    reference_year: 2024
    unaggregated_time_steps_per_year: 24
    aggregated_time_steps_per_year: 24
    conduct_time_series_aggregation: false
    optimized_years: 1
    interval_between_years: 1

Start small: one year, 24 hours, one node. You can grow every one of these
numbers once the model runs.


.. _t_build.run:

Step 6: run it
==============

.. code-block:: shell

    cd <data>
    zen-garden --dataset="my_first_model"

.. tip::
    Use ``zen-garden --dataset="my_first_model" --no_solve`` to build the
    optimization problem without solving it. This catches input-data errors in
    seconds and is the fastest way to iterate while a dataset is still taking
    shape.

If the run fails, work through :ref:`troubleshooting.troubleshooting`. The most
common first-time errors are inconsistent indentation in a ``.yaml`` file, a
missing parameter in ``attributes.yaml``, and inconsistent units.


Exercises
=========

The exercises are cumulative — each builds on the model from the previous one.

1. **Make the demand vary over the day.** Create
   ``set_carriers/electricity/demand.csv``:

   .. code-block:: text

       node,time,demand
       CH,0,5
       CH,8,12
       CH,18,20

   *Expected result: the model still solves. Demand is 10 GW — the default from
   ``attributes.yaml`` — in every hour except 0, 8 and 18. Installed PV
   capacity is now set by the 20 GW peak in hour 18 rather than by the flat
   default. This is the core idea of the input format: the ``.yaml`` gives the
   default, the ``.csv`` overwrites it where you care.*

2. **Add a second node and connect it.** Add ``DE`` to ``set_nodes.csv`` and to
   ``set_nodes`` in ``system.yaml``.

   *Expected result: the model solves with PV built at both nodes
   independently. Because there is no transport technology yet, each node must
   meet its own demand — check that the capacity at each node matches its own
   peak.*

3. **Add a natural gas boiler, so the model has a choice.** This means a second
   carrier (``natural_gas``, with ``availability_import`` set to ``inf`` and a
   non-zero ``price_import``), a second technology folder, and a
   ``conversion_factor`` linking gas input to heat output. Follow
   :ref:`t_add_techs_carriers.t_add_techs_carriers`, which covers exactly this
   step in detail.

   *Expected result: you now have a model with a genuine investment trade-off,
   which is where the rest of the tutorials start.*

.. seealso::
    Once your dataset runs, :ref:`t_analyze.t_analyze` shows how to read the
    results, and :ref:`t_yearly.t_yearly` shows how to extend it across years
    without writing out every value.
