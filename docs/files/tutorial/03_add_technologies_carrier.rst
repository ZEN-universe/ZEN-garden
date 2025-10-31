.. _t_add_techs_carriers.t_add_techs_carriers:

#########################################
Tutorial 3: Add Technologies and Carriers
#########################################

In this scenario, we want to introduce how to add new technologies and carriers from an existing dataset.
In general, adding new technologies and carriers involves the following steps:

1.  Copy an existing dataset to a new folder.
2.  Copy-paste an existing technology or carrier and rename it.
3.  Modify the parameters of the new technology or carrier as desired.
4.  Add new ``.csv`` files if necessary.
5.  Add the new technology to the appropriate sets in ``system.json``. Carriers are
    automatically detected from the technologies.
6.  Run the model with the new dataset.

This tutorial assumes that you have installed and run the example dataset
``5_multiple_time_steps_per_year`` as described in the tutorial :ref:`setup
instructions <tutorials_intro.setup>`.

Practically, we want to extend the example dataset ``5_multiple_time_steps_per_year`` by a new technology
called ``biomass_boiler`` which uses the carrier ``biomass`` to produce heat.
Therefore, we need to add the new technology ``biomass_boiler`` and the new carrier ``biomass`` to the dataset.

.. _t_add_techs_carriers.copy_folder:

Step 1: Copy the existing dataset
=================================

Copy-paste the entire folder of the dataset example ``5_multiple_time_steps_per_year`` to a new folder
called ``tutorial_3_add_technologies_carrier``.

.. _t_add_techs_carriers.copy_tech_carrier:

Step 2: Copy-paste an existing technology and carrier and rename it
===================================================================

It is easiest to copy an existing technology and modify it to create a new technology.
Of course, you can also create a new technology from scratch by creating a new ``attributes.json`` file
(:ref:`input_structure.attribute_files`).

In this case, we copy-paste the existing technology ``natural_gas_boiler`` to a new folder
``biomass_boiler`` inside the folder ``set_conversion_technologies``.

Analogously, we copy-paste the existing carrier ``natural_gas`` to a new folder
``biomass`` inside the folder ``set_carriers``.

.. _t_add_techs_carriers.modify_tech_carrier:

Step 3: Modify the parameters of the new technology and carrier
===============================================================

Now, we need to modify the parameters of the new technology and carrier.

The ``biomass_boiler`` folder is very similar to the ``natural_gas_boiler`` technology, so we only need to change
the name of the technology and the input carrier from ``natural_gas`` to ``biomass``. Furthermore, we want to
change the efficiency of the biomass boiler to 0.8. Since the ``conversion_factor`` is expressed as
``dependent_carrier/reference_carrier=[biomass]/[heat]``, we set the ``conversion_factor`` to 1/0.8=1.25.
Make sure to change the name of the ``dependent_carrier`` in the ``attributes.json`` file from
``natural_gas`` to ``biomass``.

We also want to increase the capex of the biomass boiler to 1500 Euro/kW.

Everything else remains the same:

* The reference and output carrier is still ``heat``.
* The units remain the same (kW/MW/GW).
* All other values remain the same.

For our new carrier ``biomass``, we want to change the ``carbon_intensity_import`` and ``carbon_intensity_export``
to 0.0, and increase the ``price_import`` to 40 kEuro/GWh. Everything else remains the same.

.. _t_add_techs_carriers.add_csv:

Step 4: Add new .csv files if necessary
=======================================

We want to allow infinite import of biomass, so we can remove the existing file
``availability_import.csv``.

However, we want to make ``biomass`` cheaper than ``natural_gas`` in CH, so we create a new file
``price_import.csv`` in the folder ``set_carriers/biomass/`` with the following content:

.. code-block::

    node,price_import
    CH,15

.. _t_add_techs_carriers.modify_system:

Step 5: Add the new technology to system.json
=============================================

Finally, we need to add the new technology ``biomass_boiler`` to the ``set_conversion_technologies`` in ``system.json``.
We do not need to add the new carrier ``biomass`` explicitly,
since carriers are automatically detected from the technologies.

.. _t_add_techs_carriers.run_model:

Step 6: Run the model with the new dataset
==========================================

Lastly, we can run the model with the new dataset ``tutorial_3_add_technologies_carrier``
(assuming that you are in the folder where the ``config.json`` file and the new folder are located):

.. code:: bash

    python -m zen_garden --dataset=tutorial_3_add_technologies_carrier

Example Exercise
================

1.  **What share of the heat in CH is generated from the biomass boiler in 2023?**

    View the production of heat using the ZEN-garden visualization platform,
    as described in the tutorial on :ref:`analyzing outputs<t_analyze.t_analyze>`.

    `Solution: 71.7 % (1560 GWh from biomass boiler / 2177 GWh total heat production)`

2.  **Using the same recipe as above, add a wood_mill technology that produces biomass from wood but
    but requires electricity as an input carrier**

    a. Copy-paste the existing technology ``biomass_boiler`` to a new folder
       ``wood_mill`` inside the folder ``set_conversion_technologies``.
    b. Copy-paste the existing carrier ``biomass`` to a new folder
       ``wood`` inside the folder ``set_carriers``.
    c. Change the ``reference_carrier`` of the ``wood_mill`` to ``wood``, the ``input_carrier`` to``
       ``["electricity","wood"]``, and the ``output_carrier`` to ``biomass``. Note that for the first time, the
       reference carrier is not the same as the output carrier.
    d. The unit of ``wood`` is ``ton`` for energy quantities (equivalent to ``GWh``) and ``ton/h`` for power
       quantities (equivalent to ``GW``). Change the unit of all parameters of the ``wood_mill``
       and ``wood`` carrier accordingly.
    e. Assume that 1 ton of wood produces 3 MWh of biomass (LHV of dry biomass: 5.4 kWh/kg,
       conversion efficiency: 55.5 %). Furthermore, assume that the wood mill requires 0.5 MWh of electricity per ton
       of processed wood.
       Set the ``conversion_factor`` of the ``wood_mill`` accordingly.
    f. We assume a cheap ``wood_mill`` with capex of 10 Euro/(ton/h) and a ``opex_specific_fixed=0``
    g. Prohibit the import of ``biomass`` by setting ``availability_import`` to 0 in the ``attributes.json`` file.
    h. Change the default ``price_import`` of ``wood`` to 20 Euro/ton in the ``attributes.json`` file and the price for
       ``CH`` to 2 Euro/ton in a the ``price_import.csv`` file.
    i. Finally, add the new technology ``wood_mill`` to the ``set_conversion_technologies`` in ``system.json`` and
       run the model. If you get any unit errors, check that all units are consistently changed.
    j. How much wood is consumed in CH in 2023?

       `Solution: 6.13 Mtons of wood`
    k. What share of the electricity consumption in DE is used for the wood mill in 2023?

       `Solution: 40.11 % (3.06 TWh for wood mill / 7.64 TWh total electricity consumption)`
