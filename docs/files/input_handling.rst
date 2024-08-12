################
Input data handling
################
.. _Default values:
Default values
==============

Structure of attributes file with exceptions

.. _Unit consistency:
Unit consistency
================
Our models describe physical processes, whose numeric values are always connected to a physical unit. For example, the capacity of a coal power plant is a power, thus the unit is, e.g., GW.
In our optimization models, we use a large variety of different technologies and carriers, for which the input data is often provided in different units. The optimization problem itself however only accepts numeric values.
Thus, we have to make sure that the numeric values have the same physical base unit, i.e., if our input for technology A is in GW and for technology B in MW, we want to convert both numeric values to, e.g., GW.

Another reason for using a base unit is to `keep the numerics of the optimization model in check <https://www.gurobi.com/documentation/9.5/refman/guidelines_for_numerical_i.html>`_.

What is ZEN-garden's approach to convert all numeric values to common base units?
-------------------------------------------------------------------------------------

We define a set of base units, which we can combine to represent each dimensionality in our model. Each unit has a certain physical dimensionality:

.. code-block::

    km => [distance]
    hour => [time]
    Euro => [currency]
    GW => [mass]^1 [length]^2 [time]^-3
We make use of the fact, that we can combine the base units to any unit by comparing the dimensionalities. For example, Euro/MWh can be converted to:

.. code-block::

    Euro/MWh
    => [currency]^1 [mass]^-1 [length]-2 [time]^2
    = [currency]^1 [[mass]^1 [length]^2 [time]^-3]^-1 [time]^-1
    => Euro/GW/hour

We convert the units by calculating the multiplier

.. code-block::

    (Euro/GW/hour)/(Euro/MWh) = (MW)/(GW) = 0.001

and multiplying the numeric value with the multiplier.

The base units are defined in the input data set in the file ``/energy_system/base_units.csv``.
You have to provide an input unit for all attributes in the input files. The unit is added as the ``unit`` field after the default value in the ``attributes.json`` file.

Defining new units
------------------

We are using the package `pint <https://pint.readthedocs.io/en/stable/>`_, which already has the most common units defined. However, some exotic ones, such as ``Euro``, are not yet defined. You can add new units in the file ``system_specification/unit_definitions.txt``:

.. code-block::

    Euro = [currency] = EURO = Eur
    pkm = [mileage] = passenger_km = passenger_kilometer

Here, we make use of the existing dimensionality ``[currency]``. If there is a unit you want to define with a dimensionality, that does not exist yet, you can define it the same way:
``pkm = [mileage]``.
The unit ``pkm`` now has the dimensionality ``[mileage]``.

**What do I have to look out for?**

There are a few rules to follow in choosing the base units:

1. The base units must be exhaustive, thus all input units must be represented as a combination of the base units (i.e., ``Euro/MWh => Euro/GW/hour``). Each base unit can only be raised to the power 1, -1, or 0. We do not want to represent a unit by any base unit with a different exponent, e.g., ``km => (m^3)^(1/3)``
2. The base units themselves can not be linearly dependent, e.g., you cannot choose the base units ``GW``, ``hour`` and ``GJ``.
3. The dimensionalities must be unique. While you can use ``m^3`` and ``km``, you cannot use both ``MW`` and ``GW``. You will get a warning if you define the same unit twice, but that is still ok.

The code will output errors or warnings, if the selection of base units is wrong, so play around with the base units and see what works and what doesn't.

Enforcing unit consistency
--------------------------

Converting all numeric values to the same set of base units enforces that all magnitudes are comparable; however, this does not ensure that the units are consistent across parameters and elements (technologies and carriers).
For example, a user might have defined the capacity of an electrolyzer in ``GW``, but the investment costs in ``Euro/(ton/hour)``.

To enforce unit consistency, ZEN-garden checks the units of all parameters and elements and throws an error if the units are not consistent.
In particular, ZEN-garden connects technologies to their reference carriers and checks if the units of the reference carriers are consistent with the units of the technology parameters.
If any inconsistency is found, ZEN-garden tries to guess the inconsistent unit (the least common unit) and displays it in the error message.

After ensuring unit consistency, ZEN-garden implies the units of all variables in the optimization problem based on the units of the parameters.
Each variable definition (``variable.add_variable()``) has the argument ``unit_category`` that defines the combination of units and can look like ``unit_category={"energy_quantity": 1, "time": -1}``.

.. note::

    In the results, you can retrieve the unit of all parameters and variables by calling ``r.get_unit(<variable/parameter name>)``, where ``r`` is a results object.

What are known errors with pint?
--------------------------------

The ``pint`` package that we use for the unit handling has amazing functionalities but also some hurdles to look out for. The ones we have already found are:

* ``h``: While we might interpret ``h`` as hour, it is actually treated as the planck constant. Please use ``hour`` or in combination with another unit ``GWh``. If you try to use ``h``, e.g., ``ton/h``, ZEN-garden throws an exception
* ``ton``: pint uses the keyword ``ton`` for imperial ton, not the metric ton. The keyword for those are ``metric_ton`` or ``tonne``. However, per default, ZEN-garden overwrites the definition of ``ton`` to be the metric ton, so ``ton`` and ``tonne`` can be used interchangeably. If you for some reason want to use imperial tons, set ``solver["define_ton_as_metric_ton"] = False``.

.. _Scaling:
Scaling
=============
