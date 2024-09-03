.. _input_data_handling:
################
Input data handling
################
.. _Attribute.json files:
Attribute.json files
==============
Each element in the input data folder has an ``attributes.json`` file, as shown in :ref:`Input data structure`, which defines the default values for the element.
This file must be specified for each element and must contain all parameters that this class of elements (Technology, Carrier, etc.) can have (see :ref:`optimization_problem`).

The ``attributes.json`` files have three main purposes:

1. Defining all parameters of each element
2. Providing the default value for each parameter
3. Defining the unit of each parameter

How are ``attributes.json`` files structured?
---------------------------------------------
The general structure of each ``attributes.json`` file is the following:

.. code-block::

    {
      "parameter_1": {
        "default_value": v_1,
        "unit": "u_1"
      },
      "parameter_2": {
        "default_value": v_2,
        "unit": "u_2"
      },
      ...
    }

The structure is a normal dictionary structure.
Make sure to have the correct positioning of the brackets.

* There is **one curly** bracket around all parameters ``{...}``
* Each parameter has a name, followed **by a colon and curly brackets** ``name: {...}``
* Inside the curly brackets are in most cases a ``default_value`` as a ``float`` or ``"inf"`` and a ``unit`` as a ``string`` (see :ref:`Unit consistency`).

What are particular parameters in the ``attributes.json`` file?
-------------------------------------------------------------
Some parameters do not have the structure above. These are the carriers of technologies (``"reference_carrier"``, ``"input_carrier"``, and ``"output_carrier"``), the ``"conversion_factor"`` of conversion technologies, and the ``"retrofit_flow_coupling_factor"`` of retrofitting technologies.

**Input, output, and reference carriers**

.. code-block::

    "output_carrier": {
      "default_value": [
        "heat",
        "electricity"
      ],
    }

The default value of the three carrier types are a list ``[..., ...]``. It can take the following lengths:

1. 1 carrier: Necessary in the case of the reference carrier
2. 0 carrier: Empty list if no input or output carrier
3. more than 1 carrier: for multiple input or output carriers

The units of the carriers in a technology are defined in the corresponding parameters (see :ref:`Unit consistency`) and are therefore omitted in the ``"reference_carrier"``, ``"input_carrier"``, and ``"output_carrier"`` field.

**Conversion factor**

The ``conversion_factor`` is the fixed ratio between a carrier flow and the reference carrier flow, defined for all dependent carriers, i.e., all carriers except the reference carrier.

.. code-block::

    dependent_carriers = input_carriers + output_carriers - reference_carrier

ZEN-garden will check i) that the reference carrier is not part of the input and output carriers, ii) that there is no overlap between the input and output carriers, and iii) that all carriers are defined in their respective folders in ``set_carriers``.
The default conversion factor is defined in ``attributes.json`` as:

.. code-block::

    "conversion_factor": [
      {
        "heat": {
          "default_value": 1.257,
          "unit": "GWh/GWh"
        }
      },
      {
        "natural_gas": {
          "default_value": 2.857,
          "unit": "GWh/GWh"
        }
      }
    ]
The conversion factor is **a list ``[...]`` with each dependent carrier wrapped in curly brackets**. Inside each curly bracket, there are the ``default_value`` and the ``unit``.

**Retrofitting flow coupling factor**

The retrofitting flow coupling factor couples the reference carrier flow of the retrofitting technology and the base technology (:ref:`Conversion Technologies`). The default value is defined in ``attributes.json`` as:

.. code-block::

    "retrofit_flow_coupling_factor": {
      "base_technology": <base_technology_name>,
      "default_value": 0.5,
      "unit": "GWh/GWh"
    }

The retrofitting flow coupling factor is a single parameter with the base technology as a string and the default value and unit as usual.

.. _Overwriting default values:
Overwriting default values
==========================
The paradigm of ZEN-garden is that the user only has to specify those input data that they want to specify.
Therefore, the user defines default values for all parameters in the ``attributes.json`` files.
Whenever more information is required, the user can overwrite the default values by providing a ``<parameter_name>.csv`` file in the same folder as the ``attributes.json`` file.

Let's assume the following example: The purpose of the energy system is to provide ``heat``, whose default ``demand`` is given as ``10 GW``:

.. code-block::

    {
      "demand": {
        "default_value": 10,
        "unit": "GW"
      }
    }

The energy system is modeled for two nodes, ``CH`` and ``DE`` and spans one year with 8760 time steps.

.. note::
    To retrieve the dimensions of a parameter, please refer to :ref:`optimization_problem` and to the ``index_names`` attribute in the parameter definition.

Providing extra .csv files
--------------------------
If the user wants to specify the demand ``CH`` and ``DE`` in the time steps ``0, 14, 300``, the user can create a file ``demand.csv``:

.. code-block::

    node,time,demand
    CH,0,5
    CH,14,7
    CH,300,3
    DE,0,2
    DE,14,3
    DE,300,2

The file overwrites the default value for the demand at nodes ``CH`` and ``DE`` in time steps ``0, 14,300``.

.. note::
    ZEN-garden will select that subset of data that is relevant for the optimization problem.
    If the user specifies a demand for a node in ``demand.csv`` that is not part of the optimization problem, the demand is ignored for this node.

To avoid overly long files, the dimensions can be unstacked, i.e., the values of one dimension can be the column names of the file:

.. code-block::

    node,0,14,300
    CH,5,7,3
    DE,2,3,2

or

.. code-block::

    time,CH,DE
    0,5,2
    14,7,3
    300,3,2

Therefore, the full demand time series is ``10 GW`` except for the time steps ``0, 14, 300`` where it is ``5 GW, 7 GW, 3 GW`` for ``CH`` and ``2 GW, 3 GW, 2 GW`` for ``DE``.

.. warning::
    Make sure that the unit of the values in the ``.csv`` file is consistent with the unit defined in the ``attributes.json`` file!
    Since we do not specify a unit in the ``.csv`` file, the unit of the values is assumed to be the same as the unit in the ``attributes.json`` file.

Constant dimensions
-------------------
Often, we have parameters that are constant over a certain dimension but change with another dimension.
For example, the demand of an industrial energy carrier might be constant over time but is different for all nodes.

In this case, the full ``demand.csv`` file would be:

.. code-block::

    node,0,1,2,...,8760
    CH,5,5,5,...,5
    DE,2,2,2,...,2

This is a very long file, and it is hard to see the structure of the data. Furthermore, it is prone to errors. Therefore, ZEN-garden allows you to drop dimensions that are constant. The file can be shortened to:

.. code-block::

    node,demand
    CH,5
    DE,2

The file is much shorter and easier to read. ZEN-garden will automatically fill in the missing dimensions with the constant value.

.. _Yearly variation:
Yearly variation
----------------
We specify hourly-dependent data for each hour of the year.
However, some parameters might have a yearly variation, e.g., the overall demand may increase or decrease over the year.

To this end, the user can specify a file ``<parameter_name>_yearly_variation.csv`` that multiplies the hourly-dependent data with a factor for each hour of the year.
ZEN-garden therefore assumes the same time series for each year but allows for the scaling of the time series with the yearly variation.
Per default, the yearly variation is assumed to be ``1``. Therefore, for missing values in ``<parameter_name>_yearly_variation.csv``, the hourly-dependent data is not scaled.

The user can specify the yearly variation for all dimensions except for the ``time`` dimension:

.. code-block::

    node,2020,2021,2022,...,2050
    CH,1,1.1,1.2,...,4
    DE,1,0.99,0.98,...,0.7

If all nodes have the same yearly variation, the file can be shortened to:

.. code-block::

    year,demand_yearly_variation
    2020,1
    2021,1.1
    2022,1.2
    ...
    2050,4

.. note::
    So far, ZEN-garden does not allow for different time series for each year but only for the scaling while keeping the same shape of the time series.

Data interpolation
-------------
To reduce the number of data points, ZEN-garden per-default interpolates the data points linearly between the given data points.
As an example, in :ref:`Yearly variation`, the demand increase or decrease is linear over the years.
So, the user can reduce the number of data points in the ``demand_yearly_variation.csv`` file:

.. code-block::

    year,demand_yearly_variation
    2020,1
    2050,4

If the user wants to disable the interpolation for a specific parameter, the user can create a ``parameters_interpolation_off.csv`` file and specify the parameter names in the file:

.. code-block::

    parameter_name
    demand_yearly_variation

.. note::
    The user must specify the file name, i.e., in the example above, the specified file is ``demand_yearly_variation.csv``, not ``demand.csv``.
    Therefore, the interpolation is only disabled for the yearly variation, not for the hourly-dependent data.

.. _PWA:
Piece-wise affine input data
----------------------------
In ZEN-garden, we can model the capital expenditure (CAPEX) of conversion technologies either linear or piece-wise affine (PWA).
In the linear case, the ``capex_specific_conversion`` parameter is treated like every other parameter, i.e., the user can specify a constant value and a ``.csv`` file.

In the PWA case, the user can specify a ``nonlinear_capex.csv`` file that contains the breakpoints and the CAPEX values of the PWA representation.
A PWA representation is a set of linear functions that are connected at the breakpoints. The breakpoints are the capacity additions :math:`\Delta S_m` with the corresponding CAPEX values :math:`\alpha_m`.

.. image:: ../images/PWA.png
    :alt: Piece-wise affine representation of CAPEX

The file ``nonlinear_capex.csv`` has the following structure:

.. code-block::

    capacity_addition,capex_specific_conversion
    0,2000
    20,1700
    40,1500
    60,1350
    80,1200
    100,1100
    120,1010
    140,940
    160,890
    180,860
    200,840
    GW,Euro/kW

.. note::

    Each new interval between two breakpoints adds a binary variable to the optimization problem, for each technology, each year, and each node. The binary variable is 1 if the capacity is in the interval and 0 otherwise.
    The user is advised to keep the number of breakpoints low to avoid a combinatorial explosion of binary variables.

.. _Unit consistency:
Unit consistency
================
Our models describe physical processes, whose numeric values are always connected to a physical unit. For example, the capacity of a coal power plant is a power, thus the unit is, e.g., GW.
In our optimization models, we use a large variety of different technologies and carriers, for which the input data is often provided in different units. The optimization problem itself however only accepts numeric values.

Thus, we have to make sure that the numeric values have the same physical base unit, i.e., if our input data for technology A is in GW and for technology B in MW, we want to convert both numeric values to, e.g., GW.
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
You have to provide an input unit for all attributes in the input files. The unit is added as the ``unit`` field after the default value in the ``attributes.json`` file (:ref:`Attribute.json files`).

Defining new units
------------------

We are using the package `pint <https://pint.readthedocs.io/en/stable/>`_, which already has the most common units defined. However, some exotic ones, such as ``Euro``, are not yet defined. You can add new units in the file ``system_specification/unit_definitions.txt``:

.. code-block::

    Euro = [currency] = EURO = Eur
    pkm = [mileage] = passenger_km = passenger_kilometer

Here, we make use of the existing dimensionality ``[currency]``. If there is a unit you want to define with a dimensionality that does not exist yet, you can define it the same way:
``pkm = [mileage]``.
The unit ``pkm`` now has the dimensionality ``[mileage]``.

**What do I have to look out for?**

There are a few rules to follow in choosing the base units:

1. The base units must be exhaustive, thus all input units must be represented as a combination of the base units (i.e., ``Euro/MWh => Euro/GW/hour``). Each base unit can only be raised to the power 1, -1, or 0. We do not want to represent a unit by any base unit with a different exponent, e.g., ``km => (m^3)^(1/3)``
2. The base units themselves can not be linearly dependent, e.g., you cannot choose the base units ``GW``, ``hour`` and ``GJ``.
3. The dimensionalities must be unique. While you can use ``m^3`` and ``km``, you cannot use both ``MW`` and ``GW``. You will get a warning if you define the same unit twice.

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

    In the results (:ref:`Accessing results`), you can retrieve the unit of all parameters and variables by calling ``r.get_unit(<variable/parameter name>)``, where ``r`` is a results object.

What are known errors with pint?
--------------------------------

The ``pint`` package that we use for the unit handling has amazing functionalities but also some hurdles to look out for. The ones we have already found are:

* ``ton``: pint uses the keyword ``ton`` for imperial ton, not the metric ton. The keyword for those are ``metric_ton`` or ``tonne``. However, per default, ZEN-garden overwrites the definition of ``ton`` to be the metric ton, so ``ton`` and ``tonne`` can be used interchangeably. If you for some reason want to use imperial tons, set ``solver["define_ton_as_metric_ton"] = False``.
* ``h``: Until recently, ``h`` was treated as the planck constant, not hour. Fortunately, this has been fixed in Feb 2023. If you encounter this error, please update your pint version.

.. _Scaling:
Scaling
=============
