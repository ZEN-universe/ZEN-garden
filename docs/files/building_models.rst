################
Building a model
################
Input data structure
==============
The input data is structured in a folder hierarchy. The root folder ``<data_folder>`` contains a subfolder for each dataset and a configuration file ``config.json``.
ZEN-garden is run from this root folder. The dataset folder ``<dataset>`` comprises the input data for a specific dataset and must contain the following files and subfolders:

.. code-block::

    <data_folder>/
    |--<dataset>/
    | |--energy_system/
    | | |--attributes.json
    | | |--base_units.csv
    | | |--set_nodes.csv
    | | `--set_edges.csv
    | |
    | |--set_carriers/
    | | |--<carrier1>/
    | | | `--attributes.json
    | | `--<carrier2>/
    | |   `--attributes.json
    | |
    | |--set_technologies/
    | | |--set_conversion_technologies/
    | | | |--<conversion_technology1>/
    | | | | --attributes.json
    | | | |
    | | | `--<conversion_technology2>/
    | | |   `--attributes.json
    | | |
    | | |--set_storage_technologies/
    | | |  `--<storage_technology1>/
    | | |    `--attributes.json
    | | |
    | | `--set_transport_technologies/
    | |    `--<transport_technology1>/
    | |      `--attributes.json
    | |
    | `--system.json
    |
    `--config.json

Note that all folder names in ``<>`` in the structure above can be chosen freely. The dataset is describes by the properties of the ``energy_system``, the ``set_carriers``, and the ``set_technologies``.
The system configuration is stored in the file ``system.json`` and defines dataset-specific settings, e.g., which technologies to model or how many years and time steps to include.
The configuration file ``config.json`` contains more general settings for the optimization problem and the solver. Refer to the section `Configurations`_ for more details.

Depending on your analysis, more files can be added; see `Default values`_ and `Scenario analysis`_ for more information.

Energy System
-------------

The folder ``energy_system`` contains four necessary files: ``attributes.json``, ``base_units.csv``, ``set_nodes.csv``, and ``set_edges.csv``.
The file ``attributes.json`` defines the numerical setup of the energy system, e.g., the carbon emission limits, the discount rate, or the carbon price.
``set_nodes.csv`` and ``set_edges.csv`` define the nodes and edges of the energy system graph, respectively.
``set_nodes.csv`` contains the coordinates of the nodes, which are used to calculate the default distance of the edges.

There is no predefined convention for naming nodes and edges, so the user can choose the naming freely.
In the examples, we use ``<node1>-<node2>`` to name edges, but note that you are not forced to follow that convention.
In fact, ``set_edges.csv`` defines the edges by the nodes they connect.

.. note::
    You can specify more nodes in ``set_nodes.csv`` than you end up using. In ``system.json`` you can define a subset of nodes you want to select in the model. If you do not specify any nodes in ``system.json``, all nodes from ``set_nodes.csv`` are used.

``base_units.csv`` define the base units in the model. That means that all units in the model are converted to a combination of base units.
See `Unit consistency`_ for more information.

Technologies
------------
The ``set_technologies`` folder is specified in three subfolders: ``set_conversion_technologies``, ``set_storage_technologies``, and ``set_transport_technologies``.
Each technology has its own folder in the respective subfolder. Additional files can further parametrize the carriers (see `Default values`_).

.. note::
    You can specify more technologies in the three subfolders than you end up using. That can be helpful if you want to model different scenarios with different technologies and carriers.

Each technology has a reference carrier, i.e., that carrier by which the capacity of the technology is rated.
As an example, a :math:`10kW` heat pump could refer to :math:`10kW_{th}` heat output or :math:`10kW_{el}` electricity input.
Hence, the user has to specify which carrier is the reference carrier in the file ``attributes.json``.
For storage technologies and transport technologies, the reference carrier is the carrier that is stored or transported, respectively.

**Conversion Technologies**

The conversion technologies are defined in the folder ``set_conversion_technologies``.
A conversion technology converts ``0`` to ``n`` input carriers into ``0`` to ``m`` output carriers.
Note that the conversion factor between the carriers is fixed, e.g., a combined heat and power (CHP) plant cannot sometimes generate more heat and sometimes generate more electricity.
The file ``attributes.json`` defines the properties of the conversion technology, e.g., the capacity limit, the maximum load, the conversion factor, or the investment cost.

A special case of the conversion technologies are retrofitting technologies. These technologies are defined in the folder ``set_conversion_technologies\set_retrofitting_technologies``, if any exist.
They behave equal to conversion technologies, but they are always connected to a conversion technology. They are coupled to a conversion technology by the attribute ``retrofit_flow_coupling_factor`` in the file ``attributes.json``, which couples the reference carrier flow of the retrofitting technology and the base technology.
A possible application of retrofitting technologies is the installation of a carbon-capture unit on top of a power plant. In this case, the base technology would be ``power_plant`` and the retrofitting technology would be ``carbon_capture``. Refer to the dataset example XXXX for more information.

**Storage Technologies**

The storage technologies are defined in the folder ``set_storage_technologies``.
A storage technology connects two time steps by charging at ``t=t0`` and discharging at ``t=t1``.

.. note::
    In ZEN-garden, the power-rated (charging-discharging) capacity and energy-rated (storage level) capacity of storage technologies are optimized independently.
    If you want to fix the energy-to-power ratio, the attribute ``energy_to_power_ratio`` in ``attributes.json`` can be set to anything different than ``inf``

**Transport Technologies**

The transport technologies are defined in the folder ``set_transport_technologies``.
A transport technology connects two nodes via an edge. Different to conversion technologies or storage technologies, transport technology capacities are built on the edges not the nodes.

.. note::
    By default, the distance of an edge will be calculated as the `haversine distance <https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/>`_ between the nodes. This can be overwritten for specific edges in a ``distance.csv`` file (see `Default values`_)

Carriers
-------------
Each energy carrier is defined in its own folder in ``set_carriers``. You do not need to specify the used energy carriers explicitly in ``system.json``, but the carriers are implied from the used technologies.
All input, output, and reference carriers that are used in the selected technologies (see `Technologies`_) must be defined in the ``set_carriers`` folder.
The file ``attributes.json`` defines the properties of the carrier, e.g., the carbon intensity or the cost of the carrier.
Additional files can further parametrize the carriers (see `Default values`_).

.. note::
    You can specify more carriers in ``set_carriers`` than you end up using. That can be helpful if you want to model different scenarios with different technologies and carriers.

Input data handling
==============
Default values
--------------

Structure of attributes file with exceptions

Unit consistency
--------------
Our models describe physical processes, whose numeric values are always connected to a physical unit. For example, the capacity of a coal power plant is a power, thus the unit is, e.g., GW.
In our optimization models, we use a large variety of different technologies and carriers, for which the input data is often provided in different units. The optimization problem itself however only accepts numeric values.
Thus, we have to make sure that the numeric values have the same physical base unit, i.e., if our input for technology A is in GW and for technology B in MW, we want to convert both numeric values to, e.g., GW.

Another reason for using a base unit is to `keep the numerics of the optimization model in check <https://www.gurobi.com/documentation/9.5/refman/guidelines_for_numerical_i.html>`_.

**What is ZEN-garden's approach to convert all numeric values to common base units?**

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

**Defining new units**

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

**Enforcing unit consistency**

Converting all numeric values to the same set of base units enforces that all magnitudes are comparable; however, this does not ensure that the units are consistent across parameters and elements (technologies and carriers).
For example, a user might have defined the capacity of an electrolyzer in ``GW``, but the investment costs in ``Euro/(ton/hour)``.

To enforce unit consistency, ZEN-garden checks the units of all parameters and elements and throws an error if the units are not consistent.
In particular, ZEN-garden connects technologies to their reference carriers and checks if the units of the reference carriers are consistent with the units of the technology parameters.
If any inconsistency is found, ZEN-garden tries to guess the inconsistent unit (the least common unit) and displays it in the error message.

After ensuring unit consistency, ZEN-garden implies the units of all variables in the optimization problem based on the units of the parameters.
Each variable definition (``variable.add_variable()``) has the argument ``unit_category`` that defines the combination of units and can look like ``unit_category={"energy_quantity": 1, "time": -1}``.
In the results, you can retrieve the unit of all parameters and variables by calling ``r.get_unit(<variable/parameter name>)``, where ``r`` is a results object.

**What are known errors with pint?**

The ``pint`` package that we use for the unit handling has amazing functionalities but also some hurdles to look out for. The ones we have already found are:

* ``h``: While we might interpret ``h`` as hour, it is actually treated as the planck constant. Please use ``hour`` or in combination with another unit ``GWh``. If you try to use ``h``, e.g., ``ton/h``, ZEN-garden throws an exception
* ``ton``: pint uses the keyword ``ton`` for imperial ton, not the metric ton. The keyword for those are ``metric_ton`` or ``tonne``. However, per default, ZEN-garden overwrites the definition of ``ton`` to be the metric ton, so ``ton`` and ``tonne`` can be used interchangeably. If you for some reason want to use imperial tons, set ``solver["define_ton_as_metric_ton"] = False``.

Scaling
--------------

Optimization problem
==============


Scenario analysis
==============


Configurations
==============
System, analysis, solver settings
--------------

Time series aggregation and representation
--------------
**Time steps in ZEN-garden**
ZEN-garden is a temporally resolved investment and operation optimization model. That means that in general we have three different time indices:

1. ``set_base_time_steps``: is the highest resolution in the model. It is not necessarily used to index any component, but merely as a common "beat" or "rhythm" to all other time indices. We consider each hour as the base time index. Thus, each time index can be converted to the base time index, which is then a sequence of the time steps with the length of the base time index. This sequence is called ``sequence_time_steps``. The number of occurrences of each time step is called ``time_steps_duration``.
2. ``set_time_steps_yearly``: Some components have a yearly resolution. These include for example the yearly carbon emission limit (``carbon_emissions_limit``) or the yearly costs (``cost_total``). Note that these are in general not associated with any specific element (technology or carrier).
3. ``set_time_steps_operation``: The operation of built capacities is resolved on a higher resolution than the yearly time steps. For the technologies and the carriers, this is the index ``set_time_steps_operation``.

**The time parameters in ZEN-garden**

* ``reference_year``: First year of the optimization. Used to calculate the remaining lifetime of the existing capacities and the following years of the optimization.
* ``unaggregated_time_steps_per_year``: number of base time steps per optimization year. Must be <= 8760 (total number of hours per year)
* ``aggregated_time_steps_per_year``: number of representative periods per year to aggregate the time series. Thus, all operational components are aggregated to ``aggregated_time_steps_per_year`` time steps. For further information on time series aggregation, see below.
* ``optimized_years``: number of investigated years.
* ``interval_between_years``: interval between two optimization years.
Example:

.. code-block::

    "reference_year": 2020,
    "optimized_years": 4,
    "interval_between_years": 10

The resulting investigated years are

.. code-block::

    [2020,2030,2040,2050]

* ``use_rolling_horizon``: if True, we do not optimize all years simultaneously but optimize for a subset of years and afterward move the optimization window to the next year and optimize again. For further information on rolling horizon and myopic foresight versus perfect foresight refer to, e.g., `Poncelet et al. 2016 <10.1109/EEM.2016.7521261>`_.
* ``years_in_rolling_horizon``: number of optimization periods in the subset of the optimization horizon as mentioned above. Only relevant if ``use_rolling_horizon`` is True.
* ``interval_between_optimizations``: number of optimization periods for which the decisions of each rolling horizon are saved. Must be shorter than ``years_in_rolling_horizon``; default is 1. For an example for varying decision horizon lengths, refer to `Keppo et al. 2010 <10.1016/J.ENERGY.2010.01.019>`_. Only relevant if ``use_rolling_horizon`` is True.
Example:

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

**What is the idea of time series aggregation (TSA)?**

Full time series with 8760 time steps per year are often too large so that the optimization takes too long or cannot be solved at all in feasible times.
Thus, we apply a time series aggregation (TSA) which reduces the number of time steps by aggregating time steps with similar input values to a single time step.
By doing so, we can represent our full time series (8760 base time steps) by representative time steps, e.g., 200.

**I don't investigate hourly behavior or I want to investigate a full time series. What do I do?**

Open the ``system.json`` file and set ``"conduct_time_series_aggregation"=False``. This disables the time series aggregation. If you do not want to investigate a full year, set ``"unaggregated_time_steps_per_year"<8760``

**I want to use the time series aggregation. What do I do?**

Open the ``system.json`` file and set ``"aggregated_time_steps_per_year"`` smaller than ``"unaggregated_time_steps_per_year"``. You are then aggregating ``"unaggregated_time_steps_per_year"`` (e.g., 8760 base time steps) to ``"aggregated_time_steps_per_year"`` (e.g., 200 representative time steps).
If you mistakingly set ``"aggregated_time_steps_per_year">"unaggregated_time_steps_per_year"``, don't worry, the TSA is disabled and it behaves as if ``"aggregated_time_steps_per_year"="unaggregated_time_steps_per_year"``.

For an in-depth introduction to TSA, refer to `Hoffmann et al. 2020 <https://www.mdpi.com/1996-1073/13/3/641>`_. The authors at FZ Jülich are also the developers of the TSA package `tsam <https://tsam.readthedocs.io/en/latest/>`_ that we are using in ZEN-garden.

**How are short-term and long-term storages modeled?**

The modeling of storage technologies with TSA is challenging because storages couple time steps (see `Technologies`_).
Hence, the sequence of time steps is important for the operation of the storage level.
There are different approaches to model storages with TSA, with the approaches by `Gabrielli et al. 2018 <10.1016/J.APENERGY.2017.07.142>`_ and `Kotzur et al. <10.1016/J.APENERGY.2018.01.023>`_ being the most common.
In ZEN-garden, we extend the approach by Gabrielli et al. 2018 to model storages with TSA. The approach is detailed in `Mannhardt et al. 2023 <10.1016/j.isci.2023.106750>`_.
In short, every time that the sequence of operational time steps changes, the another storage time step is added. This increases the number of variables, but explicitly enables short- and long-term storages.
In particular, this storage level representation leads to fewer time steps than the full time series without loss of information.

**Great, the TSA works. But I want more information!**

1. In the ``default_config.py``, you find the class ``TimeSeriesAggregation`` where you can set the ``clusterMethod``, ``solver``, ``extremePeriodMethod`` and ``representationMethod``. Most importantly, the ``clusterMethod`` selects which algorithm is used to determine the clusters of representative time steps. Probably, the most common ones are `k_means <https://en.wikipedia.org/wiki/K-means_clustering>`_ and `k_medoids <https://en.wikipedia.org/wiki/K-medoids>`_. While it is probably not necessary at this point to understand the difference of k-means and k-medoids in detail, it is important to know that k-means averages the input data over the representative time steps, which reduces the extreme period behavior, thus, peaks are smoothened.
2. As said before, each aggregated time step represents multiple base time steps. Thus, the behavior in each aggregated time step accounts for more than one time step. Thus, the operational costs and operational carbon emissions of each aggregated time step are multiplied with the ``time_steps_operation_duration`` of the respective time step.
3. What is this strange ``sequence_time_steps`` floating around everywhere in the code? The substitution of the base time steps by the aggregated time steps yields a sequence of time steps, which is ``len(set_base_time_steps)`` entries long and encapsulates the order in which the aggregated time steps appear in the representation of the base time steps. We use the sequence of time steps to convert one time step into another. For example we can use the order to get the yearly time step associated with a certain operational time step, or the year of a certain operational time step.


