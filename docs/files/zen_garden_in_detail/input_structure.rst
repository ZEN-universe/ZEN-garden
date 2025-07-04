.. _input_structure.input_structure:

####################
Input data structure
####################

The input data is structured in a folder hierarchy. The root folder 
``<data_folder>`` contains a subfolder for each dataset and a configuration 
file ``config.json``. ZEN-garden is run from this root folder (:ref:`running.running`). 
The dataset folder ``<dataset>`` comprises the input data for a 
specific dataset and must contain the following files and subfolders:

.. code-block::

    <data_folder>/
    |--<dataset>/
    |   |--energy_system/
    |   |   |--attributes.json
    |   |   |--base_units.csv
    |   |   |--set_nodes.csv
    |   |   `--set_edges.csv
    |   |
    |   |--set_carriers/
    |   |   |--<carrier1>/
    |   |   |   `--attributes.json
    |   |   `--<carrier2>/
    |   |       `--attributes.json
    |   |
    |   |--set_technologies/
    |   |   |--set_conversion_technologies/
    |   |   |   |--<conversion_technology1>/
    |   |   |   |   `--attributes.json
    |   |   |   |
    |   |   |   `--<conversion_technology2>/
    |   |   |       `--attributes.json
    |   |   |
    |   |   |--set_storage_technologies/
    |   |   |      `--<storage_technology1>/
    |   |   |          `--attributes.json
    |   |   |
    |   |   `--set_transport_technologies/
    |   |       `--<transport_technology1>/
    |   |           `--attributes.json
    |   |
    |   `--system.json
    |
    `--config.json

Note that all folder names in ``<>`` in the structure above can be chosen 
freely. The dataset is described by the properties of the ``energy_system``, the 
``set_carriers``, and the ``set_technologies``. The system configuration is 
stored in the file ``system.json`` and defines dataset-specific settings, e.g., 
which technologies to model or how many years and time steps to include. The 
configuration file ``config.json`` contains more general settings for the 
optimization problem and the solver. Refer to the section :ref:`configuration.configuration`
for more details.

Depending on your analysis, more files can be added; see 
:ref:`input_handling.attribute_files` and 
:ref:`t_scenario.t_scenario` for more information.


.. _input_structure.energy_system:

Energy System
==============

The folder ``energy_system`` contains four necessary files: ``attributes.json``, 
``base_units.csv``, ``set_nodes.csv``, and ``set_edges.csv``. The file 
``attributes.json`` defines the numerical setup of the energy system, e.g., the 
carbon emission limits, the discount rate, or the carbon price. 
``set_nodes.csv`` and ``set_edges.csv`` define the nodes and edges of the energy 
system graph, respectively. ``set_nodes.csv`` contains the coordinates of the 
nodes, which are used to calculate the default distance of the edges.

There is no predefined convention for naming nodes and edges, so the user can 
choose the naming freely. In the examples, we use ``<node1>-<node2>`` to name 
edges, but note that you are not forced to follow that convention. In fact, 
``set_edges.csv`` defines the edges by the nodes they connect.

.. note::
    You can specify more nodes in ``set_nodes.csv`` than you end up using. In 
    ``system.json`` you can define a subset of nodes you want to select in the 
    model. If you do not specify any nodes in ``system.json``, all nodes from 
    ``set_nodes.csv`` are used.

``base_units.csv`` define the base units in the model. That means that all units 
in the model are converted to a combination of base units. See 
:ref:`input_handling.unit_consistancy` for more information.


.. _input_structure.technologies:

Technologies
==============
The ``set_technologies`` folder is specified in three subfolders: 
``set_conversion_technologies``, ``set_storage_technologies``, and 
``set_transport_technologies``. Each technology has its own folder in the 
respective subfolder and must contain the ``attributes.json`` file. Additional 
files can further parametrize the technologies (see :ref:`input_handling.attribute_files`).

.. note::
    You can specify more technologies in the three subfolders than you end up 
    using. That can be helpful if you want to model different scenarios with 
    different technologies and carriers.

Each technology has a reference carrier, i.e., that carrier by which the 
capacity of the technology is rated. As an example, a :math:`10kW` heat pump 
could refer to :math:`10kW_{th}` heat output or :math:`10kW_{el}` electricity 
input. Hence, the user has to specify which carrier is the reference carrier in 
the file ``attributes.json``. For storage technologies and transport 
technologies, the reference carrier is the carrier that is stored or 
transported, respectively.


.. _input_structure.conversion_technologies:

Conversion Technologies
-----------------------

The conversion technologies are defined in the folder 
``set_conversion_technologies``. A conversion technology converts ``0`` to 
``n`` input carriers into ``0`` to ``m`` output carriers. Note that the 
conversion factor between the carriers is fixed, e.g., a combined heat and 
power (CHP) plant cannot sometimes generate more heat and sometimes generate 
more electricity. The file ``attributes.json`` defines the properties of the 
conversion technology, e.g., the capacity limit, the maximum load, the 
conversion factor, or the investment cost.

A special case of the conversion technologies are retrofitting technologies. 
These technologies are defined in the folder 
``set_conversion_technologies\set_retrofitting_technologies``, if any exist.
They behave equal to conversion technologies, but they are always connected to 
a conversion technology. They are coupled to a conversion technology by the 
attribute ``retrofit_flow_coupling_factor`` in the file ``attributes.json``, 
which couples the reference carrier flow of the retrofitting technology and the 
base technology. A possible application of retrofitting technologies is the 
installation of a carbon-capture unit on top of a power plant. In this case, 
the base technology would be ``power_plant`` and the retrofitting technology 
would be ``carbon_capture``. Refer to the dataset example 
``14_retrofitting_and_fuel_substitution`` for more information.


.. _input_structure.storage_technologies:

Storage Technologies
--------------------

The storage technologies are defined in the folder ``set_storage_technologies``.
A storage technology connects two time steps by charging at ``t=t0`` and 
discharging at ``t=t1``.

.. note::
    In ZEN-garden, the power-rated (charging-discharging) capacity and 
    energy-rated (storage level) capacity of storage technologies are optimized 
    independently.     If you want to fix the energy-to-power ratio, the 
    attribute ``energy_to_power_ratio`` in ``attributes.json`` can be set to 
    anything different than ``inf``.


Transport Technologies
----------------------

The transport technologies are defined in the folder 
``set_transport_technologies``. A transport technology connects two nodes via an 
edge. Different to conversion technologies or storage technologies, transport 
technology capacities are built on the edges, not the nodes.

.. note::
    By default, the distance of an edge will be calculated as the `haversine 
    distance <https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/>`_ 
    between the nodes. This can be overwritten for specific edges in a 
    ``distance.csv`` file (see :ref:`input_handling.attribute_files`).


.. _input_structure.carriers:

Carriers
==============

Each energy carrier is defined in its own folder in ``set_carriers``. You do not 
need to specify the used energy carriers explicitly in ``system.json``, but the 
carriers are implied from the used technologies. All input, output, and 
reference carriers that are used in the selected technologies 
(see `input_structure.technologies`_) must be defined in the ``set_carriers`` folder. The file 
``attributes.json`` defines the properties of the carrier, e.g., the carbon 
intensity or the cost of the carrier. Additional files can further parametrize 
the carriers (see :ref:`input_handling.attribute_files`).

.. note::
    You can specify more carriers in ``set_carriers`` than you end up using. 
    That can be helpful if you want to model different scenarios with different 
    technologies and carriers.

