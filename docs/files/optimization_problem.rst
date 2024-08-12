################
Optimization problem
################

.. _optimization_setup:
Optimization Setup
==================


.. _energy_system:
Energy System
==================

The class ``EnergySystem`` defines the energy system and takes the optimization setup as input.

.. csv-table:: Energy System Parameters
    :header: Parameter Name,Description,Name,Time Step Type,Doc String,Scope,Unit Category
    :widths: 20 20 20 20 20 20 20 20
    :stub-columns: 1

    ``carbon\_emissions\_annual\_limit`` , ``set\_time\_steps\_yearly`` , Parameter which specifies the total limit on carbon emissions ,  {"emissions": 1},
    ``carbon\_emissions\_budget`` , temporal immutable , Parameter which specifies the total budget of carbon emissions until the end of the entire time horizon ,  {"emissions": 1},
    ``carbon\_emissions\_cumulative\_existing`` , temporal immutable , Parameter which specifies the total previous carbon emissions ,  {"emissions": 1},
    ``price\_carbon\_emissions`` , ``set\_time\_steps\_yearly`` , Parameter which specifies the yearly carbon price ,  {"money": 1, "emissions": -1},
    ``price\_carbon\_emissions\_budget\_overshoot`` , temporal immutable , Parameter which specifies the carbon price for budget overshoot ,  {"money": 1, "emissions": -1},
    ``price\_carbon\_emissions\_annual\_overshoot`` , temporal immutable , Parameter which specifies the carbon price for annual overshoot ,  {"money": 1, "emissions": -1},
    ``market\_share\_unbounded`` , temporal immutable , Parameter which specifies the unbounded market share ,  {},
    , ``knowledge\_spillover\_rate`` , temporal immutable , Parameter which specifies the knowledge spillover rate , {},
    ``time\_steps\_operation\_duration`` , ``set\_time\_steps\_operation`` , Parameter which specifies the time step duration in operation for all technologies ,  {"time": 1},

.. _carrier:
Carrier
==================

.. csv-table:: Carrier Parameters
    :header: Parameter Name,Description,Name,Time Step Type,Doc String,Unit Category
    :widths: 20 20 20 20 20 20 20 20
    :stub-columns: 1

    ``demand`` , ``set\_time\_steps\_operation`` , Parameter which specifies the carrier demand , {"energy_quantity": 1, "time": -1},
    ``availability\_import`` , ``set\_time\_steps\_operation`` , Parameter which specifies the maximum energy that can be imported from outside the system boundaries ,  {"energy_quantity": 1, "time": -1},
    ``availability\_export`` , ``set\_time\_steps\_operation`` , Parameter which specifies the maximum energy that can be exported to outside the system boundaries ,  {"energy_quantity": 1, "time": -1},
    ``availability\_import\_yearly`` , ``set\_time\_steps\_yearly`` , Parameter which specifies the maximum energy that can be imported from outside the system boundaries for the entire year ,  {"energy_quantity": 1},
    ``availability\_export\_yearly`` , ``set\_time\_steps\_yearly`` , Parameter which specifies the maximum energy that can be exported to outside the system boundaries for the entire year`` ,  {"energy_quantity": 1},
    ``price\_import`` , ``set\_time\_steps\_operation`` , Parameter which specifies the import carrier price ,  {"money": 1, "energy_quantity": -1},
    ``price\_export`` , ``set\_time\_steps\_operation``, Parameter which specifies the export carrier price ,  {"money": 1, "energy_quantity": -1},
    ``price\_shed\_demand`` , ``temporal immutable`` , Parameter which specifies the price to shed demand ,  {"money": 1, "energy_quantity": -1},
    ``carbon\_intensity_carrier``, ``set\_time\_steps\_yearly`` , Parameter which specifies the carbon intensity of   {"emissions": 1, "energy_quantity": -1},



.. _technology:
Technology
==================

.. csv-table:: Technology Parameters
    :header: Parameter Name,Description,Name,Time Step Type,Doc String,Unit Category
    :widths: 20 20 20 20 20 20 20
    :stub-columns: 1

    ``capacity\_existing`` , temporal immutable , Parameter which specifies the existing technology size , {"energy_quantity": 1, "time": -1},
    ``capacity\_investment\_existing`` , ``set\_time\_steps\_yearly``\_entire\_horizon , Parameter which specifies the size of the previously invested capacities , {"energy_quantity": 1, "time": -1},
    ``capacity\_addition\_min`` , temporal immutable , Parameter which specifies the minimum capacity addition that can be installed , {"energy_quantity": 1, "time": -1},
    ``capacity\_addition\_max`` , temporal immutable , Parameter which specifies the maximum capacity addition that can be installed , {"energy_quantity": 1, "time": -1},
    ``capacity\_addition\_unbounded`` , temporal immutable , Parameter which specifies the unbounded capacity addition that can be added each year (only for delayed technology deployment) , {"energy_quantity": 1, "time": -1},
    ``lifetime\_existing`` , temporal immutable , Parameter which specifies the remaining lifetime of an existing technology , {},
    ``capex\_capacity\_existing`` , temporal immutable , Parameter which specifies the total capex of an existing technology which still has to be paid , {"money": 1, "energy_quantity": -1},
    ``opex\_specific\_variable`` , ``set\_time\_steps\_operation`` , Parameter which specifies the variable specific opex , {"money": 1, "energy_quantity": -1},
    ``opex\_specific\_fixed`` , ``set\_time\_steps\_yearly`` , Parameter which specifies the fixed annual specific opex , {"money": 1, "energy_quantity": -1, "time": 1},
    ``lifetime`` , temporal immutable , Parameter which specifies the lifetime of a newly built technology , {},
    ``construction\_time`` , temporal immutable , Parameter which specifies the construction time of a newly built technology , {},
    ``max\_diffusion\_rate`` , ``set\_time\_steps\_yearly`` , Parameter which specifies the maximum diffusion rate which is the maximum increase in capacity between investment steps , {},
    ``capacity\_limit`` , temporal immutable , Parameter which specifies the capacity limit of technologies , {"energy_quantity": 1, "time": -1},
    ``min\_load`` , ``set\_time\_steps\_operation`` , Parameter which specifies the minimum load of technology relative to installed capacity , {},
    ``max\_load`` , ``set\_time\_steps\_operation`` , Parameter which specifies the maximum load of technology relative to installed capacity , {},
    ``carbon\_intensity\_technology`` , temporal immutable , Parameter which specifies the carbon intensity of each technology , {"emissions": 1, "energy_quantity": -1},



.. _conversion_technology:
Conversion Technology
----------------------

.. csv-table:: Conversion Technology Parameters
    :header: Parameter Name,Description,Name,Time Step Type,Doc String,Unit Category
    :widths: 20 20 20 20 20 20 20
    :stub-columns: 1

    ``capex\_specific\_conversion`` , ``set\_time\_steps\_yearly`` , Parameter which specifies the slope of the capex if approximated linearly , {"money": 1, "energy_quantity": -1, "time": 1},
    ``conversion\_factor`` , ``set\_time\_steps\_yearly`` , Parameter which specifies the slope of the conversion efficiency if approximated linearly , {"energy_quantity": 1, "energy_quantity": -1},


*Retrofitting Technology**

.. csv-table:: Retrofitting Technology Parameters
    :header: Parameter Name,Description,Name,Time Step Type,Doc String,Unit Category
    :widths: 20 20 20 20 20 20 20
    :stub-columns: 1

    ``retrofit\_flow\_coupling\_factor`` , ``set\_time\_steps\_operation`` , Parameter which specifies the flow coupling between the retrofitting technologies and its base technology , technology, {"energy_quantity": 1, "energy_quantity": -1},

.. _storage_technology:
Storage Technology
----------------------

.. csv-table:: Storage Technology Parameters
    :header: Parameter Name,Description,Name,Time Step Type,Doc String,Unit Category
    :widths: 20 20 20 20 20 20 20
    :stub-columns: 1

    ``time\_steps\_storage\_level\_duration`` , ``set\_time\_steps\_storage\_level`` , Parameter which specifies the time step duration in StorageLevel for all technologies , {"time": 1},
    ``efficiency\_charge`` , ``set\_time\_steps\_yearly`` , efficiency during charging for storage technologies , {},
    ``efficiency\_discharge`` , ``set\_time\_steps\_yearly`` , efficiency during discharging for storage technologies , {},
    ``self\_discharge`` , temporal immutable , self-discharge of storage technologies , {},
    ``capex\_specific\_storage`` , ``set\_time\_steps\_yearly`` , specific capex of storage technologies , {"money": 1, "energy_quantity": -1, "time": 1},

.. _transport_technology:
Transport Technology
----------------------

.. csv-table:: Transport Technology Parameters
    :header: Parameter Name,Description,Name,Time Step Type,Doc String,Unit Category
    :widths: 20 20 20 20 20 20 20
    :stub-columns: 1

    ``distance`` , temporal immutable , distance between two nodes for transport technologies , {"distance": 1},
    ``capex\_specific\_transport`` , ``set\_time\_steps\_yearly`` , capex per unit for transport technologies , {"money": 1, "energy_quantity": -1, "time": 1},
    ``capex\_per\_distance\_transport`` , ``set\_time\_steps\_yearly`` , capex per distance for transport technologies , {"money": 1, "distance": -1, "energy_quantity": -1, "time": 1},
    ``transport\_loss\_factor`` , temporal immutable , carrier losses due to transport with transport technologies , {"distance": -1},
    ``transport\_loss\_factor\_exponential`` , temporal immutable , exponential carrier losses due to transport with transport technologies , {"distance": -1},





