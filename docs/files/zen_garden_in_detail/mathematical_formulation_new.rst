:orphan:

.. _mathematical_formulation.mathematical_formulation:

Mathematical formulation
========================

ZEN-garden formulates energy-system design and operation as a mixed-integer
linear program (MILP), or as a linear program (LP) when no binary decisions are
required. The following
sections describe the constraints and equations that define the optimization
problem. For a complete list of symbols, see :ref:`notation.notation`.  

.. note::

   Important notes on the development and maintenance of this formulation:

   * constraint ``build`` docstrings are the source of truth for equations and 
     implementation-specific conditions. Do not write equations into this file. 
     If an equation changes, its docstring should be updated in the same change.
   * :ref:`notation.notation` is the source of truth for sets, parameters,
     variables, symbols, time-step types, descriptions, and units. If a
     symbol or component description changes, its entry in the notation tables
     should be updated instead.


Objectives and cost accounting
------------------------------

Two objective functions are available:

1. minimize cumulative net present cost
2. minimize cumulative emissions



Total annual system cost
^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.energy_system.CostTotalConstraint.build
   :sections: summary, formulation

Net present cost
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.energy_system.NetPresentCostConstraint.build
   :sections: summary, formulation

Annualized technology CAPEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^

CAPEX symbols and parameters are defined in :ref:`notation.notation`.

.. docstring_method:: zen_garden.constraints.technology.CostCapexYearlyConstraint.build
   :sections: summary, formulation

Total annual technology CAPEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.CostCapexYearlyTotalConstraint.build
   :sections: summary, formulation

Annual technology OPEX
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.CostOpexYearlyConstraint.build
   :sections: summary, formulation

Total annual technology OPEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.CostOpexYearlyTotalConstraint.build
   :sections: summary, formulation

Carrier import and export cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Carrier parameters and variables are catalogued in :ref:`notation.notation`.

.. docstring_method:: zen_garden.constraints.carrier.CostCarrierConstraint.build
   :sections: summary, formulation

Demand-shedding cost and limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.carrier.CostLimitShedDemandConstraint.build
   :sections: summary, formulation

Total annual carrier cost
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.carrier.CostCarrierTotalConstraint.build
   :sections: summary, formulation

Carbon-emissions cost
^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.energy_system.CostCarbonEmissionsTotalConstraint.build
   :sections: summary, formulation


Technology operating costs and emissions
----------------------------------------

The following constraints calculate time-dependent variable OPEX and operating
emissions. Their annual aggregation is documented in the preceding section and
in :ref:`model_formulation_docstrings.emissions`.

Conversion technologies
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.conversion_technology.OpexEmissionsTechnologyConversionConstraint.build
   :sections: summary, formulation

Storage technologies
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.storage_technology.OpexEmissionsTechnologyStorageConstraint.build
   :sections: summary, formulation

Transport technologies
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.transport_technology.OpexEmissionsTechnologyTransportConstraint.build
   :sections: summary, formulation


Carrier balance and external exchange
-------------------------------------

The nodal balance equates all sources and sinks for every carrier, node, and
operational time step. Imports, exports, and shed demand provide the external
exchange and feasibility terms. Their notation is defined in
:ref:`notation.notation`.

Nodal carrier balance
^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.carrier.NodalEnergyBalanceConstraint.build
   :sections: summary, formulation

Time-dependent import and export availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.carrier.AvailabilityImportExportConstraint.build
   :sections: summary, formulation

Annual import and export availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.carrier.AvailabilityImportExportYearlyConstraint.build
   :sections: summary, formulation


.. _model_formulation_docstrings.emissions:

Emissions accounting and limits
-------------------------------

The parameter and variable definitions used by these constraints are provided
in :ref:`notation.notation`.

Carrier emissions
^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.carrier.CarbonEmissionsCarrierConstraint.build
   :sections: summary, formulation

Total annual carrier emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.carrier.CarbonEmissionsCarrierTotalConstraint.build
   :sections: summary, formulation

Total annual technology emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.CarbonEmissionsTechnologyTotalConstraint.build
   :sections: summary, formulation

Total annual system emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.energy_system.CarbonEmissionsAnnualConstraint.build
   :sections: summary, formulation

Cumulative emissions
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.energy_system.CarbonEmissionsCumulativeConstraint.build
   :sections: summary, formulation

Annual emissions limit
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.energy_system.CarbonEmissionsAnnualLimitConstraint.build
   :sections: summary, formulation

Annual-limit overshoot
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.energy_system.CarbonEmissionsAnnualOvershootConstraint.build
   :sections: summary, formulation

Cumulative emissions budget
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.energy_system.CarbonEmissionsBudgetConstraint.build
   :sections: summary, formulation

Budget overshoot
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.energy_system.CarbonEmissionsBudgetOvershootConstraint.build
   :sections: summary, formulation


Operational technology constraints
----------------------------------

Conversion, storage, transport, and retrofit technology notation is defined in
the corresponding sections of :ref:`notation.notation`.

Conversion capacity factor
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.conversion_technology.CapacityFactorConversionConstraint.build
   :sections: summary, formulation

Carrier conversion
^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.conversion_technology.CarrierConversionConstraint.build
   :sections: summary, formulation

Minimum full-load hours
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.conversion_technology.MinimumFullLoadHoursConstraint.build
   :sections: summary, formulation

Storage capacity factor
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.storage_technology.CapacityFactorStorageConstraint.build
   :sections: summary, formulation

Storage-level coupling
^^^^^^^^^^^^^^^^^^^^^^

The storage-level formulation maps storage time steps to operational time
steps. The canonical storage variables and temporal parameters are listed in
:ref:`notation.notation`.

.. docstring_method:: zen_garden.constraints.storage_technology.CoupleStorageLevelConstraint.build
   :sections: summary, formulation

Maximum storage level
^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.storage_technology.StorageLevelMaxConstraint.build
   :sections: summary, formulation

Storage spillage
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.storage_technology.FlowStorageSpillageConstraint.build
   :sections: summary, formulation

Mutually exclusive charging and discharging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.storage_technology.ChargeDischargeBinaryConstraint.build
   :sections: summary, formulation

Transport capacity factor
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.transport_technology.CapacityFactorTransportConstraint.build
   :sections: summary, formulation

Transport losses
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.transport_technology.TransportTechnologyLossesFlowConstraint.build
   :sections: summary, formulation

Retrofit flow coupling
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.retrofitting_technology.RetrofitFlowCouplingConstraint.build
   :sections: summary, formulation

Technology on/off operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This generic constraint applies the appropriate activity-flow expression for
each technology type.

.. docstring_method:: zen_garden.constraints.technology.TechnologyOnOffConstraint.build
   :sections: summary, formulation


Investment and capacity constraints
-----------------------------------

These constraints determine when capacity becomes available, how long it
remains active, and which bounds apply to additions and total installed
capacity. General technology notation is defined in
:ref:`notation.notation`.

Technology lifetime
^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.TechnologyLifetimeConstraint.build
   :sections: summary, formulation

Capacity upper limit
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.TechnologyCapacityLimitConstraint.build
   :sections: summary, formulation

Capacity lower limit
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.TechnologyCapacityLowerLimitConstraint.build
   :sections: summary, formulation

Minimum capacity addition
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.TechnologyMinCapacityAdditionConstraint.build
   :sections: summary, formulation

Maximum capacity addition
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.TechnologyMaxCapacityAdditionConstraint.build
   :sections: summary, formulation

Construction time
^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.TechnologyConstructionTimeConstraint.build
   :sections: summary, formulation

Technology diffusion
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.technology.TechnologyDiffusionLimitConstraint.build
   :sections: summary, formulation

Linear conversion CAPEX
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.conversion_technology.LinearCapexConstraint.build
   :sections: summary, formulation

Storage energy-to-power ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.storage_technology.CapacityEnergyToPowerRatioConstraint.build
   :sections: summary, formulation

Storage CAPEX
^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.storage_technology.StorageTechnologyCapexConstraint.build
   :sections: summary, formulation

Transport CAPEX
^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.constraints.transport_technology.TransportTechnologyCapexConstraint.build
   :sections: summary, formulation

