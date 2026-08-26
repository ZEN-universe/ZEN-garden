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
   * The objective function definitions are documented here because they are implemented
     as objective methods rather than constraint ``build`` methods.
   * :ref:`notation.notation` is the source of truth for sets, parameters,
     variables, symbols, time-step types, descriptions, and units. If a
     symbol or component description changes, its entry in the notation tables
     should be updated instead.


Objectives and cost accounting
------------------------------

Two objective functions are available:

1. minimize cumulative net present cost
2. minimize cumulative emissions


Minimizing net present cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The net present cost of the energy system is minimized over the planning
horizon:

.. math::
    :label: min_cost_new

    \mathrm{min} \quad \sum_{y\in\mathcal{Y}} C^{\mathrm{NPC}}_y

Minimizing total emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^

The cumulative carbon emissions at the end of the planning horizon are
minimized:

.. math::
    :label: min_emissions_new

    \mathrm{min} \quad M^{\mathrm{cum}}_Y



Total annual system cost
^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CostTotalConstraint.build
   :sections: summary, formulation

Net present cost
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.NetPresentCostConstraint.build
   :sections: summary, formulation

Annualized technology CAPEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^

CAPEX symbols and parameters are defined in :ref:`notation.notation`.

.. docstring_method:: zen_garden.elements.technology.constraints.CostCapexYearlyConstraint.build
   :sections: summary, formulation

Total annual technology CAPEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.CostCapexYearlyTotalConstraint.build
   :sections: summary, formulation

Annual technology OPEX
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.CostOpexYearlyConstraint.build
   :sections: summary, formulation

Total annual technology OPEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.CostOpexYearlyTotalConstraint.build
   :sections: summary, formulation

Carrier import and export cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Carrier parameters and variables are catalogued in :ref:`notation.notation`.

.. docstring_method:: zen_garden.elements.carrier.constraints.CostCarrierConstraint.build
   :sections: summary, formulation

Demand-shedding cost and limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.CostLimitShedDemandConstraint.build
   :sections: summary, formulation

Total annual carrier cost
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.CostCarrierTotalConstraint.build
   :sections: summary, formulation

Carbon-emissions cost
^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CostCarbonEmissionsTotalConstraint.build
   :sections: summary, formulation


Technology operating costs and emissions
----------------------------------------

The following constraints calculate time-dependent variable OPEX and operating
emissions. Their annual aggregation is documented in the preceding section and
in :ref:`model_formulation_docstrings.emissions`.

Conversion technologies
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.OpexEmissionsTechnologyConversionConstraint.build
   :sections: summary, formulation

Storage technologies
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.OpexEmissionsTechnologyStorageConstraint.build
   :sections: summary, formulation

Transport technologies
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.transport_technology.constraints.OpexEmissionsTechnologyTransportConstraint.build
   :sections: summary, formulation


Carrier balance and external exchange
-------------------------------------

The nodal balance equates all sources and sinks for every carrier, node, and
operational time step. Imports, exports, and shed demand provide the external
exchange and feasibility terms. Their notation is defined in
:ref:`notation.notation`.

Nodal carrier balance
^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.NodalEnergyBalanceConstraint.build
   :sections: summary, formulation

Time-dependent import and export availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityImportExportConstraint.build
   :sections: summary, formulation

Annual import and export availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityImportExportYearlyConstraint.build
   :sections: summary, formulation


.. _model_formulation_docstrings.emissions:

Emissions accounting and limits
-------------------------------

The parameter and variable definitions used by these constraints are provided
in :ref:`notation.notation`.

Carrier emissions
^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.CarbonEmissionsCarrierConstraint.build
   :sections: summary, formulation

Total annual carrier emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.CarbonEmissionsCarrierTotalConstraint.build
   :sections: summary, formulation

Total annual technology emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.CarbonEmissionsTechnologyTotalConstraint.build
   :sections: summary, formulation

Total annual system emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsAnnualConstraint.build
   :sections: summary, formulation

Cumulative emissions
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsCumulativeConstraint.build
   :sections: summary, formulation

Annual emissions limit
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsAnnualLimitConstraint.build
   :sections: summary, formulation

Annual-limit overshoot
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsAnnualOvershootConstraint.build
   :sections: summary, formulation

Cumulative emissions budget
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsBudgetConstraint.build
   :sections: summary, formulation

Budget overshoot
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsBudgetOvershootConstraint.build
   :sections: summary, formulation


Operational technology constraints
----------------------------------

Conversion, storage, transport, and retrofit technology notation is defined in
the corresponding sections of :ref:`notation.notation`.

Conversion capacity factor
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.CapacityFactorConversionConstraint.build
   :sections: summary, formulation

Carrier conversion
^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.CarrierConversionConstraint.build
   :sections: summary, formulation

Minimum full-load hours
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.MinimumFullLoadHoursConstraint.build
   :sections: summary, formulation

Storage capacity factor
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CapacityFactorStorageConstraint.build
   :sections: summary, formulation

Storage-level coupling
^^^^^^^^^^^^^^^^^^^^^^

The storage-level formulation maps storage time steps to operational time
steps. The canonical storage variables and temporal parameters are listed in
:ref:`notation.notation`.

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CoupleStorageLevelConstraint.build
   :sections: summary, formulation

Maximum storage level
^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageLevelMaxConstraint.build
   :sections: summary, formulation

Storage spillage
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.FlowStorageSpillageConstraint.build
   :sections: summary, formulation

Mutually exclusive charging and discharging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.ChargeDischargeBinaryConstraint.build
   :sections: summary, formulation

Transport capacity factor
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.transport_technology.constraints.CapacityFactorTransportConstraint.build
   :sections: summary, formulation

Transport losses
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.transport_technology.constraints.TransportTechnologyLossesFlowConstraint.build
   :sections: summary, formulation

Retrofit flow coupling
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.retrofitting_technology.constraints.RetrofitFlowCouplingConstraint.build
   :sections: summary, formulation

Technology on/off operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This generic constraint applies the appropriate activity-flow expression for
each technology type.

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyOnOffConstraint.build
   :sections: summary, formulation


Investment and capacity constraints
-----------------------------------

These constraints determine when capacity becomes available, how long it
remains active, and which bounds apply to additions and total installed
capacity. General technology notation is defined in
:ref:`notation.notation`.

Technology lifetime
^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyLifetimeConstraint.build
   :sections: summary, formulation

Capacity upper limit
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyCapacityLimitConstraint.build
   :sections: summary, formulation

Capacity lower limit
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyCapacityLowerLimitConstraint.build
   :sections: summary, formulation

Minimum capacity addition
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyMinCapacityAdditionConstraint.build
   :sections: summary, formulation

Maximum capacity addition
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyMaxCapacityAdditionConstraint.build
   :sections: summary, formulation

Construction time
^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyConstructionTimeConstraint.build
   :sections: summary, formulation

Technology diffusion
^^^^^^^^^^^^^^^^^^^^

The diffusion limit controls how quickly a technology can be deployed. It links
the permitted capacity addition to the accumulated installation knowledge from
earlier additions and existing capacity. Older knowledge is depreciated over
time. It is active when the configured maximum diffusion rate is finite. This
approach is based on
`Leibowicz et al. (2016)
<https://www.sciencedirect.com/science/article/pii/S0040162515001675>`_.

The unbounded market share (``market_share_unbounded``), denoted by
:math:`\chi` in the equations, is a separate mechanism. It provides an
additional capacity-addition allowance based on the capacity available before
the current addition for other technologies in the same technology class
(excluding the target technology) with the same reference carrier. The
allowance is calculated at the same location as the target technology. It is
not knowledge spillover and does not use :math:`\omega`.

Knowledge spillover means that experience gained by installing a technology at
one location can also support its deployment at other locations. For finite
spillover :math:`\omega`, a location-wise limit is imposed at every applicable
location. Its permitted addition is based on the knowledge at that location
plus :math:`\omega` times the knowledge at the other nodes. This limits the
pace of deployment at each location. Transport technologies are indexed by
edges and are excluded from node-to-node spillover.

A global limit is imposed in addition. It constrains total additions using the
no-spillover knowledge stock summed over all locations, while allowing those
additions to be distributed across locations. If spillover is infinite
(:math:`\omega=\infty`), only this global limit is created, corresponding to
knowledge being freely transferable between locations. For storage
technologies, the diffusion limits are applied independently to power and
energy capacity.

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyDiffusionLimitConstraint.build
   :sections: summary, formulation

Linear conversion CAPEX
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.LinearCapexConstraint.build
   :sections: summary, formulation

Storage energy-to-power ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CapacityEnergyToPowerRatioConstraint.build
   :sections: summary, formulation

Storage CAPEX
^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageTechnologyCapexConstraint.build
   :sections: summary, formulation

Transport CAPEX
^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.transport_technology.constraints.TransportTechnologyCapexConstraint.build
   :sections: summary, formulation
