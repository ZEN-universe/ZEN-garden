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

   * constraint ``get_constraint`` docstrings are the source of truth for equations and
     implementation-specific conditions. Do not write equations into this file.
     If an equation changes, its docstring should be updated in the same change.
   * The objective function definitions are documented here because they are implemented
     as objective methods rather than constraint ``get_constraint`` methods.
   * :ref:`notation.notation` is the source of truth for sets, parameters,
     variables, symbols, time-step types, descriptions, and units. If a
     symbol or component description changes, its entry in the notation tables
     should be updated instead.

.. _mathematical_formulation.objectives:

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

.. _mathematical_formulation.emissions_objective:

Minimizing total emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^

The cumulative carbon emissions at the end of the planning horizon are
minimized:

.. math::
    :label: min_emissions_new

    \mathrm{min} \quad M^{\mathrm{cum}}_Y



Total annual system cost
^^^^^^^^^^^^^^^^^^^^^^^^

The annual total cost combines capital expenditure, technology operating
expenditure, carrier import/export and demand-shedding costs, and
carbon-emissions costs.

.. docstring_method:: zen_garden.elements.energy_system.constraints.CostTotalConstraint.get_constraint

Net present cost
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.NetPresentCostConstraint.get_constraint

Here, :math:`y` indexes planning periods rather than calendar years. The
uppercase :math:`\Delta y` is the number of calendar years between planning
periods. The lowercase :math:`\delta_y` is the number of annual cost terms
assigned to period :math:`y`: it equals :math:`\Delta y` for ordinary periods
and one for the final period. Costs are therefore discounted for every year in
each interval, including years without an optimization decision, while the
final period is counted once.

Annualized technology CAPEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Investment costs are converted to annual cash flows with an annuity factor
based on the discount rate and depreciation time. These cash flows cover both
new capacity investments and the remaining CAPEX of existing capacity.

CAPEX symbols and parameters are defined in :ref:`notation.notation`.

.. docstring_method:: zen_garden.elements.technology.constraints.CostCapexYearlyConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CostCapexYearlyEnergyConstraint.get_constraint

Total annual technology CAPEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.CostCapexYearlyTotalConstraint.get_constraint

Annual technology OPEX
^^^^^^^^^^^^^^^^^^^^^^

Annual technology OPEX combines variable costs that depend on carrier flow
with fixed costs proportional to installed capacity; storage technologies also
have an energy-capacity term.

.. docstring_method:: zen_garden.elements.technology.constraints.CostOpexYearlyConstraint.get_constraint

Total annual technology OPEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.CostOpexYearlyTotalConstraint.get_constraint

Carrier import and export cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Carrier parameters and variables are catalogued in :ref:`notation.notation`.

.. docstring_method:: zen_garden.elements.carrier.constraints.CostCarrierConstraint.get_constraint

Demand-shedding cost and limit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.CostShedDemandConstraint.get_constraint

.. docstring_method:: zen_garden.elements.carrier.constraints.LimitShedDemandConstraint.get_constraint

Total annual carrier cost
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.CostCarrierTotalConstraint.get_constraint

Carbon-emissions cost
^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CostCarbonEmissionsTotalConstraint.get_constraint


Technology operating costs and emissions
----------------------------------------

The following constraints calculate time-dependent variable OPEX and operating
emissions. Their annual aggregation is documented in the preceding section and
in :ref:`model_formulation_docstrings.emissions`.

Conversion technologies
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.OpexTechnologyConversionConstraint.get_constraint

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.CarbonEmissionsTechnologyConversionConstraint.get_constraint

Storage technologies
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.OpexTechnologyStorageConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CarbonEmissionsTechnologyStorageConstraint.get_constraint

Transport technologies
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.transport_technology.constraints.OpexTechnologyTransportConstraint.get_constraint

.. docstring_method:: zen_garden.elements.transport_technology.constraints.CarbonEmissionsTechnologyTransportConstraint.get_constraint


Carrier balance and external exchange
-------------------------------------

The energy balance constraint ensures a correct balance of carrier flows at
each node and time step. Carrier sources are conversion outputs, incoming
transport flows after losses, storage discharge, and imports. Carrier sinks are
served demand, conversion inputs, outgoing transport flows, storage charge, and
exports; shed demand reduces served demand. Their notation is defined in
:ref:`notation.notation`.

.. _mathematical_formulation.nodal_carrier_balance:

Nodal carrier balance
^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.NodalEnergyBalanceConstraint.get_constraint

Time-dependent import and export availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting an import or export availability to infinity removes the corresponding
availability limit.

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityImportConstraint.get_constraint

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityExportConstraint.get_constraint

Annual import and export availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityImportYearlyConstraint.get_constraint

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityExportYearlyConstraint.get_constraint


.. _model_formulation_docstrings.emissions:
.. _mathematical_formulation.emissions_constraints:

Emissions accounting and limits
-------------------------------

The parameter and variable definitions used by these constraints are provided
in :ref:`notation.notation`.

Carrier emissions
^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.CarbonEmissionsCarrierConstraint.get_constraint

Total annual carrier emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.CarbonEmissionsCarrierTotalConstraint.get_constraint

Total annual technology emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.CarbonEmissionsTechnologyTotalConstraint.get_constraint

Total annual system emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsAnnualConstraint.get_constraint

Cumulative emissions
^^^^^^^^^^^^^^^^^^^^

Cumulative emissions are attributed to the end of each year. They include the
initial value in the first planning period and are carried forward in
subsequent periods.

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsCumulativeConstraint.get_constraint

Annual emissions limit
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsAnnualLimitConstraint.get_constraint

Annual-limit overshoot
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsAnnualOvershootConstraint.get_constraint

Cumulative emissions budget
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsBudgetConstraint.get_constraint

Budget overshoot
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsBudgetOvershootConstraint.get_constraint


Operational technology constraints
----------------------------------

Conversion, storage, transport, and retrofit technology notation is defined in
the corresponding sections of :ref:`notation.notation`.

Conversion capacity factor
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.CapacityFactorConversionConstraint.get_constraint

Carrier conversion
^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.CarrierConversionConstraint.get_constraint

Minimum full-load hours
^^^^^^^^^^^^^^^^^^^^^^^

This constraint is currently available only for conversion technologies.

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.MinimumFullLoadHoursConstraint.get_constraint

Storage capacity factor
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CapacityFactorStorageConstraint.get_constraint

Storage-level coupling
^^^^^^^^^^^^^^^^^^^^^^

When ``storage_periodicity`` is enabled, the first storage level of a period is
coupled to the last level, closing the storage balance across the period
boundary.

The storage-level formulation maps storage time steps to operational time
steps. The canonical storage variables and temporal parameters are listed in
:ref:`notation.notation`.

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CoupleStorageLevelConstraint.get_constraint

Maximum storage level
^^^^^^^^^^^^^^^^^^^^^

Because storage flows are constant within an aggregated time step, the storage
level cannot have an unmodeled peak between its start and end. Enforcing the
capacity limit at both endpoints therefore also enforces it throughout the
step.

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageLevelMaxConstraint.get_constraint

Storage spillage
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.FlowStorageSpillageConstraint.get_constraint

Mutually exclusive charging and discharging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.ChargeStorageBinaryConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.DischargeStorageBinaryConstraint.get_constraint

Transport capacity factor
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.transport_technology.constraints.CapacityFactorTransportConstraint.get_constraint

Transport losses
^^^^^^^^^^^^^^^^

Transport losses can be modeled with either a linear or an exponential loss
factor based on transport distance.

.. docstring_method:: zen_garden.elements.transport_technology.constraints.TransportTechnologyLossesFlowConstraint.get_constraint

Retrofit flow coupling
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.retrofitting_technology.constraints.RetrofitFlowCouplingConstraint.get_constraint

.. _mathematical_formulation.technology_on_off:

Technology on/off operation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This generic constraint applies the appropriate activity-flow expression for
each technology type.

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyOnOffCapacityHelperConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyOnOffCapacityHelperLowerBoundConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyOnOffCapacityHelperUpperBoundConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyOnOffOperationLowerBoundConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyOnOffOperationUpperBoundConstraint.get_constraint


Investment and capacity constraints
-----------------------------------

These constraints determine when capacity becomes available, how long it
remains active, and which bounds apply to additions and total installed
capacity. General technology notation is defined in
:ref:`notation.notation`.

Technology lifetime
^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyLifetimeConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyLifetimePreviousConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyLifetimeConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyLifetimePreviousConstraint.get_constraint

Capacity upper limit
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyCapacityLimitReachedConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyCapacityLimitNotReachedConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyCapacityLimitReachedConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyCapacityLimitNotReachedConstraint.get_constraint

Capacity lower limit
^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyCapacityLowerLimitConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyCapacityLowerLimitConstraint.get_constraint

.. _mathematical_formulation.minimum_capacity_addition:

Minimum capacity addition
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyMinCapacityAdditionConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyMinCapacityAdditionConstraint.get_constraint

Maximum capacity addition
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyMaxCapacityAdditionConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyMaxCapacityAdditionConstraint.get_constraint

.. _mathematical_formulation.construction_time:

Construction time
^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyConstructionTimeConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyConstructionTimeOutsideConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyConstructionTimeConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyConstructionTimeOutsideConstraint.get_constraint

.. _mathematical_formulation.technology_diffusion:

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

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyDiffusionLimitTotalConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyDiffusionLimitLocationConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyDiffusionLimitTotalConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyDiffusionLimitLocationConstraint.get_constraint

Linear conversion CAPEX
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.LinearCapexConstraint.get_constraint

Storage energy-to-power ratio
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CapacityEnergyToPowerRatioMinConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CapacityEnergyToPowerRatioMaxConstraint.get_constraint

Storage CAPEX
^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageTechnologyCapexConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageTechnologyCapexEnergyConstraint.get_constraint

Transport CAPEX
^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.transport_technology.constraints.TransportTechnologyCapexConstraint.get_constraint
