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

.. docstring_method:: zen_garden.elements.energy_system.constraints.CostTotalConstraint.build
   :sections: summary, formulation

Net present cost
^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.energy_system.constraints.NetPresentCostConstraint.build
   :sections: summary, formulation

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

.. docstring_method:: zen_garden.elements.technology.constraints.CostCapexYearlyConstraint.build
   :sections: summary, formulation

Total annual technology CAPEX
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.CostCapexYearlyTotalConstraint.build
   :sections: summary, formulation

Annual technology OPEX
^^^^^^^^^^^^^^^^^^^^^^

Annual technology OPEX combines variable costs that depend on carrier flow
with fixed costs proportional to installed capacity; storage technologies also
have an energy-capacity term.

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

The energy balance constraint ensures a correct balance of carrier flows at
each node and time step. Carrier sources are conversion outputs, incoming
transport flows after losses, storage discharge, and imports. Carrier sinks are
served demand, conversion inputs, outgoing transport flows, storage charge, and
exports; shed demand reduces served demand. Their notation is defined in
:ref:`notation.notation`.

.. _mathematical_formulation.nodal_carrier_balance:

Nodal carrier balance
^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.NodalEnergyBalanceConstraint.build
   :sections: summary, formulation

Time-dependent import and export availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setting an import or export availability to infinity removes the corresponding
availability limit.

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityImportExportConstraint.build
   :sections: summary, formulation

Annual import and export availability
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityImportExportYearlyConstraint.build
   :sections: summary, formulation


.. _model_formulation_docstrings.emissions:
.. _mathematical_formulation.emissions_constraints:

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

Cumulative emissions are attributed to the end of each year. They include the
initial value in the first planning period and are carried forward in
subsequent periods.

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

This constraint is currently available only for conversion technologies.

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.MinimumFullLoadHoursConstraint.build
   :sections: summary, formulation

Storage capacity factor
^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CapacityFactorStorageConstraint.build
   :sections: summary, formulation

Storage-level coupling
^^^^^^^^^^^^^^^^^^^^^^

When ``storage_periodicity`` is enabled, the first storage level of a period is
coupled to the last level, closing the storage balance across the period
boundary.

The storage-level formulation maps storage time steps to operational time
steps. The canonical storage variables and temporal parameters are listed in
:ref:`notation.notation`.

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CoupleStorageLevelConstraint.build
   :sections: summary, formulation

Maximum storage level
^^^^^^^^^^^^^^^^^^^^^

Because storage flows are constant within an aggregated time step, the storage
level cannot have an unmodeled peak between its start and end. Enforcing the
capacity limit at both endpoints therefore also enforces it throughout the
step.

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

Transport losses can be modeled with either a linear or an exponential loss
factor based on transport distance.

.. docstring_method:: zen_garden.elements.transport_technology.constraints.TransportTechnologyLossesFlowConstraint.build
   :sections: summary, formulation

Retrofit flow coupling
^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.retrofitting_technology.constraints.RetrofitFlowCouplingConstraint.build
   :sections: summary, formulation

.. _mathematical_formulation.technology_on_off:

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

.. _mathematical_formulation.minimum_capacity_addition:

Minimum capacity addition
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyMinCapacityAdditionConstraint.build
   :sections: summary, formulation

Maximum capacity addition
^^^^^^^^^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyMaxCapacityAdditionConstraint.build
   :sections: summary, formulation

.. _mathematical_formulation.construction_time:

Construction time
^^^^^^^^^^^^^^^^^

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyConstructionTimeConstraint.build
   :sections: summary, formulation

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
