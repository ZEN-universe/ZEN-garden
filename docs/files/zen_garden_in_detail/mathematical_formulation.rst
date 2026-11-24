.. _mathematical_formulation.mathematical_formulation:

Mathematical formulation
========================

ZEN-garden formulates energy-system design and operation as a mixed-integer
linear program (MILP), or as a linear program (LP) when no binary decisions are
required. The following
sections describe the objectives and constraints that define the optimization
problem. For a complete list of symbols, see :ref:`notation.notation`. Reusable
calculations are documented separately in :ref:`api_reference.expressions`.

.. note::

   Important notes on the development and maintenance of this formulation:

   * Constraint ``get_constraint`` docstrings are the source of truth for equations and
     implementation-specific conditions. They expand reused expressions into variables
     and parameters. Each constraint is rendered under its declared name using only the
     summary and formulation sections. Do not write constraint equations into this file.
     If an equation changes, its docstring should be updated in the same change.
   * Objective formulas come from the selected expression's declarative ``doc`` field.
     Update that field when an objective expression changes.
   * :ref:`notation.notation` is the source of truth for sets, parameters,
     variables, symbols, time-step types, descriptions, and units. If a
     symbol or component description changes, its entry in the notation tables
     should be updated instead.

.. _mathematical_formulation.objectives:

Objectives
----------

Total discounted system cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default ``total_cost`` objective minimizes the complete discounted system
cost over all modeled years.

.. component_documentation:: zen_garden.elements.energy_system.expressions.TotalCost
   :include-summary:

Total carbon emissions
^^^^^^^^^^^^^^^^^^^^^^

The previously supported emissions objective, ``total_carbon_emissions``,
selects cumulative emissions at the final modeled year.

.. component_documentation:: zen_garden.elements.energy_system.expressions.TotalCarbonEmissions
   :include-summary:

Other objectives
^^^^^^^^^^^^^^^^

Any other registered scalar expression can also be selected through
``analysis.objective``. ``analysis.sense`` selects minimization or
maximization. ZEN-garden does not sum or otherwise reduce the selected
expression automatically, so a user-defined objective must include any
required sum or selection explicitly. The available expressions are listed in
:ref:`api_reference.expressions`.


Carrier balance and external exchange
-------------------------------------

The energy balance constraint ensures a correct balance of carrier flows at
each node and time step. Carrier sources are conversion outputs, incoming
transport flows after losses, storage discharge, and imports. Carrier sinks are
served demand, conversion inputs, outgoing transport flows, storage charge, and
exports; shed demand reduces served demand. Their notation is defined in
:ref:`notation.notation`.

.. _mathematical_formulation.nodal_carrier_balance:

.. docstring_method:: zen_garden.elements.carrier.constraints.NodalEnergyBalanceConstraint.get_constraint

Setting an import or export availability to infinity removes the corresponding
availability limit.

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityImportConstraint.get_constraint

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityExportConstraint.get_constraint

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityImportYearlyConstraint.get_constraint

.. docstring_method:: zen_garden.elements.carrier.constraints.AvailabilityExportYearlyConstraint.get_constraint

.. docstring_method:: zen_garden.elements.carrier.constraints.LimitShedDemandConstraint.get_constraint


.. _model_formulation_docstrings.emissions:
.. _mathematical_formulation.emissions_constraints:

Emissions accounting and limits
-------------------------------

The parameter and variable definitions used by these constraints are provided
in :ref:`notation.notation`. The emissions calculations used by the limits are
documented in :ref:`api_reference.expressions`.

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsAnnualLimitConstraint.get_constraint

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsAnnualOvershootConstraint.get_constraint

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsBudgetConstraint.get_constraint

.. docstring_method:: zen_garden.elements.energy_system.constraints.CarbonEmissionsBudgetOvershootConstraint.get_constraint


Operational technology constraints
----------------------------------

Conversion, storage, transport, and retrofit technology notation is defined in
the corresponding sections of :ref:`notation.notation`.

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.CapacityFactorConversionConstraint.get_constraint

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.CarrierConversionConstraint.get_constraint

This constraint is currently available only for conversion technologies.

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.MinimumFullLoadHoursConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CapacityFactorStorageConstraint.get_constraint

When ``storage_periodicity`` is enabled, the first storage level of a period is
coupled to the last level, closing the storage balance across the period
boundary.

The storage-level formulation maps storage time steps to operational time
steps. The canonical storage variables and temporal parameters are listed in
:ref:`notation.notation`.

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CoupleStorageLevelConstraint.get_constraint

Because storage flows are constant within an aggregated time step, the storage
level cannot have an unmodeled peak between its start and end. Enforcing the
capacity limit at both endpoints therefore also enforces it throughout the
step.

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageLevelMaxConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.FlowStorageSpillageConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.ChargeStorageBinaryConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.DischargeStorageBinaryConstraint.get_constraint

.. docstring_method:: zen_garden.elements.transport_technology.constraints.CapacityFactorTransportConstraint.get_constraint

.. docstring_method:: zen_garden.elements.retrofitting_technology.constraints.RetrofitFlowCouplingConstraint.get_constraint

This generic constraint applies the appropriate activity-flow expression for
each technology type.

.. _mathematical_formulation.technology_on_off:

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

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyLifetimeConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyLifetimePreviousConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyLifetimeConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyLifetimePreviousConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyCapacityLimitReachedConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyCapacityLimitNotReachedConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyCapacityLimitReachedConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyCapacityLimitNotReachedConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyCapacityLowerLimitConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyCapacityLowerLimitConstraint.get_constraint

.. _mathematical_formulation.minimum_capacity_addition:

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyMinCapacityAdditionConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyMinCapacityAdditionConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyMaxCapacityAdditionConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyMaxCapacityAdditionConstraint.get_constraint

.. _mathematical_formulation.construction_time:

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyConstructionTimeConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyConstructionTimeOutsideConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyConstructionTimeConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyConstructionTimeOutsideConstraint.get_constraint

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

.. _mathematical_formulation.technology_diffusion:

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyDiffusionLimitTotalConstraint.get_constraint

.. docstring_method:: zen_garden.elements.technology.constraints.TechnologyDiffusionLimitLocationConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyDiffusionLimitTotalConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.StorageEnergyDiffusionLimitLocationConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CapacityEnergyToPowerRatioMinConstraint.get_constraint

.. docstring_method:: zen_garden.elements.storage_technology.constraints.CapacityEnergyToPowerRatioMaxConstraint.get_constraint
