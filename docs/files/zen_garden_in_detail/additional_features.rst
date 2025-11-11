.. _additional_features.additional_features:

###################
Additional Features
###################
Besides the main features of ZEN-garden, there are some additional features that 
can be used to enhance the user experience:

1. :ref:`additional_features.milp`
2. :ref:`additional_features.year_specific_input_data`
3. :ref:`additional_features.multi_year_storage_periodicity`
4. :ref:`additional_features.construction_times`
5. :ref:`additional_features.technology_diffusion`
6. :ref:`additional_features.retrofitting_technologies`
7. :ref:`additional_features.availability_yearly`
8. :ref:`additional_features.modeling_carbon_emissions`
9. :ref:`additional_features.demand_shedding`

.. _additional_features.milp:

Additional mixed-integer constraints
------------------------------------

Besides, PWA representation of the CAPEX (see :ref:`input_structure.pwa`), ZEN-garden allows the 
use of three additional mixed-integer linear constraints:


.. _additional_features.min_load:

**Minimum load**

If the user sets the parameter ``min_load`` to anything other than zero, 
ZEN-garden will add a mixed-integer linear constraint that ensures that the 
output of a technology is above minimum load when turned on, otherwise it is 
zero. The constraints are described in :ref:`math_formulation.min_load_constraints`.


.. _additional_features.min_capacity_addition:

**Technology installation**

If the user sets the parameter ``capacity_addition_min`` to anything other than 
zero, ZEN-garden will add a mixed-integer linear constraint that ensures that 
the capacity addition of a technology is above the minimum capacity addition, 
otherwise it is zero. The associated binary variable ``technology_installation`` 
is 1 if the technology is installed and 0 otherwise. The constraints are 
described in :ref:`math_formulation.min_capacity_installation`.

``technology_installation`` is also used in determining the CAPEX of transport 
technologies, which depend both on the distance between nodes and the quantity 
of the transported good. The parameter ``capex_specific_transport`` is the CAPEX 
per unit of transported good, whereas ``capex_per_distance_transport`` is the 
CAPEX per unit of distance. If both parameters are set, ZEN-garden will add a 
mixed-integer linear constraint where the installation, i.e., the use of a 
certain edge, itself already accrues a cost, and then on top of that, the 
quantity cost is added. Note that ``capex_specific_transport`` can vary with the 
length of an edge. In particular, if the user only specifies 
``capex_per_distance_transport``, then ZEN-garden multiplies 
``capex_per_distance_transport`` with the length of the edge to get the CAPEX 
per unit of transported good. This is the most commonly used case, but it does 
not account for the fact that there might be an initial investment purely from 
the installation before adding the cost for the size of the capacity.


.. _additional_features.pwa_conversion_technologies:

**Piecewise affine linearization of the CAPEX of conversion technologies**

The user can specify a ``nonlinear_capex.csv`` file to approximate the CAPEX 
values of a conversion technology by a set of linear functions. :ref:`input_structure.pwa` 
provides detailed description on how to use the piecewise affine representation.
Moreover, :ref:`math_formulation.pwa_constraints` outlines the mathematical constraints that 
are added.


.. _additional_features.year_specific_input_data:

Year-specific hourly time series
---------------------------------

ZEN-garden allows to provide input hourly time series that are specific to a 
certain year. It overwrites the default values or the given csv file for the 
respective year. This can be useful for example to model dark doldrums in the 
electricity sector, where the solar and wind generation is lower than usual.
The additional csv file can be given as a ``<parameter_name>_<year>.csv`` file, 
where ``<parameter_name>`` is the name of the parameter whose default values
should be overwritten and ``<year>`` is the year for which the data is specific 
(e.g. ``demand_2023.csv`` if we want to provide demand data for the year 2023).
Note that the input data structure stays the same as described in the 
:ref:`input_structure.overwrite_defaults` section.

**Year-specific time series aggregation**

Furthermore, if time series aggregation is activated (see :ref:`t_tsa.using_the_tsa`), 
for each year with year-specific input data, the time series aggregation is 
performed separately.


.. _additional_features.multi_year_storage_periodicity:

Multi-year storage periodicity
---------------------------------

The user can choose to enable storage level periodicity over multiple years, 
instead of just within single years (see ``storage_periodicity`` and 
``multiyear_periodicity`` in :ref:`configuration.system`). This can be very useful when 
modeling inter-annual variability. For instance, years with high natural gas 
supply can be followed by years with low availability, where a storage can be 
filled in the high supply years and used in the low supply years.
To use this feature, the user has to set ``multiyear_periodicity`` to ``TRUE`` 
in the ``system.json`` file (see :ref:`configuration.system`). The multiyear periodicity 
enforces the storage level at the beginning of the planning horizon to be equal 
to the storage level at the end of the planning horizon. Note that as of now the 
multi-year periodicity is only usable if the interval between years of the 
planning horizon is one year, i.e. the parameter ``interval_between_years`` in 
``system.json`` is set to 1.


.. _additional_features.distance_dependent_transport_capex:

Distance-dependent capital investment cost for transport technologies
---------------------------------------------------------------------

The capital investment cost for transport technologies can be determined based 
on a distance independent cost term :math:`\alpha^\mathrm{const}_{j,y}`, and a 
distance dependent cost term :math:`\alpha^\mathrm{dist}_{j,e,y}`. The distance 
independent cost term is multiplied by the capacity of the transport technology, 
whereas the distance dependent cost term is multiplied by the distance between 
the nodes. The investment decision is modeled with the binary variable 
:math:`g_{h,p,y}`. The binary variable :math:`g_{h,p,y}` equals 1 if the 
transport technology is installed and 0 otherwise.

.. math::
    :label: cost_capex_transport_dist_dependent

    I_{j,e,y} = \alpha^\mathrm{const}_{j,y} \Delta S_{j,e,y} + 
    \alpha^\mathrm{dist}_{j,e,y} h_{j,e} q_{j,e,y}


.. _additional_features.construction_times:

Construction times
---------------------------------

The user can specify construction times for technologies in ZEN-garden
(:math:`dy^\mathrm{construction}` :eq:`construction_time`). The construction time is the time between the investment
decision and the availability of the new capacity.

Note that as of now, no costs are incurred during the construction time.

.. _additional_features.technology_diffusion:

Technology diffusion
---------------------------------

ZEN-garden allows for endogenously constraining the annual capacity as a function of past capacity additions.
The capacity additions are depreciated over time to reflect knowledge depreciation.
An example for knowledge depreciation is the loss of skilled personnel and engineering firms
over time if a technology is not continuously deployed.
The equations are detailed in :eq:`constrained_technology_deployment_i`, :eq:`constrained_technology_deployment_k`,
and :eq:`constrained_technology_deployment_j`.

The user can set five parameters:

1. The ``max_diffusion_rate`` (:math:`\vartheta_i`, indexed by each technology):
The ``max_diffusion_rate`` limits the maximum annual capacity addition as a fraction of the existing knowledge.
Since the maximum capacity addition is proportional to the existing capacity, this constraint is linear
in the capacity but results in an exponential capacity growth. Therefore, it describes the exponential growth phase
of the logistic S-curve of technology diffusion.

2. The ``knowledge_spillover_rate`` (:math:`\omega`):
The knowledge spillover rate allows for learning effects from other nodes. A value of 0.05 means that
5% of the knowledge from other nodes is added to the local knowledge stock. If setting the spillover rate to ``inf``,
perfect spillover is assumed and only the global capacity additions are constrained by the global knowledge stock.

3. The ``market_share_unbounded`` (:math:`\xi`):
The unbounded market share allows for a small (we have found values of 1%-2% to be realistic)
contribution of the existing capacity of all technologies in the same sector (i.e., technologies
with the same reference carrier) to the capacity addition limit of a technology.
For example, a value of 0.01 means that every year, 1% of the existing capacity of all technologies in the same sector
can be added, even if no capacity of the considered technology exists.
(If no capacity of the considered technology exists, the knowledge stock is zero,
and thus no capacity addition would be possible otherwise.)

4. The ``capacity_addition_unbounded`` (:math:`\zeta_i`, indexed by each technology):
The unbounded capacity addition allows for a fixed amount of capacity addition each year,
regardless of the existing knowledge stock of the considered technology and all other technologies in the same sector.
This should only be used when there is no existing capacity of any technology in the same sector,
An example would be an emerging sector like carbon capture and storage.
No technology has any existing capacity, but still capacity addition should be possible
(otherwise the capacity could never be expanded).

5. The ``knowledge_depreciation_rate`` (:math:`\delta`, default value: 0.1):
The knowledge depreciation rate models the loss of knowledge over time.
A value of 0.1 means that 10% of the existing knowledge is lost each year.
So if a capacity of 1 GW is added in 2020, in 2030 only 0.9^10 * 1 GW = 0.349 GW of knowledge remains in 2030.

.. warning::

    The technology diffusion feature makes the solution and the result interpretation more complex.
    First, the feature introduces inter-year time coupling, which increases the solution time. With spillover
    effects, the coupling extends across all nodes, further increasing the solution time.
    Second, the interpretation of the results is more complex, because the interconnections between technologies,
    nodes, and years are now too complex to be interpreted in isolation. Furthermore, the technology expansion
    constraint can reduce the feasible space quite strongly and lead to undesired effects, such as prohibiting
    the uptake of certain sectors entirely.
    Therefore, we recommend to only use this feature when necessary and to carefully analyze the results.


.. _additional_features.retrofitting_technologies:

Retrofitting technologies
---------------------------------

ZEN-garden allows for modeling retrofitting technologies. A retrofitting technology behaves exactly like a normal
conversion technology, with the difference that the reference flow is linked to
the reference flow of another conversion technology. Specifically, the reference flow of the retrofitting technology
must be lower or equal to the reference flow of the converted technology times the ``retrofit_flow_coupling_factor``.
The lower-or-equal sign allows for partial retrofitting of the converted technology.

Check out the dataset example :ref:`dataset_examples.14_retrofitting_and_fuel_substitution`.

Retrofitting technologies are useful for two main applications:

1. Retrofitting existing technologies with a new technology, e.g., retrofitting existing gas turbines with CCS.
2. Fuel switching of existing technologies, e.g., converting existing coal power plants to biomass power plants.

**Retrofitting existing technologies**

The most straightforward application of retrofitting technologies is to retrofit existing conversion technologies.
As an example, we might want to retrofit an existing gas power plant with a carbon capture and storage (CCS) unit.
The gas power plant is modeled as a conversion technology, and the CCS unit is modeled as a retrofitting technology.
The gas power plant consumes gas and produces electricity, whereas the CCS unit produces (sequestered) carbon
while consuming, e.g., additional electricity for the capture process. The reference flow of the CCS unit is carbon,
so the ``retrofit_flow_coupling_factor`` is the flow of carbon per unit of electricity produced by the gas power plant.

**Fuel switching**

Less intuitively, retrofitting technologies can also be used for fuel switching of existing conversion technologies.
For example, we might want to substitute some of the natural gas input of the natural gas turbine with biogenic gas.
Instead of modeling a new conversion technology that consumes both natural gas and biogenic gas at a fixed ratio,
we can model a retrofitting technology that produces natural gas and consumes biogenic gas at a fixed ratio of 1.
The retrofitting base technology is the natural gas turbine, so the ``retrofit_flow_coupling_factor`` is the ratio
between the natural gas output of the retrofitting technology and the electricity output of the natural gas turbine.

If we did not model the fuel switching technology as a retrofitting technology but as a normal conversion technology,
all of the natural gas in the system could be substituted by biogenic gas. By coupling the fuel switching technology
to the existing natural gas turbine, the fuel substitution is limited to the use in the natural gas turbine.

.. _additional_features.availability_yearly:

Availability yearly
---------------------------------

ZEN-garden allows to specify yearly import availabilities for carriers. For example, we might want to limit the yearly
import of biomass, but not the hourly import. This can be done by specifying the parameter
``availability_import_yearly`` for the respective carrier. The same applies for exports with the parameter
``availability_export_yearly``.

Note that the ``availability_import_yearly`` and ``availability_export_yearly`` parameters are yearly parameters,
so the ``.csv`` files are indexed by the year, not the hour.

.. _additional_features.modeling_carbon_emissions:

Carbon emission modeling
---------------------------------

ZEN-garden allows for three different ways to constrain or penalize carbon emissions:

1. Setting annual carbon emission limits.
2. Setting a cumulative carbon budget for the entire planning horizon.
3. Setting a carbon price.

The three options can be used individually or in combination.

**Annual carbon emission limits**

The user can also set annual carbon emission limits by specifying the parameter
``carbon_emissions_annual_limit`` (:ref:`notation.energy_system`). The parameter is indexed by year, so a separate limit can be set for each year in
``carbon_emissions_annual_limit.csv``. The annual limits can be overshot, if
``price_carbon_emissions_annual_overshoot != inf`` (:ref:`notation.energy_system`).

**Cumulative carbon budget**

A cumulative carbon budget can be set by specifying the parameter ``carbon_emissions_budget`` (:ref:`notation.energy_system`). Note that the budget
is for the entire planning horizon, not per year, so it is sufficient to specify a single value in the
``attributes.json`` file of the energy system.
The budget can be overshot, if ``price_carbon_emissions_budget_overshoot != inf`` (:ref:`notation.energy_system`).
Using a carbon budget instead of annual limits allows the optimizer to allocate the optimal annual emission
levels over the planning horizon.

**Carbon price**

Instead of setting hard limits on carbon emissions, the user can also set a carbon price by specifying the parameter
``price_carbon_emissions`` (:ref:`notation.energy_system`). The carbon price is indexed by year, so a separate price can be set for each year in
``price_carbon_emissions.csv``. The carbon price penalizes all carbon emissions in the objective function.

.. _additional_features.demand_shedding:

Demand shedding
---------------------------------

ZEN-garden allows for demand shedding by specifying the parameter ``price_shed_demand`` (:ref:`notation.carrier`).
The shed demand acts as an additional source in the energy balance (:eq:`energy_balance`); hence, demand can be
supplied either by actual supply or by shedding demand. Shedding demand incurs a cost in the objective function
based on the ``price_shed_demand``. If ``price_shed_demand=inf``, demand shedding is disabled.

.. tip::

    If your optimization is infeasible (:ref:`t_infeasibilities.t_infeasibilities`),
    consider enabling demand shedding with a high ``price_shed_demand`` for all energy carriers. Then, the solution
    should be feasible, because at worst, the optimizer can shed all demand. After obtaining a feasible solution,
    you can analyze which carrier sheds demand to identify bottlenecks in your energy system.