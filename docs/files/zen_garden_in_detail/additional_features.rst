.. _additional_features.additional_features:

###################
Additional Features
###################
Besides the main features of ZEN-garden, there are some additional features that 
can be used to enhance the user experience:

1. :ref:`additional_features.milp`
2. :ref:`additional_features.construction_times`
3. :ref:`additional_feaures.year_specific_input_data`
4. :ref:`additional_features.multi_year_storage_periodicity`
5. :ref:`additional_features.technology_diffusion`
6. :ref:`additional_features.retrofitting_technologies`
7. :ref:`additional_features.fuel_replacement`
8. :ref:`additional_features.availability_yearly`
9. :ref:`additional_features.modeling_carbon_emissions`
10. :ref:`additional_features.demand_shedding`



.. _additional_features.milp:

Additional mixed-integer constraints
------------------------------------

Besides, PWA representation of the CAPEX (see :ref:`input_handling.pwa`), ZEN-garden allows the 
use of two additional mixed-integer linear constraints:


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
values of a conversion technology by a set of linear functions. :ref:`input_handling.pwa` 
provides detailed description on how to use the piecewise affine representation.
Moreover, :ref:`math_formulation.pwa_constraints` outlines the mathematical constraints that 
are added.


.. _additional_features.construction_times:

Construction times
---------------------------------


.. _additional_feaures.year_specific_input_data:

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
:ref:`input_handling.overwrite_defaults` section.

**Year-specific time series aggregation**

Furthermore, if time series aggregation is activated (see :ref:`tsa.using_the_tsa`), 
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


.. _additional_features.technology_diffusion:

Technology diffusion
---------------------------------


.. _additional_features.retrofitting_technologies:

Retrofitting technologies
---------------------------------


.. _additional_features.fuel_replacement:

Fuel replacement
---------------------------------


.. _additional_features.availability_yearly:

Availability yearly
---------------------------------


.. _additional_features.modeling_carbon_emissions:

Carbon emission constraints
---------------------------------


.. _additional_features.demand_shedding:

Demand shedding
---------------------------------