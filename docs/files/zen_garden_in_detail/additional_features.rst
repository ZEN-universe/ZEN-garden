##############
Additional Features
##############
Besides the main features of ZEN-garden, there are some additional features that can be used to enhance the user experience:

1. :ref:`MILP`
2. :ref:`construction_times`
3. :ref:`technology_diffusion`
4. :ref:`retrofitting_technologies`
5. :ref:`fuel_replacement`
6. :ref:`availability_yearly`
7. :ref:`modeling_carbon_emissions`
8. :ref:`demand_shedding`



.. _MILP:
Additional mixed-integer constraints
---------------------------------
Besides, PWA representation of the CAPEX (see :ref:`PWA`), ZEN-garden allows the use of two additional mixed-integer linear constraints:

.. _min_load:
**Minimum load**

If the user sets the parameter ``min_load`` to anything other than zero, ZEN-garden will add a mixed-integer linear constraint that ensures that the output of a technology is above minimum load when turned on, otherwise it is zero. The constraints are described in :ref:`min_load_constraints`.

.. _min_capacity_addition:
**Technology installation**

If the user sets the parameter ``capacity_addition_min`` to anything other than zero, ZEN-garden will add a mixed-integer linear constraint that ensures that the capacity addition of a technology is above the minimum capacity addition, otherwise it is zero.
The associated binary variable ``technology_installation`` is 1 if the technology is installed and 0 otherwise. The constraints are described in :ref:`min_capacity_installation`.

``technology_installation`` is also used in determining the CAPEX of transport technologies, which depend both on the distance between nodes and the quantity of the transported good.
The parameter ``capex_specific_transport`` is the CAPEX per unit of transported good, whereas ``capex_per_distance_transport`` is the CAPEX per unit of distance.
If both parameters are set, ZEN-garden will add a mixed-integer linear constraint where the installation, i.e., the use of a certain edge, itself already accrues a cost, and then on top of that, the quantity cost is added.
Note that ``capex_specific_transport`` can vary with the length of an edge. In particular, if the user only specifies ``capex_per_distance_transport``, then ZEN-garden multiplies ``capex_per_distance_transport`` with the length of the edge to get the CAPEX per unit of transported good.
This is the most commonly used case, but it does not account for the fact that there might be an initial investment purely from the installation before adding the cost for the size of the capacity.

.. _pwa_conversion_technologies:
**Piecewise affine linearization of the CAPEX of conversion technologies**

The user can specify a ``nonlinear_capex.csv`` file to approximate the CAPEX values of a conversion technology by a set of linear functions. :ref:`PWA` provides detailed description on how to use the piecewise affine representation.
Moreover, :ref:`PWA_constraints` outlines the mathematical constraints that are added.

.. _construction_times:
Construction times
---------------------------------


.. _distance_dependent_transport_capex:
Distance-dependent capital investment cost for transport technologies
---------------------------------

The capital investment cost for transport technologies can be determined based on a distance independent cost term :math:`\alpha^\mathrm{const}_{j,y}`, and a distance dependent cost term :math:`\alpha^\mathrm{dist}_{j,e,y}`. The distance independent cost term is multiplied by the capacity of the transport technology, whereas the distance dependent cost term is multiplied by the distance between the nodes. The investment decision is modeled with the binary variable :math:`g_{h,p,y}`. The binary variable :math:`g_{h,p,y}` equals 1 if the transport technology is installed and 0 otherwise.

.. math::
    :label: cost_capex_transport

    I_{j,e,y} = \alpha^\mathrm{const}_{j,y} \Delta S_{j,e,y} + alpha^\mathrm{dist}_{j,e,y} h_{j,e} q_{j,e,y}

.. _technology_diffusion:
Technology diffusion
---------------------------------


.. _retrofitting_technologies:
Retrofitting technologies
---------------------------------


.. _fuel_replacement:
Fuel replacement
---------------------------------


.. _availability_yearly:
Availability yearly
---------------------------------


.. _modeling_carbon_emissions:
Carbon emission constraints
---------------------------------


.. _demand_shedding:
Demand shedding
---------------------------------