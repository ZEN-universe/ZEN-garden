##############
Additional Features
##############
Besides the main features of ZEN garden, there are some additional features that can be used to enhance the user experience:

1. :ref:`MILP`


.. _MILP:
Additional mixed-integer constraints
---------------------------------
Besides, PWA representation of the CAPEX (see :ref:`PWA`), ZEN-garden allows the use of two additional mixed-integer linear constraints:

**Minimum load**

If the user sets the parameter ``min_load`` to anything other than zero, ZEN-garden will add a mixed-integer linear constraint that ensures that the output of a technology is above minimum load when turned on, otherwise it is zero.

**Technology installation**

If the user sets the parameter ``capacity_addition_min`` to anything other than zero, ZEN-garden will add a mixed-integer linear constraint that ensures that the capacity addition of a technology is above the minimum capacity addition, otherwise it is zero.
The associated binary variable ``technology_installation`` is 1 if the technology is installed and 0 otherwise.

``technology_installation`` is also used in determining the CAPEX of transport technologies, which depend both on the distance between nodes and the quantity of the transported good.
The parameter ``capex_specific_transport`` is the CAPEX per unit of transported good, whereas ``capex_per_distance_transport`` is the CAPEX per unit of distance.
If both parameters are set, ZEN-garden will add a mixed-integer linear constraint where the installation, i.e., the use of a certain edge, itself already accrues a cost, and then on top of that, the quantity cost is added.
Note that ``capex_specific_transport`` can vary with the length of an edge. In particular, if the user only specifies ``capex_per_distance_transport``, then ZEN-garden multiplies ``capex_per_distance_transport`` with the length of the edge to get the CAPEX per unit of transported good.
This is the most commonly used case, but it does not account for the fact that there might be an initial investment purely from the installation before adding the cost for the size of the capacity.

