.. _math_formulation.math_formulation:

Mathematical formulation
========================

ZEN-garden optimizes the design and operation of energy system models to
investigate transition pathways towards decarbonization. The optimization
problem is in general formulated as a mixed-integer linear program (MILP), but
reduced to a linear program (LP) if the binary variables are not needed. In the
following, we provide an overview of the objective function and constraints of
the optimization problem.


.. _math_forumlation.objective:

Objective function
-------------------

Two objective functions are available:

1. minimize cumulative net present cost
2. minimize cumulative emissions

Minimizing net present cost
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The net present cost :math:`C^{\mathrm{NPC}}_y` of the energy system is minimized over the
entire planning horizon :math:`y \in {\mathcal{Y}}`.

.. math::
    :label: min_cost

    \mathrm{min} \quad \sum_{y\in\mathcal{Y}} C^{\mathrm{NPC}}_y

We define :math:`y` as a planning period rather than an actual year and
:math:`\Delta y` as the interval between planning periods. For example, if
:math:`\Delta y=2` the optimization is conducted every second year. The net present
cost :math:`C^{\mathrm{NPC}}_y` of each planning period :math:`y\in[y_0,\mathcal{Y}-1]`,
where :math:`y_0` is the first planning period, are computed by discounting the
total energy system cost of each planning period :math:`C_y^{\mathrm{total}}` with a constant
discount rate :math:`r^{\mathrm{disc}}`:

.. math::
    :label: net_present_cost_before_last_year

    C^{\mathrm{NPC}}_y = \sum_{i \in [0,\Delta y-1]} \left( \dfrac{1}{1+r^{\mathrm{disc}}} \right)^{\left(\Delta y
    (y-y_0) + i \right)} C_y^{\mathrm{total}}

Hence, we discount each year of the time horizon, also the years for which the
optimization is not conducted. Moreover, we assume that the optimization is only
conducted until the end of the first year of the last planning period. The last
period of the planning horizon :math:`Y=\max(y)` is therefore only counted as a
single year regardless of the interval between planning periods and the net
present cost :math:`C^{\mathrm{NPC}}_{\mathcal{Y}}` is defined as:

.. math::
    :label: net_present_cost_last_year

    C^{\mathrm{NPC}}_{\mathcal{Y}} = \left( \dfrac{1}{1+r^{\mathrm{disc}}} \right)^{\left(\Delta y
    (\mathcal{Y}-y_0) \right)} C_{\mathcal{Y}}

For example, suppose :math:`\Delta y=2` meaning that every planning period is 2 years
long. With an initial planning period :math:`y_0=0`, the energy system costs
:math:`C_1` occur in planning period 1, meaning in years 2 and 3. Therefore,
math:`C_1` must be discounted according to the years they are incurred, relative
to the initial time start, which are years 2 and 3.

The total cost :math:`C_y^{\mathrm{total}}` includes the annual capital expenditures
:math:`C_y^{\mathrm{cap}}` and the operational expenditures for operating technologies
:math:`C_y^{\mathrm{op}}`, importing and exporting carriers :math:`C_y^{\mathrm{carrier}}`,
and the cost of carbon emissions :math:`C_y^{\mathrm{CO_2}}`.

.. math::
    :label: npc

    C_y^{\mathrm{total}} = C_y^{\mathrm{cap}}+C_y^{\mathrm{op}}+C_y^{\mathrm{carrier}}+C_y^{\mathrm{CO_2}}


**Capital expenditures**

:math:`C_y^{\mathrm{cap}}` accounts for the annual cash flows due to capacity investments
:math:`C^{\mathrm{cap,ann}}_{h,p,y}` in technologies:

.. math::
    :label: capex_y

    C_y^{\mathrm{cap}} = \sum_{h\in\mathcal{H}}
    \sum_{p\in\mathcal{P}} C^{\mathrm{cap,ann}}_{h,p,y}

Each technology :math:`h\in\mathcal{H}` is either a conversion technology
:math:`h\in\mathcal{H}^{\mathrm{conv}}\subseteq\mathcal{H}`, a transport technology
:math:`h\in\mathcal{H}^{\mathrm{trans}}\subseteq\mathcal{H}` or a storage technology
:math:`h\in\mathcal{H}^{\mathrm{stor}}\subseteq\mathcal{H}`. **For sake of simplicity, we index
those variables and parameters that apply to all technology types with**
:math:`h`. For storage capacities, both the energy and power-rated capacity can
be expanded. Conversion and storage technologies are installed and operated on
odes :math:`n\in\mathcal{N}`. Transport technologies are installed and operated
on edges :math:`e\in\mathcal{E}`. We summarize nodes and edges to positions
:math:`p\in\mathcal{P}=\mathcal{N}\cup\mathcal{E}`.

The investment costs are annualized by multiplying the total investment cost
with the annuity factor :math:`f_h`, which is a function of the technology
depreciation time :math:`L_h^{\mathrm{dep}}` and the discount rate :math:`r^{\mathrm{disc}}`:

.. math::
    :label: annuity

    f_h=\frac{\left(1+r^{\mathrm{disc}}\right)^{L_h^{\mathrm{dep}}}r^{\mathrm{disc}}}
    {\left(1+r^{\mathrm{disc}}\right)^{L_h^{\mathrm{dep}}}-1}

The annual cash flows accrue over the technology deprecation time :math:`L_h^{\mathrm{dep}}` and
comprise the capital investment cost of newly installed and existing technology
capacities :math:`C^{\mathrm{cap,overnight}}_{h,p,y}` and :math:`i_{h,p,y}^\mathrm{ex}`. The annual
capital expenditure :math:`C^{\mathrm{cap,ann}}_{h,p,y}` for technology :math:`h\in\mathcal{H}` in
position :math:`p\in\mathcal{P}` and period :math:`y\in\mathcal{Y}` are computed
as:

.. math::
    :label: cost_capex_yearly

    C^{\mathrm{cap,ann}}_{h,p,y}= f_h\left(\left(\sum_{\tilde{y}=
    \max\left(y_0,y-\lceil\frac{L_h^{\mathrm{dep}}}{\Delta y}\rceil+1\right)}^y
    C^{\mathrm{cap,overnight}}_{h,p,\tilde{y}} \right)+
    \left(\sum_{\hat{y}=\psi \left(y-\lceil\frac{L_h^{\mathrm{dep}}}{\Delta y}\rceil+1\right)}^
    {\psi(y_0-1)} i_{h,p,y}^\mathrm{ex}\right)\right)

where :math:`\lceil\cdot\rceil` is the ceiling function and :math:`\psi(y)` is a
function that maps the planning period :math:`y` to the actual year.

.. note::
    The depreciation time :math:`L_h^{\mathrm{dep}}` is an optional parameter that reflects the time range for which technology
    investments have to be paid back. In case the depreciation time is not defined or not needed, the default value will
    be set to the technology lifetime.

The capital investment cost :math:`C^{\mathrm{cap,overnight}}_{h,p,y}` for conversion technology
:math:`h\in\mathcal{H}^{\mathrm{conv}}` is calculated as the product of the unit cost of capital
investment :math:`\kappa^{\mathrm{cap}}_{h,y}` and the capacity addition
:math:`\Delta K_{h,n,y}` on each node :math:`n\in\mathcal{N}`:

.. math::
    :label: cost_capex_conversion

    C^{\mathrm{cap,overnight}}_{h,n,y} = \kappa^{\mathrm{cap}}_{h,y} \Delta K_{h,n,y}

For existing conversion technology capacities :math:`s_{h,n,y}^{ex}` that were
installed before :math:`y_0`, we apply the unit cost of the first investment
period :math:`\kappa^{\mathrm{cap}}_{h,y_0}`:

.. math::
    :label: cost_capex_conversion_ex

    C^{\mathrm{cap,ex}}_{h,n,y} = \kappa^{\mathrm{cap}}_{h,y_0} k^{\mathrm{ex}}_{h,n,y}

For transport technologies :math:`h\in\mathcal{H}^{\mathrm{trans}}`, the unit investment cost
:math:`\kappa^{\mathrm{cap}}_{h,e,y}` can be defined 1) through a distance independent unit
cost of capital investment :math:`\kappa^{\mathrm{cap,fixed}}_{h,y}`
(:eq:`unit_cost_capex_transport_const`) or 2) a distance dependent unit cost of
capital investment :math:`\kappa^{\mathrm{cap,dist}}_{h,e,y}` which is multiplied by
the distance :math:`d^{\mathrm{dist}}_{h,e}` of the corresponding edge :math:`e\in\mathcal{E}`
(:eq:`unit_cost_capex_transport_dist`).

.. math::
    :label: unit_cost_capex_transport_const

    \kappa^{\mathrm{cap}}_{h,e,y} = \kappa^{\mathrm{cap,fixed}}_{h,y}


.. math::
    :label: unit_cost_capex_transport_dist

    \kappa^{\mathrm{cap}}_{h,e,y} = \kappa^{\mathrm{cap,dist}}_{h,e,y} d^{\mathrm{dist}}_{h,e}

.. note::
    Are both, a distance independent and a distance dependent unit cost factor
    defined, the distance dependent unit cost is used to determine the unit
    investment cost :math:`\kappa^{\mathrm{cap}}_{h,e,y}`.

The total capital investment cost :math:`C^{\mathrm{cap,ann}}_{h,p,y}` for each transport technology
:math:`h\in\mathcal{H}^{\mathrm{conv}}` is calculated as the product of the unit cost of capital
investment :math:`\kappa^{\mathrm{cap}}_{h,y}` multiplied by the capacity addition
:math:`\Delta K_{h,e,y}` on each edge :math:`e\in\mathcal{E}`:

.. math::
    :label: cost_capex_transport

    C^{\mathrm{cap,overnight}}_{h,e,y} = \kappa^{\mathrm{cap}}_{h,e,y} \Delta K_{h,e,y}

It is also possible, to apply both, a distance independent and a distance
dependent cost term by setting ``double_capex_transport=True`` in your
``system.json``. Please note that using ``double_capex_transport=True``
introduces binary variables. For more information on the distance dependent unit
cost of capital investment refer to :ref:`additional_features.distance_dependent_transport_capex`.

For existing transport technology capacities :math:`k^{\mathrm{ex}}_{h,e,y}` that were
installed before :math:`y_0`, we apply the unit cost of the first investment
period :math:`\kappa^{\mathrm{cap}}_{h,y_0}`:

.. math::
    :label: cost_capex_transport_ex

    C^{\mathrm{cap,ex}}_{h,e,y} = \kappa^{\mathrm{cap}}_{h,e,y_0} k^{\mathrm{ex}}_{h,e,y}

The total investment cost for each storage technology :math:`h\in\mathcal{H}^{\mathrm{stor}}` is
the product of the unit cost of capital investment and the capacity addition for
both the power-rated capacity (:math:`\kappa^{\mathrm{cap,power}}_{h,y}` and
:math:`\Delta K_{h,n,y}`) and the energy-rated capacity
(:math:`\kappa^{\mathrm{cap,energy}}_{h,y}` and :math:`\Delta K^{\mathrm{energy}}_{h,n,y}`).

.. math::
    :label: cost_capex_storage

    C^{\mathrm{cap,overnight}}_{h,n,y} = \kappa^{\mathrm{cap,power}}_{h,y} \Delta K_{h,n,y} + \kappa^{\mathrm{cap,energy}}_{h,y}
    \Delta K^{\mathrm{energy}}_{h,n,y}

For existing storage technology capacities :math:`k^{\mathrm{ex}}_{h,n,y}` that were installed
before :math:`y_0`, we apply the unit cost of the first investment period
:math:`\kappa^{\mathrm{cap,power}}_{h,y_0}` and :math:`\kappa^{\mathrm{cap,energy}}_{h,y_0}`:

.. math::
    :label: cost_capex_storage_ex

    C^{\mathrm{cap,ex}}_{h,n,y} = \kappa^{\mathrm{cap,power}}_{h,y_0} k^{\mathrm{ex}}_{h,n,y}

**Operational expenditures**

The annual operational expenditure for technology operation
:math:`C_y^{\mathrm{op}}` includes the variable operational costs of the
technologies :math:`C_y^{\mathrm{op,var}}` and the fixed operational expenditure
for the technology operation :math:`C_y^{\mathrm{op,fix}}`.

.. math::
    :label: opex_t

    C_y^{\mathrm{op}} = C_y^{\mathrm{op,var}} + C_y^{\mathrm{op,fix}}.

*Operational expenditures technology*

The fixed technology operational expenditures :math:`C_y^{\mathrm{op,fix}}` are the
product of the specific fixed operational expenditures :math:`\kappa^{\mathrm{op,fix}}_{h,y}` and
the capacity :math:`K_{h,p,y}`, summed over all technologies and positions
:math:`p\in\mathcal{P}`:

.. math::
    :label: opex_f

    C_y^{\mathrm{op,fix}} = \sum_{h\in\mathcal{H}}\sum_{p\in\mathcal{P}}
    \kappa^{\mathrm{op,fix}}_{h,y}K_{h,p,y}+\sum_{h\in\mathcal{H}^{\mathrm{stor}}}
    \sum_{n\in\mathcal{N}}\kappa^{\mathrm{op,fix,energy}}_{h,y}K^{\mathrm{energy}}_{h,n,y}.

The variable technology operational expenditures :math:`C_y^{\mathrm{op,var}}` are
the sum of the variable operational expenditures of each technology over the
entire year, where each timestep is multiplied by the time step duration
:math:`\Delta t_t`:

.. math::
    :label: opex_v

    C_y^{\mathrm{op,var}} = \sum_{t\in\mathcal{T}_y}\Delta t_t
    \bigg(\sum_{h\in\mathcal{H}}
    \sum_{p\in\mathcal{P}} C^{\mathrm{op,var}}_{h,p,t} \bigg).

For conversion technologies :math:`h \in \mathcal{H}^{\mathrm{conv}}`, the variable operational
expenditure are the product of the specific variable operational expenditure
:math:`\kappa^{\mathrm{op,var}}_{h,y}` and the reference flows :math:`F^{\mathrm{ref}}_{h,n,t}`:

.. math::
    :label: cost_opex_conversion

    C^{\mathrm{op,var}}_{h,n,t} = \kappa^{\mathrm{op,var}}_{h,y} F^{\mathrm{ref}}_{h,n,t}

Similarly, for transport technologies :math:`h \in \mathcal{H}^{\mathrm{trans}}`, the variable
operational expenditures are the product of the specific variable operational
expenditure :math:`\kappa^{\mathrm{op,var}}_{h,y}` and the reference flows :math:`F^{\mathrm{trans}}_{h,e,t}`:

.. math::
    :label: cost_opex_transport

    C^{\mathrm{op,var}}_{h,e,t} = \kappa^{\mathrm{op,var}}_{h,y} F^{\mathrm{trans}}_{h,e,t}

Finally, for storage technologies :math:`h \in \mathcal{H}^{\mathrm{stor}}`, the variable
operational expenditure are the product of the charge and discharge cost
:math:`\kappa^{\mathrm{op,var,ch}}_{h,y}` and :math:`\kappa^{\mathrm{op,var,dis}}_{h,y}`
multiplied by the storage charge :math:`F^{\mathrm{ch}}_{h,n,t}` and discharge
:math:`F^{\mathrm{dis}}_{h,n,t}`, respectively:

.. math::
    :label: cost_opex_storage

    C^{\mathrm{op,var}}_{h,n,t} = \kappa^{\mathrm{op,var,ch}}_{h,y} F^{\mathrm{ch}}_{h,n,t} +
    \kappa^{\mathrm{op,var,dis}}_{h,y} F^{\mathrm{dis}}_{h,n,t}

*Operational expenditures carrier*

The operational carrier cost :math:`C_y^{\mathrm{carrier}}` are the sum of the node-
and time dependent carrier cost :math:`C^{\mathrm{carrier}}_{c,n,t}` for all carriers
multiplied by the time step duration :math:`\Delta t_t`:

.. math::
    :label: opex_c

    C_y^{\mathrm{carrier}} = \sum_{c\in\mathcal{C}}\sum_{n\in\mathcal{N}}
    \sum_{t\in\mathcal{T}_y}\Delta t_t C^{\mathrm{carrier}}_{c,n,t}.

The node- and time dependent carrier costs :math:`C^{\mathrm{carrier}}_{c,n,t}` are composed of
three terms: the carrier import :math:`F^{\mathrm{imp}}_{c,n,t}` multiplied by
the import price :math:`\pi^{\mathrm{imp}}_{c,n,t}`, the carrier export
:math:`F^{\mathrm{exp}}_{c,n,t}` multiplied by the export price
math:`\pi^{\mathrm{exp}}_{c,n,t}`, and the shed demand :math:`F^{\mathrm{shed}}_{c,n,t}`
multiplied by demand shedding price :math:`\pi^{\mathrm{shed}}_c`:

.. math::
    :label: cost_carrier

    C^{\mathrm{carrier}}_{c,n,t} = \pi^{\mathrm{imp}}_{c,n,t}F^{\mathrm{imp}}_{c,n,t}-
    \pi^{\mathrm{exp}}_{c,n,t}F^{\mathrm{exp}}_{c,n,t}+\pi^{\mathrm{shed}}_c F^{\mathrm{shed}}_{c,n,t}

*Operational expenditures emissions*

The annual operational emission expenditures :math:`C_y^{\mathrm{CO_2}}` are
composed of three terms: the annual carbon emissions :math:`M_y`  multiplied by
the carbon emission price :math:`\pi^{\mathrm{CO_2}}`, the annual carbon emission overshoot
:math:`M_y^{\mathrm{ann,over}}` multiplied by the annual carbon overshoot price
:math:`\pi^{\mathrm{CO_2,ann}}`, and the budget carbon emission overshoot
math:`M_y^{\mathrm{bud,over}}` multiplied by the carbon emission budget overshoot price
:math:`\pi^{\mathrm{CO_2,bud}}`:

.. math::
    :label: opex_e

    C_y^{\mathrm{CO_2}} = M_y \pi^{\mathrm{CO_2}} +
    M_y^{\mathrm{ann,over}}\pi^{\mathrm{CO_2,ann}}+M_y^{\mathrm{bud,over}}\pi^{\mathrm{CO_2,bud}}.

For a detailed description on how to use the annual carbon emission overshoot
price and the carbon emission budget overshoot price refer to
:ref:`additional_features.modeling_carbon_emissions`.

.. _math_formulation.emissions_objective:

Minimizing total emissions
^^^^^^^^^^^^^^^^^^^^^^^^^^

The cumulative carbon emissions at the end of the time horizon
:math:`M^{\mathrm{cum}}_Y` of the energy system are minimized.

.. math::
    :label: min_emissions

    \mathrm{min} \quad M^{\mathrm{cum}}_Y

The cumulative carbon emissions at the end of the time horizon
:math:`M^{\mathrm{cum}}_Y` account for the total operational carbon emissions
for importing and exporting carriers :math:`M^{\mathrm{carrier}}_y` and for
operating technologies :math:`M^{\mathrm{tech}}_y`:

.. math::
    :label: total_annual_carbon_emissions

    M_y = M^{\mathrm{carrier}}_y + M^{\mathrm{tech}}_y.


For a detailed description of the computation of the total operational emissions
for importing and exporting carriers, and for operating for operating
technologies refer to :ref:`math_formulation.emissions_constraints`.


.. _math_formulation.energy_balance:

Energy balance
---------------

The sources and sinks of a carrier :math:`c\in\mathcal{C}` must be in
equilibrium for all carriers at all nodes :math:`n\in\mathcal{N}` and in all
time steps :math:`t\in\mathcal{T}_y`. The source terms for carrier :math:`c` on
node :math:`n` are:

* the output flow :math:`F^{\mathrm{conv,out}}_{h,c^{\mathrm{out}},n,t}` of all conversion
  technologies :math:`h\in\mathcal{H}^{\mathrm{conv}}` if :math:`c\in\mathcal{C}^{\mathrm{out}}_h`.
* the transported flow :math:`F^{\mathrm{trans}}_{h,e,t}` on ingoing edges
  :math:`e\in\mathcal{E}^{\mathrm{in}}_n` minus the losses
  :math:`F^{\mathrm{loss}}_{h,e,t}` for all transport technologies
  :math:`h\in\mathcal{H}^{\mathrm{trans}}` if :math:`c=c_h^{\mathrm{ref}}`.
* the discharge flow :math:`F^{\mathrm{dis}}_{h,n,t}` for all storage technologies
  :math:`h\in\mathcal{H}^{\mathrm{stor}}` if :math:`c=c_h^{\mathrm{ref}}`.
* the imported flow :math:`F^{\mathrm{imp}}_{c,n,t}`.

The sinks of carrier :math:`c` on node :math:`n` are:

* the exogenous demand :math:`d_{c,n,t}` minus the shed demand
  :math:`F^{\mathrm{shed}}_{c,n,t}`.
* the input flow :math:`F^{\mathrm{conv,in}}_{h,c^{\mathrm{in}},n,t}` of all conversion
  technologies :math:`h\in\mathcal{H}^{\mathrm{conv}}` if :math:`c\in\mathcal{C}^{\mathrm{in}}_h`.
* the transported flow :math:`F^{\mathrm{trans}}_{h,e',t}` on outgoing edges
  :math:`e'\in\mathcal{E}^{\mathrm{out}}_n` for all transport technologies
  :math:`h\in\mathcal{H}^{\mathrm{trans}}` if :math:`c=c_h^{\mathrm{ref}}`.
* the charge flow :math:`F^{\mathrm{ch}}_{h,n,t}` for all storage technologies
  :math:`h\in\mathcal{H}^{\mathrm{stor}}` if :math:`c=c_h^{\mathrm{ref}}`.
* the exported flow :math:`F^{\mathrm{exp}}_{c,n,t}`.

The energy balance for carrier :math:`c\in\mathcal{C}` is then calculated as:

.. math::
    :label: energy_balance

    0 = -\left(d_{c,n,t}-F^{\mathrm{shed}}_{c,n,t}\right) +
    \sum_{h\in\mathcal{H}^{\mathrm{conv}}}\left(F^{\mathrm{conv,out}}_{h,c^{\mathrm{out}},n,t}-
    F^{\mathrm{conv,in}}_{h,c^{\mathrm{in}},n,t}\right) +
    \sum_{h\in\mathcal{H}^{\mathrm{trans}}}\left(\sum_{e\in\mathcal{E}^{\mathrm{in}}_n}\left(F^{\mathrm{trans}}_{h,e,t} -
    F^{\mathrm{loss}}_{h,e,t}\right)-\sum_{e'\in\mathcal{E}^{\mathrm{out}}_n}F^{\mathrm{trans}}_{h,e',t}\right) +
     \sum_{h\in\mathcal{H}^{\mathrm{stor}}}\left(F^{\mathrm{dis}}_{h,n,t}-F^{\mathrm{ch}}_{h,n,t}\right)+
     F^{\mathrm{imp}}_{c,n,t} - F^{\mathrm{exp}}_{c,n,t}.

.. note::
    :math:`\sum_{h\in\mathcal{H}^{\mathrm{stor}}}\left(F^{\mathrm{dis}}_{h,n,t}-F^{\mathrm{ch}}_{h,n,t}\right)`
    are zero if :math:`c\neq c_h^{\mathrm{ref}}` and :math:`c\neq c_h^{\mathrm{ref}}`,
    respectively.

The carrier import :math:`F^{\mathrm{imp}}_{c,n,t}` is limited by the carrier
import availability :math:`a^{\mathrm{imp}}_{c,n,t}` for all carriers
:math:`c\in\mathcal{C}` in all nodes :math:`n\in\mathcal{N}` and time steps
:math:`t\in\mathcal{T}_y`:

.. math::
    :label: carrier_import

    0 \leq F^{\mathrm{imp}}_{c,n,t} \leq a^{\mathrm{imp}}_{c,n,t}.

In addition, annual carrier import limits can be applied:

.. math::
    :label: carrier_import_yearly

    0 \leq \sum_{t\in\mathcal{T}_y} \Delta t_t F^{\mathrm{imp}}_{c,n,t} \leq
    a^{\mathrm{imp,yr}}_{c,n,y}.

Similarly, the carrier export :math:`F^{\mathrm{exp}}_{c,n,t}` is limited by the
carrier export availability :math:`a^{\mathrm{exp}}_{c,n,t}` for all carriers
:math:`c\in\mathcal{C}` in all nodes :math:`n\in\mathcal{N}` and time steps
:math:`t\in\mathcal{T}_y`:

.. math::
    :label: carrier_export

    0 \leq F^{\mathrm{exp}}_{c,n,t} \leq a^{\mathrm{exp}}_{c,n,t}.

In addition, annual carrier export limits can be applied:

.. math::
    :label: carrier_export_yearly

    0 \leq \sum_{t\in\mathcal{T}_y} \Delta t_t F^{\mathrm{exp}}_{c,n,t} \leq
    a^{\mathrm{exp,yr}}_{c,n,y}.

.. note::
    You can skip the import and export availability constraints by setting the
    import and export availabilities to infinity.

Lastly, the following constraint ensures that the shed demand
:math:`F^{\mathrm{shed}}_{c,n,t}` does not exceed the demand :math:`d_{c,n,t}`:

.. math::
    :label: demand_shedding

    0 \leq F^{\mathrm{shed}}_{c,n,t} \leq d_{c,n,t}.

.. note::
    Setting the shed demand cost to infinity forces :math:`F^{\mathrm{shed}}_{c,n,t}=0` and
    demand shedding will not be possible. :ref:`additional_features.demand_shedding` provides a more
    detailed description on demand shedding.

.. _math_formulation.emissions_constraints:

Emissions constraints
-----------------------

The total annual carrier carbon emissions :math:`M^{\mathrm{carrier}}_y` represent
the sum of the carrier carbon emissions
:math:`M^{\mathrm{carrier}}_{c,n,t}`:

.. math::
    :label: total_carbon_emissions_carrier

    M^{\mathrm{carrier}}_y = \sum_{t\in\mathcal{T}_y} \sum_{n\in\mathcal{N}}
    \sum_{c\in\mathcal{C}} \left( \Delta t_t M^{\mathrm{carrier}}_{c,n,t}
    \right).

The carrier carbon emissions include the operational emissions of importing and
exporting carriers :math:`c\in\mathcal{C}` (carbon intensity
:math:`\varepsilon^{\mathrm{imp}}_{c,y}` and :math:`\varepsilon^{\mathrm{exp}}_{c,y}`):

.. math::
    :label: carbon_emissions_carrier

    M^{\mathrm{carrier}}_{c,n,t} =
    \varepsilon^{\mathrm{imp}}_{c,y} F^{\mathrm{imp}}_{c,n,t} -
    \varepsilon^{\mathrm{exp}}_{c,y} F^{\mathrm{exp}}_{c,n,t}.

The total annual technology carbon emissions :math:`M^{\mathrm{tech}}_y` represent
the sum of the technology carbon emissions :math:`M^{\mathrm{tech}}_{h,n,t}`:

.. math::
    :label: total_carbon_emissions_technology

    M^{\mathrm{tech}}_y = \sum_{t\in\mathcal{T}_y} \sum_{n\in\mathcal{N}}
    \sum_{h\in\mathcal{H}} \left( M^{\mathrm{tech}}_{h,n,t} \Delta t_t \right).

The technology carbon emissions :math:`M^{\mathrm{tech}}_{h,n,t}` include
the emissions for operating the technologies :math:`h\in\mathcal{H}` (carbon
intensity :math:`\varepsilon^{\mathrm{op}}_h`). For conversion technologies
:math:`h\in\mathcal{H}^{\mathrm{conv}}`, the carbon intensity of operating the technology is
multiplied with the reference flows :math:`F^{\mathrm{ref}}_{h,n,t}`:

.. math::
    :label: carbon_emissions_conversion

    M^{\mathrm{tech}}_{h,n,t} = \varepsilon^{\mathrm{op}}_h F^{\mathrm{ref}}_{h,n,t}.

For storage technologies :math:`h\in\mathcal{H}^{\mathrm{stor}}`, the carbon intensity of
operating the technology is multiplied with the storage charge and discharge
flows :math:`F^{\mathrm{dis}}_{h,n,t}` and :math:`F^{\mathrm{ch}}_{h,n,t}`:

.. math::
    :label: carbon_emissions_storage

    M^{\mathrm{tech}}_{h,n,t} =
    \varepsilon^{\mathrm{op}}_h \left( F^{\mathrm{dis}}_{h,n,t}+F^{\mathrm{ch}}_{h,n,t} \right).

Lastly, for transport technologies :math:`h\in\mathcal{H}^{\mathrm{trans}}`, the carbon intensity
of operating the technology is multiplied with the transported flow
:math:`F^{\mathrm{trans}}_{h,e,t}`:

.. math::
    :label: carbon_emissions_transport

    M^{\mathrm{tech}}_{h,e,t} = \varepsilon^{\mathrm{op}}_h F^{\mathrm{trans}}_{h,e,t}.

The annual carbon emissions :math:`M_y` are limited by the annual carbon
emissions limit :math:`\overline{m}_y`:

.. math::
    :label: carbon_emissions_annual_limit

    M_y - M_y^{\mathrm{ann,over}} \leq \overline{m}_y.

Note that :math:`\overline{m}_y` can be infinite, in which case the constraint is skipped.

:math:`M_y^{\mathrm{ann,over}}` is the annual carbon emission limit overshoot and
allows exceeding the annual carbon emission limits. However, overshooting the
annual carbon emission limits is penalized in the objective function
(compare Eq. :eq:`opex_e`). This overshoot cost is computed by multiplying the
annual carbon emission limit overshoot :math:`M_y^{\mathrm{ann,over}}` with the annual
carbon emission limit overshoot price :math:`\pi^{\mathrm{CO_2,ann}}`. To strictly
enforce the annual carbon emission limit (i.e., :math:`M_y^{\mathrm{ann,over}}=0`),
use an infinite carbon overshoot price :math:`\pi^{\mathrm{CO_2,ann}}`.

The cumulative carbon emissions :math:`M_y^{\mathrm{cum}}` are attributed to the
end of the year. For the first planning period :math:`y=y_0`,
:math:`M_y^{\mathrm{cum}}` is calculated as:

.. math::
    :label: carbon_emissions_cum_0

    M_y^{\mathrm{cum}} = M_y.

In the subsequent periods :math:`y>y_0`, :math:`M_y^{\mathrm{cum}}` is calculated
as:

.. math::
    :label: carbon_emissions_cum_1

    M_y^{\mathrm{cum}} =
    M_{y-1}^{\mathrm{cum}} + \left(\Delta y-1\right)M_{y-1}+M_y.

The cumulative carbon emissions :math:`M_y^{\mathrm{cum}}` are constrained by the
carbon emission budget :math:`\overline{m}^{\mathrm{budget}}`:

.. math::
    :label: emission_budget

    M_y^{\mathrm{cum}} + \left( \Delta y-1 \right) M_y  -
    M_y^{\mathrm{bud,over}} \leq \overline{m}^{\mathrm{budget}}.

Note that :math:`\overline{m}^{\mathrm{budget}}` can be infinite, in which case the constraint is
skipped. :math:`M_y^{\mathrm{bud,over}}` is the cumulative carbon emission overshoot and
allows exceeding the carbon emission budget :math:`\overline{m}^{\mathrm{budget}}`, where
exceeding the carbon emission budget in the last year of the planning horizon
:math:`\mathrm{Y}=\max(y)` (i.e., :math:`M_\mathrm{Y}^{\mathrm{bud,over}}>0`) is
penalized with the carbon emissions budget overshoot price
:math:`\pi^{\mathrm{CO_2,bud}}` in the objective function (compare Eq. :eq:`opex_c`).
By setting the carbon emission budget overshoot price to infinite, you can
enforce that the cumulative carbon emissions stay below the carbon emission
budget :math:`\overline{m}^{\mathrm{budget}}` across all years (i.e.,
:math:`E_\mathrm{y}^\mathrm{bo} = 0 ,\forall y\in\mathcal{Y}`).


.. _math_formulation.operational_constraints:

Operational constraints
----------------------------

The conversion factor :math:`\eta^{\mathrm{conv}}_{h,c,t}` describes the ratio between the
carrier flow :math:`c\in\mathcal{C}` and the reference carrier flow
:math:`F^{\mathrm{ref}}_{h,n,t}` of a conversion technology
:math:`h\in\mathcal{H}^{\mathrm{conv}}`. If the carrier flow is an input carrier, i.e.
:math:`c\in\mathcal{C}^{\mathrm{in}}_h`:

.. math::

    \eta^{\mathrm{conv}}_{h,c,t} =
    \frac{F^{\mathrm{conv,in}}_{h,c^{\mathrm{in}},n,t}^{\mathrm{d}}}{F^{\mathrm{ref}}_{h,n,t}.

If the carrier flow is an output carrier, i.e.
:math:`c\in\mathcal{C}^{\mathrm{out}}_h`:

.. math::

    \eta^{\mathrm{conv}}_{h,c,t} =
    \frac{F^{\mathrm{conv,out}}_{h,c^{\mathrm{out}},n,t}^{\mathrm{d}}}{F^{\mathrm{ref}}_{h,n,t}.

All carrier flows that are not reference carrier flows are called dependent
carrier flows :math:`F^{\mathrm{conv,dep}}_{h,c,n,t}`.

The transport flow losses :math:`F^{\mathrm{loss}}_{h,e,t}` through a transport
technology :math:`h\in\mathcal{H}^{\mathrm{trans}}` on edge :math:`e\in\mathcal{E}` are expressed
by the loss function :math:`\lambda^{\mathrm{loss}}_{h,e}` and the transported quantity:

.. math::

    F^{\mathrm{loss}}_{h,e,t} = \lambda^{\mathrm{loss}}_{h,e} d^{\mathrm{dist}}_{h,e} F^{\mathrm{trans}}_{h,e,t}.

The loss function is described through a linear loss factor
:math:`\lambda^{\mathrm{lin}}_h`, applied to the transport distance
:math:`d^{\mathrm{dist}}_{h,e}`:

.. math::
    :label: transport_flow_loss_linear

    \lambda^{\mathrm{loss}}_{h,e} = d^{\mathrm{dist}}_{h,e} \lambda^{\mathrm{lin}}_h

The flow of the reference carrier :math:`c_h^{\mathrm{ref}}` of all technologies
:math:`h\in\mathcal{H}` is constrained by the maximum load
:math:`\ell^{\mathrm{max}}_{h,p,t}` and the installed capacity :math:`K_{h,p,y}`.
For conversion technologies :math:`h\in\mathcal{H}^{\mathrm{conv}}`, it follows:

.. math::

    0 \leq F^{\mathrm{ref}}_{h,n,t} \leq \ell^{\mathrm{max}}_{h,n,t}K_{h,n,y}.

Analogously for transport technologies :math:`h\in\mathcal{H}^{\mathrm{trans}}` it follows:

.. math::

    0 \leq F^{\mathrm{trans}}_{h,e,t} \leq \ell^{\mathrm{max}}_{h,e,t}K_{h,e,y}.

Since a storage technology does not charge (:math:`F^{\mathrm{ch}}_{h,n,t}`) and
discharge (:math:`F^{\mathrm{dis}}_{h,n,t}`) at the same time, the sum of both
flows is constrained by the maximum load:

.. math::

    0 \leq F^{\mathrm{ch}}_{h,n,t}+
    F^{\mathrm{dis}}_{h,n,t}\leq \ell^{\mathrm{max}}_{h,n,t}K_{h,n,y}.

In addition, minimum load constraints can be added. Please note, that adding a
minimum load :math:`\ell^{\mathrm{min}}_{h,p,t}` introduces binary variables, which
can increase the computational complexity of the optimization problem
substantially. The min-load constraints are described in
:ref:`math_formulation.min_load_constraints`.

Furthermore, the reference flow of retrofitting technologies is linked to the
reference flow of their base technology. The set of base technologies links each
retrofitting technology :math:`i^\mathrm{r}` to their base technology :math:`i`.
The retrofit flow coupling factor can be interpreted as a conversion factor
:math:`\eta^{\mathrm{retro}}_{h,n,t}` that describes the ratio
between the reference flow of the retrofitting technology and the reference flow
of the base technology:

.. math::

    F^{\mathrm{ref}}_{h^{\mathrm{retro}},n,t} =
    \eta^{\mathrm{retro}}_{h^{\mathrm{retro}},n,t}
    F^{\mathrm{ref}}_{h^{\mathrm{base}},n,t}.

The temporal representation of storage technologies :math:`h\in\mathcal{H}^{\mathrm{stor}}` is
particular because the storage constraints are time-coupled and the sequence of
time steps must be preserved. To enable both the modeling of short- and
medium-term storage, e.g., battery and pumped hydro storage, and long-term
storage, e.g., natural gas storage, we present a novel formulation, where the
energy-rated storage variables are resolved on a different time sequence. The
approach is detailed in `Mannhardt et al. 2023 <https://www.sciencedirect.com/science/article/pii/S2589004223008271>`_.
In particular, each change in the aggregated time sequence for power-rated
variables yields an additional time step for the energy-rated storage variables.
Assume the representation of the exemplary full time index
:math:`\mathcal{T}^{\mathrm{full}}_y=[0,...,9]` by four representative time steps
:math:`\mathcal{T}_y=[0,...,3]` with the sequence
:math:`\sigma= [0,0,1,2,1,1,3,3,2,0]` for power-rated variables. The resulting
sequence for energy-rated storage variables :math:`\widetilde{\sigma}` of the
storage time steps :math:`\widetilde{\mathcal{T}}_y=[0,...,6]` is then:

.. math::
    :label: storage_time_sequence

    \widetilde{\sigma} = [0,0,1,2,3,3,4,4,5,6]

While this formulation enables both the short-term and long-term operation of
storages, it increases the number of time steps
:math:`\vert \widetilde{\mathcal{T}}_y\vert` and thus the number of variables.

For sake of simplicity, let :math:`\sigma:\widetilde{\mathcal{T}}_y\to \mathcal{T}_y`
denote the unique mapping of a storage level time step :math:`\tilde{t}` to a
power-rated time step :math:`t`. The time-coupled equation for the storage level
:math:`S^{\mathrm{level}}_{h,n,\tilde{t}}` of storage technology :math:`k` at node :math:`n`
is formulated for each storage level time step except the first
:math:`\tilde{t}\in\widetilde{\mathcal{T}}_y\setminus\{0\}` as:

.. math::
    :label: storage_level

    S^{\mathrm{level}}_{h,n,\tilde{t}} =
    S^{\mathrm{level}}_{h,n,\tilde{t}-1}\left(1-\lambda^{\mathrm{self}}_h\right)^
    {\Delta \tilde{t}_{\tilde{t}}}+
    \left(\eta^{\mathrm{ch}}_hF^{\mathrm{ch}}_{h,n,\sigma(\tilde{t})}-
    \frac{F^{\mathrm{dis}}_{h,n,\sigma(\tilde{t})}}{\eta^{\mathrm{dis}}_h} +
    q^{\mathrm{in}}_{h,n,\sigma(\tilde{t})} - F^{\mathrm{spill}}_{h,n,\sigma(\tilde{t})} \right)
    \sum_{\tau=0}^{\Delta \tilde{t}_{\tilde{t}}-1}
    \left(1-\lambda^{\mathrm{self}}_h\right)^{\tau}

with the self-discharge rate :math:`\lambda^{\mathrm{self}}_h`, the charge and discharge
efficiency, :math:`\eta^{\mathrm{ch}}_h` and :math:`\eta^{\mathrm{dis}}_h`, the
duration of a storage level time step :math:`\Delta \tilde{t}_{\tilde{t}}`,
the inflow in the storage :math:`q^{\mathrm{in}}_{h,n,\sigma(\tilde{t})}`, and the
spillage out of the storage :math:`F^{\mathrm{spill}}_{h,n,\sigma(\tilde{t})}`.
Note that we reformulate :math:`\sum_{\tau=0}^{\Delta \tilde{t}_{\tilde{t}}-1}\left(1-\lambda^{\mathrm{self}}_h\right)^{\tau}`
in the optimization problem with the partial geometric series to avoid
constructing an additional summation term:

.. math::
    :label: partial_geom_series

    \sum_{\tau=0}^
    {\Delta \tilde{t}_{\tilde{t}}-1}
    \left(1-\lambda^{\mathrm{self}}_h\right)^{\tau} =
    \frac{1-\left(1-\lambda^{\mathrm{self}}_h\right)^
    {\Delta \tilde{t}_{\tilde{t}}}}{\lambda^{\mathrm{self}}_h}

If storage periodicity is enforced (``system.storage_periodicity = True``), the
storage level at :math:`\tilde{t}=0` is coupled with the level in the last
time step of the period :math:`\tilde{t}=\max(\widetilde{\mathcal{T}}_y)`:

.. math::
    :label: storage_level_periodicity

    S^{\mathrm{level}}_{h,n,0} = S^{\mathrm{level}}_{h,n,\max(\widetilde{\mathcal{T}}_y)}\left(1-\lambda^{\mathrm{self}}_h\right)^
    {\Delta \tilde{t}_{\tilde{t}}}+
    \left(\eta^{\mathrm{ch}}_hF^{\mathrm{ch}}_{h,n,\sigma(0)}-
    \frac{F^{\mathrm{dis}}_{h,n,\sigma(0)}}{\eta^{\mathrm{dis}}_h} +
    q^{\mathrm{in}}_{h,n,\sigma(0)} - F^{\mathrm{spill}}_{h,n,\sigma(0)} \right)
    \sum_{\tau=0}^
    {\Delta \tilde{t}_{\tilde{t}}-1}
    \left(1-\lambda^{\mathrm{self}}_h\right)^{\tau}

Moreover, the :math:`S^{\mathrm{level}}_{h,n,\tilde{t}}` is constrained by the energy-rated
storage capacity :math:`K^{\mathrm{energy}}_{h,n,y}`:

.. math::
    :label: limit_storage_level

    0 \leq S^{\mathrm{level}}_{h,n,\tilde{t}}\leq K^{\mathrm{energy}}_{h,n,y}

:math:`S^{\mathrm{level}}_{h,n,\tilde{t}}` is monotonous between :math:`\tilde{t}` and
:math:`\tilde{t}+1`. Hence, :math:`S^{\mathrm{level}}_{h,n,\tilde{t}}` and
:math:`S^{\mathrm{level}}_{h,n,\tilde{t}+1}` are the local extreme values and Eq.
:eq:`limit_storage_level` constrains the entire time interval between
:math:`\tilde{t}` and :math:`\tilde{t}+1`. We prove this below.

The storage level at :math:`\tilde{t}=0` can be set to an initial storage
level :math:`s^{\mathrm{init}}_{h,n}` as a share of :math:`K^{\mathrm{energy}}_{h,n,y}`:

.. math::

    S^{\mathrm{level}}_{h,n,0} = s^{\mathrm{init}}_{h,n}K^{\mathrm{energy}}_{h,n,y}

The spillage is a non-negative variable that is constrained by the inflow
:math:`q^{\mathrm{in}}_{h,n,\tilde{t}}`:

.. math::
    :label: spillage_limit

    0 \leq F^{\mathrm{spill}}_{h,n,\tilde{t}} \leq q^{\mathrm{in}}_{h,n,\tilde{t}}


**Proof of storage level monotony**

We prove that Eq. :eq:`storage_level` is monotonous on the entire time interval
that is aggregated to a single storage time step :math:`\tilde{t}`. Consider
Eq. :eq:`storage_level` for one storage time step :math:`\tilde{t}`, during
which :math:`F^{\mathrm{ch}}_{h,n,\sigma(\tilde{t})}` and
:math:`F^{\mathrm{dis}}_{h,n,\sigma(\tilde{t})}` are constant. Neglecting all
further indices without loss of generality, the storage level :math:`L(t)` for
the intermediate time steps :math:`t\in[1,\Delta \tilde{t}_{\tilde{t}}]`
follows as:

.. math::
    :label: storage_level_simpl

    L(t) = L_0\kappa^t + \Delta H\sum_{\tilde{t}=0}^{t-1}\kappa^{\tilde{t}},

with :math:`\kappa=1-\varphi` and :math:`\Delta H=\left(\underline{\eta}F^{\mathrm{ch}}-\frac{F^{\mathrm{dis}}}{\overline{\eta}}\right)`.
:math:`L_0` is the storage level at the end of the previous storage time step
:math:`\tilde{t}-1`. Without self-discharge
(:math:`\varphi=0\Rightarrow\kappa=1`), it follows:

.. math::

    L(t) = L_0 + \Delta Ht \Rightarrow \frac{\mathrm{d}L(t)}{\mathrm{d}t}=\Delta H.

Since :math:`\frac{\mathrm{d}L(t)}{\mathrm{d}t}` is independent of :math:`t`,
Eq. :eq:`storage_level_simpl` is monotonous for :math:`\varphi=0`.

For :math:`0<\varphi<1`, :math:`\sum_{\tilde{t}=0}^{t-1}\kappa^{\tilde{t}}` is
reformulated as the partial geometric series (compare Eq.
:eq:`partial_geom_series`).

.. math::

    \sum_{\tilde{t}=0}^{t-1}\kappa^{\tilde{t}} = \frac{1-\kappa^t}{1-\kappa}.

Eq. :eq:`storage_level_simpl` is reformulated to:

.. math::
    :label: storage_level_selfdisch

    L(t) = L_0\kappa^t + \Delta H\frac{1-\kappa^t}{1-\kappa} =
    \frac{\Delta H}{1-\kappa}+
    \left(L_0-\frac{\Delta H}{1-\kappa}\right)\kappa^t.

The derivative of Eq. :eq:`storage_level_selfdisch` follows as:

.. math::

    \frac{\mathrm{d}L(t)}{\mathrm{d}t} =
    \underbrace{\left(L_0-\frac{\Delta H}{1-\kappa}\right)\ln(\kappa)}_
    {= \text{ constant }\forall t\in[1,\Delta \tilde{t}_{\tilde{t}}]}\kappa^t.

With :math:`\kappa^t>0`, it follows that Eq. :eq:`storage_level_simpl` is
monotonous for :math:`0<\varphi<1`.

.. _math_formulation.investment_constraints:

Investment constraints
----------------------

The capacity :math:`K_{h,p,y}` of a technology :math:`h\in\mathcal{H}` at a
position :math:`p\in\mathcal{P}` in period :math:`y` is the sum of all previous
capacity additions :math:`\Delta K_{h,p,y}` and existing capacities
:math:`k^{\mathrm{ex}}_{h,p,y}`, that are still within their usable
technical lifetime :math:`L_h` (compare Eq. :eq:`annuity`):

.. math::
    :label: capacity

    K_{h,p,y}=\sum_{\tilde{y}=
    \max\left(y_0,y-\left\lceil\frac{L_h}{\Delta y}\right\rceil+1\right)}^y
    \Delta K_{h,p,\tilde{y}}+
    \sum_{\hat{y}=\psi\left(\min
    \left(y_0-1,y-\left\lceil\frac{L_h}{\Delta y}\right\rceil+1\right)\right)}^
    {\psi(y_0)} k^{\mathrm{ex}}_{h,p,\hat{y}}.

The technology capacity :math:`K_{h,p,y}` is constrained by the capacity limit
:math:`\overline{k}_{h,p,y}`:

.. math::

    K_{h,p,y} \leq \overline{k}_{h,p,y}.

The technology capacity :math:`K_{h,p,y}` is constrained from below by the capacity lower limit
:math:`\underline{k}_{h,p,y}`:

.. math::
   :label: capacity_lower_limit

   K_{h,p,y} \geq \underline{k}_{h,p,y}.

The capacity addition :math:`\Delta K_{h,p,y}` is constrained by the maximum
capacity addition :math:`\overline{\Delta k}_{h,p,y}`:

.. math::

    0 \leq \Delta K_{h,p,y} \leq \overline{\Delta k}_{h,p,y}

.. note::

    You can skip the maximum capacity addition constraint for a technology by
    setting the maximum capacity addition to infinity.

You can also introduce a minimum capacity addition
:math:`\underline{\Delta k}_{h,p,y}`. However, please note, that adding a
minimum capacity addition :math:`\underline{\Delta k}_{h,p,y}` introduces
binary variables, which can increase the computational complexity of the
optimization problem substantially. The min-capacity addition constraints are
described in :ref:`math_formulation.min_capacity_installation`.

Furthermore, for storage technologies the ratios of the energy- and power rated
capacity additions are constrained by the energy-to-power ratio
:math:`r_h^{\mathrm{EP}}`. Minimum and maximum energy-to-power ratios can be defined.
For infinite power ratios, the constraints are skipped.

.. math::
    r_h^{\mathrm{EP,min}} K^{\mathrm{energy}}_{h,n,y} \le K_{h,n,y}

.. math::
    K_{h,n,y} \le r_h^{\mathrm{EP,max}} K^{\mathrm{energy}}_{h,n,y}

To account for technology construction times :math:`\Delta y^\mathrm{construction}`
we introduce an auxiliary variable, :math:`\Delta K^{\mathrm{inv}}_{h,p,y}`,
representing the technology investments. The following constraint ensures that
the new technology capacities do not become available before the construction
time has passed:

.. math::
    :label: construction_time

    \Delta K_{h,p,y} =
    \Delta K_{h,p,\left(y-\Delta y^\mathrm{construction}\right)}^\mathrm{invest}

Furthermore, if :math:`y-\Delta y^\mathrm{construction}<0`:

.. math::

    \Delta K_{h,p,y} = 0

**Constrained technology deployment**

In case you are using constrained technology deployment
(``max_diffusion_rate != np.inf`` for a technology), :math:`\Delta K_{h,p,y}` is
constrained by the existing knowledge of how to install the technology
:math:`K_{h,p,y}` with the technology diffusion rate :math:`r^{\mathrm{diff}}_h`. This
approach is based on `Leibowicz et al. (2016)
<https://www.sciencedirect.com/science/article/pii/S0040162515001675>`_.

For node-based technologies, i.e., conversion and storage technologies,
spillover effects from other nodes
:math:`\tilde{\mathcal{N}} = \mathcal{N}\setminus\{n\}` can be utilized
(knowledge spillover rate :math:`\omega`). To allow for an entry into a niche
market, we add an unbounded market share :math:`\chi` of the total capacity of
all other technologies with the same reference carrier:

.. math::

    \tilde{\mathcal{H}}=
    \Set{\tilde{h}\in\mathcal{H}\setminus\{h\}
    \mid c_{\tilde{h}}^{\mathrm{ref}} = c_{h}^{\mathrm{ref}}}

With the unbounded capacity addition :math:`k^{\mathrm{add,free}}_h`, it follows for the
conversion technologies :math:`h\in\mathcal{H}^{\mathrm{conv}}`:

.. math::
    :label: constrained_technology_deployment_i

    \Delta K_{h,n,y}\leq
    \left((1+r^{\mathrm{diff}}_h)^{\Delta y}-1\right)\left(K_{h,n,y}+
    \omega\sum_{\tilde{n}\in\tilde{\mathcal{N}}}K_{h,\tilde{n},y}\right)+
    \Delta y\left(\chi\sum_{\tilde{h}\in\widetilde{\mathcal{H}}^{\mathrm{conv}}}K_{\tilde{h},n,y} +
    k^{\mathrm{add,free}}_h\right)

Analogously, it follows for the storage technologies :math:`h\in\mathcal{H}^{\mathrm{stor}}`:

.. math::
    :label: constrained_technology_deployment_k

    \Delta K_{h,n,y}\leq \left((1+r^{\mathrm{diff}}_h)^{\Delta y}-1\right)\left(K_{h,n,y}+
    \omega\sum_{\tilde{n}\in\tilde{\mathcal{N}}}K_{h,\tilde{n},y}\right)+
    \Delta y\left(\chi\sum_{\tilde{h}\in\widetilde{\mathcal{H}}^{\mathrm{stor}}}K_{\tilde{h},n,y} +
    k^{\mathrm{add,free}}_h\right)

We prohibit spillover effects for transport technologies :math:`h\in\mathcal{H}^{\mathrm{trans}}`
from other edges:

.. math::
    :label: constrained_technology_deployment_j

    \Delta K_{h,e,y}\leq \left((1+r^{\mathrm{diff}}_h)^{\Delta y}-1\right)K_{h,e,y}+
    \Delta y\left(\chi\sum_{\tilde{h}\in\widetilde{\mathcal{H}}^{\mathrm{trans}}}K_{\tilde{h},e,y} +
    k^{\mathrm{add,free}}_h\right)


To avoid the unrealistically excessive use of spillover effects, we constrain
the capacity additions in all positions as follows:

.. math::
    :label: constrained_technology_deployment_all

    \sum_{p\in\mathcal{P}}\Delta K_{h,p,y}\leq
    \sum_{p\in\mathcal{P}}\Bigg(\left((1+r^{\mathrm{diff}}_h)^{\Delta y}-1\right)K_{h,p,y}+
    \Delta y\left(\chi\sum_{\tilde{h}\in\tilde{\mathcal{H}}}K_{\tilde{h},p,y} +
    k^{\mathrm{add,free}}_h\right)\Bigg)

.. note::

    If you set :math:`\omega=\infty`, we assume infinite spillover effects
    between nodes and Eqs. :eq:`constrained_technology_deployment_i`-:eq:`constrained_technology_deployment_j`
    are skipped.     Then the constrained technology expansion for the entire
    energy system is governed by Eq. :eq:`constrained_technology_deployment_all`.

:math:`K_{h,p,y}` is a function of the previous capacity additions
:math:`\Delta K_{h,p,y}` and :math:`k^{\mathrm{ex}}_{h,p,y}` as it
represents the expertise and knowledge of the industry on how to install a
certain amount of capacity. This knowledge is depreciated over time with the
knowledge depreciation rate :math:`\delta`:

.. math::

    K_{h,p,y} = \sum_{\tilde{y}=y_0}^{y-1}\left(1-\delta\right)^
    {\Delta y (y-\tilde{y})}\Delta K_{h,p,\tilde{y}} +
    \sum_{\hat{y}=-\infty}^{\psi(y_0)}\left(1-\delta\right)^{\left(\Delta y(y-y_0) +
    (\psi(y_0)-\hat{y})\right)}k^{\mathrm{ex}}_{h,p,\hat{y}}

.. _math_formulation.min_load_constraints:

Minimum load constraints
------------------------

A binary variable :math:`z^{\mathrm{on}}_{h,p,t}` is introduced to model the on-, and off-
behaviour of a technology. If :math:`z^{\mathrm{on}}_{h,p,t}=1`, the technology is on, if
:math:`z^{\mathrm{on}}_{h,p,t}=0` the technology is considered off. With :math:`z^{\mathrm{on}}_{h,p,t}` the
minimum load constraint of a conversion technology can be formulated as follows:

.. math::
    :label: min_load_conversion_bilinear

    \ell^{\mathrm{min}}_{h,p,t} z^{\mathrm{on}}_{h,p,t}  K_{h,p,y} \leq
    F^{\mathrm{ref}}_{h,p,t} \leq z^{\mathrm{on}}_{h,p,t}  K_{h,p,y}

However, this constraint would introduce a bilinearity. To resolve the
bilinearity, we use a big-M formulation and approximate
:math:`z^{\mathrm{on}}_{h,p,t} K_{h,n,y}` with :math:`\widehat{K}_{h,p,t}`. Thus, Eq.
:eq:`min_load_conversion_bilinear` can be rewritten as:

.. math::
    :label: min_load_conversion

    \ell^{\mathrm{min}}_{h,n,t} \widehat{K}_{h,n,t} \leq
     F^{\mathrm{ref}}_{h,n,t} \leq \widehat{K}_{h,n,t}

Similarly, for transport technologies it follows:

.. math::
    :label: min_load_transport

    \ell^{\mathrm{min}}_{h,e,t} \widehat{K}_{h,e,t} \leq
     F^{\mathrm{trans}}_{h,e,t}^\mathrm{r} \leq \widehat{K}_{h,e,t}

For storage technologies, the minimum load constraint is formulated as the sum
of the charge and discharge flows as storage technologies do not charge and
discharge at the same time:

.. math::
    :label: min_load_storage

    \ell^{\mathrm{min}}_{h,n,t} \widehat{K}_{h,n,t} \leq
    F^{\mathrm{ch}}_{h,n,t} + F^{\mathrm{dis}}_{h,n,t} \leq \widehat{K}_{h,n,t}

Two more constraints are added to ensure that :math:`\widehat{K}_{h,p,t}`
equals the installed capacity if the technology is on (i.e.,
:math:`z^{\mathrm{on}}_{h,p,t}=1`), and that :math:`\widehat{K}_{h,p,t}` equals zero
if the technology is off (i.e., :math:`z^{\mathrm{on}}_{h,p,t}=0`):

.. math::
    :label: binary_constraint_on

    0 \leq \widehat{K}_{h,p,t} \leq \overline{k}_{h,p,y} z^{\mathrm{on}}_{h,p,t}\\\\
    K_{h,p,y} + (1-z^{\mathrm{on}}_{h,p,t}) \overline{k}_{h,p,y} \leq
    \widehat{K}_{h,p,t} \leq K_{h,p,y}

If no physically motivated capacity limit :math:`\overline{k}_{h,p,y}` exists,
:math:`\overline{k}_{h,p,y}` must be large enough to ensure that the
technology is not constrained by the capacity limit (Big-M parameter).

Minimum full-load hours
-----------------------

.. docstring_method:: zen_garden.elements.conversion_technology.constraints.MinimumFullLoadHoursConstraint.build

This constraint is currently only available for conversion technologies.

.. _math_formulation.min_capacity_installation:

Minimum capacity installation
-----------------------------

A binary variable :math:`z^{\mathrm{install}}_{h,p,y}` is introduced to model the technology
installation decision. If :math:`z^{\mathrm{install}}_{h,p,y}=1`, the technology is installed,
otherwise :math:`z^{\mathrm{install}}_{h,p,y}=0`. The following constraint ensures that if
technology capacity is added, at minimum :math:`\underline{\Delta k}_{h,p,y}`
is installed.

.. math::
    :label: min_capacity_constraint

    \Delta \widehat{K}_{h,p,y} \geq
    \underline{\Delta k}_{h,p,y} z^{\mathrm{install}}_{h,p,y}

where :math:`\Delta\widehat{K}_{h,p,y}` approximates the capacity addition to
avoid bilinearities. The following two constraints link the capacity addition
variable :math:`\Delta K_{h,p,y}` and the approximation of the capacity addition
variable :math:`\Delta\widehat{K}_{h,p,y}`:

.. math::
    :label: min_capacity_constraint_bigM

    \Delta\widehat{K}_{h,p,y} \leq \Delta K_{h,p,y} \\\\
    \Delta\widehat{K}_{h,p,y} \geq
    \left(1-z^{\mathrm{install}}_{h,p,y}\right)\overline{\Delta k}_{h,p,y}
    + \Delta K_{h,p,y}

Eq. :eq:`min_capacity_constraint_bigM` ensure that
:math:`\Delta\widehat{K}_{h,p,y}` equals the capacity addition if the
capacity is expanded (i.e., :math:`z^{\mathrm{install}}_{h,p,y}=1`) and equals
zero otherwise. The big-M value is represented by the maximum capacity addition
for each technology :math:`\overline{\Delta k}_{h,p,y}`.
