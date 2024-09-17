Mathematical formulation
======================

ZEN-garden optimizes the design and operation of energy system models to investigate transition pathways towards decarbonization.
The optimization problem is formulated as a mixed-integer linear program (MILP). In the following, we provide an overview of the objective function and constraints of the optimization problem.

.. _objective-function:
Objective function
-----------------
Two objective functions are available:

1. minimize net present cost
2. minimize total emissions

Minimizing net present cost
^^^^^^^^^^^^^^^^^^^^^

The net present cost :math:`NPC_y` of the energy system are minimized over the entire planning horizon :math:`y \in {\mathcal{Y}}`. 

.. math::
    :label: min_cost

    \sum_{y\in\mathcal{Y}} NPC_y

The net present cost :math:`NPC_y` of each year :math:`y\in\mathcal{Y}` are computed by discounting the total energy system cost of each year, :math:`C_y` with a constant discount rate :math:`r`:

.. math::
    :label: net_present_cost

    NPC_y = \sum_{y \in \mathcal{Y}} \sum_{i \in [0,dy]} \left( \dfrac{1}{1+r} \right)^{left\(dy (y-y_0) + i \right)} C_y

where :math:`y_0` represents the first year of the planning horizon and :math:`dy` represents the interval between planning periods. E.g., if :math:`dy=2` the optimization is conducted for every second year. Moreover, we assume that the optimization is only conducted until the end of the first year of the last planning period. The last period of the planning horizon :math:`Y=\max(y)` is therefore only counted as a single year regardless of the interval between planning periods.

The total cost :math:`C_y` includes the annual capital expenditures :math:`CAPEX_y` and the operational expenditures for operating technologies :math:`OPEX_y^{t}`, importing and exporting carriers :math:`OPEX_y^\mathrm{c}`, and the cost of carbon emissions :math:`OPEX_y^\mathrm{e}`. 

.. math::
    :label: npc

    C_y = CAPEX_y+OPEX_y^\mathrm{t}+OPEX_y^\mathrm{c}+OPEX_y^\mathrm{e}


**Capital expenditures**

:math:`CAPEX_y` accounts for the annual cash flows due to capacity investments :math:`A_{h,p,y}` in technologies:

.. math::
    :label: capex_y

    CAPEX_y = \sum_{h\in\mathcal{H}}\sum_{s\in\mathcal{S}}\sum_{p\in\mathcal{P}} A_{h,p,y}.

Each technology :math:`h\in\mathcal{H}` is either a conversion technology :math:`i\in\mathcal{I}\subseteq\mathcal{H}`, a transport technology :math:`j\in\mathcal{J}\subseteq\mathcal{H}` or a storage technology :math:`k\in\mathcal{K}\subseteq\mathcal{H}`. For sake of simplicity, we index those variables and parameters that apply to all technology types with :math:`h`. For storage capacities, both the energy and power-rated capacity can be expanded. Conversion and storage technologies are installed and operated on nodes :math:`n\in\mathcal{N}`. Transport technologies are installed and operated on edges :math:`e\in\mathcal{E}`. We summarize nodes and edges to positions :math:`p\in\mathcal{P}=\mathcal{N}\cup\mathcal{E}`.

The investment costs are annualized by multiplying the total investment cost with the annuity factor :math:`f_h`, which is a function of the technology lifetime  :math:`l_h` and the discount rate :math:`r`:

.. math::
    :label: annuity

    f_h=\frac{\left(1+r\right)^{l_h}r}{\left(1+r\right)^{l_h}-1}.

The annual cash flows accrue over the technology lifetime :math:`l_h` and comprise the capital investment cost of newly installed and existing technology capacities :math:`I_{h,p,y}` and :math:`I^\mathrm{ex}_{h,p,y}`. The annual capital expenditure :math:`A_{h,p,y}` for technology :math:`h\in\mathcal{H}` in position :math:`p\in\mathcal{P}` and period :math:`y\in\mathcal{Y}` are computed as:

.. math::
    :label: capex_yearly

    A_{h,p,y}= f_h\left(\sum_{\tilde{y}=\max\left(y_0,y-\left(\lceil\frac{l_h}{\Delta^\mathrm{y}}\right)\rceil+1\right)}^y I_{h,p,\tilde{y}} \right)+\left(\sum_{\hat{y}=\psi \left(y-\left(\lceil\frac{l_h}{\Delta^\mathrm{y}}\right)\rceil+1\right)}^{\psi(y_0-1)} I^\mathrm{ex}_{h,p,y}\right),

where :math:`\lceil\cdot\rceil` is the ceiling function and :math:`\psi(y)` is a function that maps the planning period :math:`y` to the actual year.

The capital investment cost :math:`I_{h,p,y}` for conversion technology :math:`i\in\mathcal{I}` is calculated as the product of the unit cost of capital investment :math:`\alpha_{i,y}` and the capacity addition :math:`\Delta S_{i,n,y}` on each node `n\in\mathcal{N}`:

.. math::
    :label: cost_capex_conversion

    I_{i,n,y} = \alpha_{i,y} \Delta S_{i,n,y}

.. note::
    The capex of conversion technologies can also be approximated by a piecewise linear approximation as described in :ref:`PWA` and :ref:`PWA_constraints`.

For existing conversion technology capacities :math:`s_{h,n,y}` that were installed before :math:`y_0`, we apply the unit cost of the first investment period :math:`\alpha_{h,y_0}`:

.. math::
    :label: cost_capex_conversion_ex

    I^\mathrm{ex}_{i,n,y} = \alpha_{i,y_0} \Delta s^\mathrm{ex}_{i,n,y}

For transport technologies :math:`j\in\mathcal{J}`, the unit investment cost :math:`\alpha_{j,e,y}` can be defined through a distance independent unit cost of capital investment :math:`\alpha^\mathrm{const}_{j,y}` and/or a distance dependent unit cost of capital investment :math:`\alpha^\mathrm{dist}_{j,e,y}` which is multiplied by the distance :math:`h_{j,e}` of the corresponding edge :math:`e\in\mathcal{E}`:

.. math::
    :label: unit_cost_capex_transport

    \alpha_{j,e,y} = \alpha^\mathrm{const}_{j,y}+\alpha^\mathrm{dist}_{j,e,y} h_{j,e}

.. note::
    Per default the distance independent unit investment cost is set to zero if a distance dependent cost factor is defined in the input data. To apply both cost terms, set ``double_capex_transport=True`` in your ``system.json``.

The total capital investment cost :math:`A_{h,p,y}` for each transport technology :math:`i\in\mathcal{I}` is calculated as the product of the unit cost of capital investment :math:`\alpha_{j,y}` multiplied by the capacity addition :math:`\Delta S_{j,e,y}` on each edge :math:`e\in\mathcal{E}`:

.. math::
    :label: cost_capex_transport

    I_{j,e,y} = \alpha_{j,e,y} \Delta S_{j,e,y}

For existing transport technology capacities :math:`s_{j,e,y}` that were installed before :math:`y_0`, we apply the unit cost of the first investment period :math:`\alpha_{j,y_0}`:

.. math::
    :label: cost_capex_transport_ex

    I^\mathrm{ex}_{j,e,y} = \alpha_{j,e,y_0} \Delta s^\mathrm{ex}_{j,e,y}

The total investment cost for each storage technology :math:`k\in\mathcal{K}` is the product of the unit cost of capital investment and the capacity addition for both the power-rated capacity (:math:`\alpha_{k,y}` and :math:`\Delta S_{k,n,y}`) and the energy-rated capacity (:math:`\alpha^\mathrm{e}_{k,y}` and :math:`\Delta S^\mathrm{e}_{k,n,y}`).

.. math::
    :label: cost_capex_storage

    I_{k,n,y} = \alpha_{k,y} \Delta S_{k,n,y} + \alpha^\mathrm{e}_{k,y} \Delta S^\mathrm{e}_{k,n,y}

For existing storage technology capacities :math:`s_{k,n,y}` that were installed before :math:`y_0`, we apply the unit cost of the first investment period :math:`\alpha_{k,y_0}` and :math:`\alpha^\mathrm{e}_{k,y_0}`:

.. math::
    :label: cost_capex_storage_ex

    I^\mathrm{ex}_{k,n,y} = \alpha_{k,y_0} \Delta s^\mathrm{ex}_{k,n,y}

**Operational expenditures technology**

The annual operational expenditure for technology operation :math:`OPEX_y^\mathrm{t}` includes the variable operational costs of the technologies :math:`OPEX_y^\mathrm{t,v}` and the fixed operational expenditure for the technology operation :math:`OPEX_y^\mathrm{t,f}`.

.. math::
    :label: opex_t

    OPEX_y^\mathrm{t} = OPEX_y^\mathrm{t,v} + OPEX_y^\mathrm{t,f}.

The fixed technology operational expenditures :math:`OPEX_y^\mathrm{f}` are the product of the specific fixed operational expenditures :math:`\gamma_{h,y}` and the capacity :math:`S_{h,p,y}`, summed over all technologies and positions :math:`p\in\mathcal{P}`:

.. math::
    :label: opex_f

    OPEX_y^\mathrm{f} = \sum_{h\in\mathcal{H}}\sum_{p\in\mathcal{P}}\gamma_{h,y}S_{h,p,y}+\sum_{k\in\mathcal{K}}\sum_{n\in\mathcal{N}}\gamma^\mathrm{e}_{k,y}S^\mathrm{e}_{k,n,y}.

The variable technology operational expenditures :math:`OPEX_y^\mathrm{t,v}` are the sum of the variable operational expenditures for each technology over the entire year, where each timestep is multiplied by the time step duration :math:`\tau_t`:

.. math::
    :label: opex_v

    OPEX_y^\mathrm{t,v} = \sum_{t\in\mathcal{T}}\tau_t \bigg(\sum_{h\in\mathcal{H}} \sum_{s\in\mathcal{S}} \sum_{p\in\mathcal{P}} O^\mathrm{t}_{h,p,t,y} \bigg).

For conversion technologies :math:`i \in \mathcal{I}`, the variable operational expenditure are the product of the specific variable operational expenditure :math:`\beta_{h,y}` and the reference flows :math:`G_{i,n,t,y}^\mathrm{r}`:

.. math:: 
    :label: cost_opex_conversion

    O^\mathrm{t}_{h,\mathrm{power},t,y} = \beta_{i,y} G_{i,n,t,y}^\mathrm{r}

Similarly, for transport technologies :math:`j \in \mathcal{J}`, the variable operational expenditures are the product of the specific variable operational expenditure :math:`\beta_{j,y}` and the reference flows :math:`F_{j,e,t,y}`:

.. math:: 
    :label: cost_opex_transport

    O^\mathrm{t}_{j,\mathrm{power},t,y} = \beta_{j,y} F_{j,e,t,y}

Finally, for storage technologies :math:`k \in \mathcal{K}`, the variable operational expenditure are the product of the charge and discharge cost :math:`\beta^\mathrm{charge}_{j,e,y}` and :math:`\beta^\mathrm{discharge}_{j,e,y}` multiplied by the storage charge :math:`\underline{H}_{k,n,t,y}` and discharge :math:`\overline{H}_{k,n,t,y}`, respectively:

.. math:: 
    :label: cost_opex_storage

    O^\mathrm{t}_{k,\mathrm{power},t,y} = \beta^\mathrm{charge}_{k,y} \underline{H}_{k,n,t,y} + \beta^\mathrm{discharge}_{k,y} \overline{H}_{k,n,t,y}

**Operational expenditures carrier**

The operational carrier cost :math:`OPEX_y^\mathrm{c}` are the sum of the node- and time dependent carrier cost :math:`O^c_{c,n,t,y}` for all carriers multiplied by the time step duration :math:`\tau_t`:

.. math::
    :label: opex_c

    OPEX_y^\mathrm{c} = \sum_{c\in\mathcal{C}}\sum_{n\in\mathcal{N}}\sum_{t\in\mathcal{T}}\tau_t O^c_{c,n,t,y}.

The node- and time dependent carrier cost :math:`O^c_{c,n,t,y}` is composed of three term: 1) the carrier import :math:`U_{c,n,t,y}` multiplied by the import price :math:`u_{c,n,t,y}`, 2) the carrier export :math:`V_{c,n,t,y}` multiplied by the export price :math:`v_{c,n,t,y}`, and 3) the shed demand :math:`D_{c,n,t,y}` multiplied by demand shedding price :math:`\nu_c`:

.. math:: 
    :label: cost_carrier

    O^c_{c,n,t,y} = u_{c,n,t,y}U_{c,n,t,y}-v_{c,n,t,y}V_{c,n,t,y}+\nu_c D_{c,n,t,y}

*Operational expenditures emissions*

The operational emission expenditure :math:`OPEX_y^\mathrm{e}` is composed of three terms: 1) the annual carbon emissions :math:`E_y`  multiplied by the carbon emission price :math:`\mu`, 2) the annual carbon emission overshoot :math:`E_y^\mathrm{o}` multiplied by the annual carbon overshoot price :math:`\mu^\mathrm{o}`, and 3) the budget carbon emission overshoot :math:`E_y^\mathrm{o}` multiplied by the carbon emission budget overshoot price :math:`\mu^\mathrm{o}`:

.. math::
    :label: opex_e

    OPEX_y^\mathrm{e} = E_y \mu + E_y^\mathrm{o}\mu^\mathrm{o}+E_y^\mathrm{bo}\mu^\mathrm{bo}.

.. _emissions_objective
**Minimizing total emissions**

The total annual carbon emissions emissions :math:`E_y` of the energy system are minimized over the entire planning horizon :math:`y \in {\mathcal{Y}}`. 

.. math::
    :label: min_emissions
    \sum_{y\in\mathcal{Y}} E_y

The total annual carbon emissions :math:`E_y` account for the total operational carbon emissions for importing and exporting carriers :math:`E^\mathrm{carrier}_y` and for operating technologies :math:`E^\mathrm{tech}_y`:

.. math::
    :label: total_annual_carbon_emissions
    E_y = E^\mathrm{carrier}_y + E^\mathrm{tech}_y.

For a detailed description of the computation of the total operational emissions for importing and exporting carriers, and for operating for operating technologies refer to :ref:`_tech_carrier_emissions`.

.. _energy_balance:
Energy balance
---------------

The sources and sinks of a carrier :math:`c\in\mathcal{C}` must be in equilibrium for all carriers at all nodes :math:`n\in\mathcal{N}` and in all time steps :math:`t\in\mathcal{T}`. The source terms for carrier :math:`c` on node :math:`n` are:
* the output flow :math:`\overline{G}_{c,i,n,t,y}` of all conversion technologies :math:`i\in\mathcal{I}` if :math:`c\in\overline{\mathcal{C}}_i`.
* the transported flow :math:`F_{j,e,t,y}` on ingoing edges :math:`e\in\underline{\mathcal{E}}_n` minus the losses :math:`F^\mathrm{l}_{j,e,t,y}` for all transport technologies :math:`j\in\mathcal{J}` if :math:`c=c_j^\mathrm{r}`.
* the discharge flow :math:`\overline{H}_{k,n,t,y}` for all storage technologies :math:`k\in\mathcal{K}` if :math:`c=c_k^\mathrm{r}`.
* the imported flow :math:`U_{c,n,t,y}`.

The sinks of carrier :math:`c` on node :math:`n` are:
* the exogenous demand :math:`d_{c,n,t,y}` minus the shed demand :math:`D_{c,n,t,y}`.
* the input flow :math:`\underline{G}_{c,i,n,t,y}` of all conversion technologies :math:`i\in\mathcal{I}` if :math:`c\in\underline{\mathcal{C}}_i`.
* the transported flow :math:`F_{j,e',t,y}` on outgoing edges :math:`e'\in\overline{\mathcal{E}}_n` for all transport technologies :math:`j\in\mathcal{J}` if :math:`c=c_j^\mathrm{r}`.
* the charge flow :math:`\underline{H}_{k,n,t,y}` for all storage technologies :math:`k\in\mathcal{K}` if :math:`c=c_k^\mathrm{r}`.
* the exported flow :math:`V_{c,n,t,y}`.

The energy balance for carrier :math:`c\in\mathcal{C}` is then calculated as:

.. math::
    :label: energy_balance

    0 = -\left(d_{c,n,t,y}-D_{c,n,t,y}\right) + \sum_{i\in\mathcal{I}}\left(\overline{G}_{c,i,n,t,y}-\underline{G}_{c,i,n,t,y}\right) + \sum_{j\in\mathcal{J}}\left(\sum_{e\in\underline{\mathcal{E}}_n}\left(F_{j,e,t,y} - F^\mathrm{l}_{j,e,t,y}\right)-\sum_{e'\in\overline{\mathcal{E}}_n}F_{j,e',t,y}\right) + \sum_{k\in\mathcal{K}}\left(\overline{H}_{k,n,t,y}-\underline{H}_{k,n,t,y}\right)+ U_{c,n,t,y} - V_{c,n,t,y}.

Note that :math:`\sum_{k\in\mathcal{K}}\left(\overline{H}_{k,n,t,y}-\underline{H}_{k,n,t,y}\right)` are zero if :math:`c\neq c^\mathrm{r}_j` and :math:`c\neq c^\mathrm{r}_k`, respectively.

The carrier import :math:`U_{c,n,t,y}` is limited by the carrier import availability :math:`\underline{a}_{c,n,t,y}` for all carriers :math:`c\in\mathcal{C}` in all nodes :math:`n\in\mathcal{N}` and time steps :math:`t\in\mathcal{T}`:

.. math::
    :label: carrier_import

    0 \leq U_{c,n,t,y} \leq \underline{a}_{c,n,t,y}.

In addition, annual carrier import limits can be applied:

.. math::
    :label: carrier_import_yearly

    0 \leq \sum_{t\in\mathcal{T}} \tau U_{c,n,t,y} \leq \underline{a}^{Y}_{c,n,t,y}.

Similarly, the carrier export :math:`V_{c,n,t,y}` is limited by the carrier export availability :math:`\overline{a}_{c,n,t,y}` for all carriers :math:`c\in\mathcal{C}` in all nodes :math:`n\in\mathcal{N}` and time steps :math:`t\in\mathcal{T}`:

.. math::
    :label: carrier_import

    0 \leq V_{c,n,t,y} \leq \overline{a}_{c,n,t,y}.

In addition, annual carrier export limits can be applied:

.. math::
    :label: carrier_export_yearly

    0 \leq \sum_{t\in\mathcal{T}} \tau V_{c,n,t,y} \leq \overline{a}^{Y}_{c,n,t,y}.

.. note:: 
    You can skip the import and export avaialbility constraints by setting the import and export availabilities to infinity.

Lastly, the following constraint ensures that the shed demand :math:`D_{c,n,t,y}` cannot exceed the demand :math:`d_{c,n,t,y}`:

.. math::

    0 \leq D_{c,n,t,y} \leq d_{c,n,t,y}.

.. note::
    Setting the shed demand cost to infinity forces :math:`D_{c,n,t,y}=0` and demand shedding will not be possible.

.. _emissions_constraints:
Emissions constraints
-----------------------

The total annual carrier carbon emissions :math:`E^\mathrm{carrier}_y` represent the sum of the carrier carbon emissions :math:`\theta^\mathrm{carrier}_{c,n,t,y}`:

.. math::
    :label: total_carbon_emissions_carrier

    E^\mathrm{carrier}_y = \sum_{t\in\mathcal{T}} \sum_{n\in\mathcal{N}} \sum_{c\in\mathcal{C}} \left( \theta^\mathrm{carrier}_{c,n,t,y} \tau_t \right).

The carrier carbon emissions include the operational emissions of importing and exporting carriers :math:`c\in\mathcal{C}` (carbon intensity :math:`\underline{\epsilon_c}` and :math:`\overline{\epsilon_c}`):

.. math::
    :label: carbon_emissions_carrier

    \theta^\mathrm{carrier}_{c,n,t} = \underline{\epsilon_c} U_{c,n,t,y} - \overline{\epsilon_c} V_{c,n,t,y}.
    
The total annual technology carbon emissions :math:`E^\mathrm{tech}_y` represent the sum of the technology carbon emissions :math:`\theta^\mathrm{tech}_{h,n,t,y}`:

.. math::
    :label: total_carbon_emissions_technology
    E^\mathrm{tech}_y = \sum_{t\in\mathcal{T}} \sum_{n\in\mathcal{N}} \sum_{h\in\mathcal{H}} \left( \theta^\mathrm{tech}_{h,n,t,y} \tau_t \right).

The technology carbon emission :math:`\theta^\mathrm{tech}_{h,n,t,y}` include the emissions for operating the technologies :math:`h\in\mathcal{H}` (carbon intensity :math:`\epsilon_h`). For conversion technologies :math:`i\in\mathcal{I}`, the carbon intensity of operating the technology is multiplied with the reference flows :math:`G_{i,n,t,y}^\mathrm{r}`:

.. math::
    :label: carbon_emissions_conversion

    \theta^\mathrm{tech}_{i,n,t,y} =  \epsilon_i G_{i,n,t,y}^\mathrm{r}.

For storage technologies :math:`k\in\mathcal{K}`, the carbon intensity of operating the technology is multiplied with the storage charge and discharge flows :math:`\overline{H}_{k,n,t,y}` and :math:`\overline{H}_{k,n,t,y}`:
    
.. math::
    :label: carbon_emissions_storage
    \theta^\mathrm{tech}_{k,n,t,y} =  \epsilon_k \left( \overline{H}_{k,n,t,y}+\underline{H}_{k,n,t,y} \right).

Finally, for transport technologies :math:`j\in\mathcal{J}`, the carbon intensity of operating the technology is multiplied with the transported flow :math:`F_{j,e,t,y}`:

.. math::
    :label: carbon_emissions_transport

    \theta^\mathrm{tech}_{k,n,t,y} = \epsilon_j F_{j,e,t,y}.

The annual carbon emissions are limited :math:`E_y` by the annual carbon emissions limit :math:`e_y`:

.. math::
    :label: carbon_emissions_annual_limit

    E_y - E_{y}^\mathrm{bo} \leq e_y.

Note that :math:`e_y` can be infinite, in which case the constraint is skipped.

:math:`E_{y}^\mathrm{o}` is the annual carbon emission limit overshoot and allows exceeding the annual carbon emission limits. However, overshooting the annual carbon emission limits is penalized in the objective function (:eq:`opex_c`). This overshoot cost is computed by multiplying the annual carbon emission limit overshoot :math:`E_{y}^\mathrm{o}`with the annual carbon emission limit overshoot price :math:`\mu_1\mathrm{o}`. To strictly enforce the annual carbon emission limit (i.e.,:math:`E_{y}^\mathrm{bo}=0`), an infinite carbon overshoot price :math:`\mu_1\mathrm{o}` must be used.

The cumulative carbon emissions :math:`E_y^\mathrm{cum}` are attributed to the end of the current year. For the first planning period :math:`y=y_0`, :math:`E_y^\mathrm{cum}` is calculated as:

.. math::
    :label: carbon_emissions_cum_0

    E_y^\mathrm{cum} = E_y.

In the subsequent periods :math:`y>y_0`, :math:`E_y^\mathrm{c}` is calculated as:

.. math::
    :label: carbon_emissions_cum_1

    E_y^\mathrm{c} = E_{y-1}^\mathrm{c} + \left(\Delta^\mathrm{y}-1\right)E_{y-1}+E_y.

The cumulative carbon emissions :math:`E_y^\mathrm{c}` are constrained by the carbon emission budget :math:`e^\mathrm{b}`:

.. math::
    :label: emission_budget

    E_y^\mathrm{cum} + \left(\Delta^\mathrm{y}-1\right)E_{y}  - E_{y}^\mathrm{bo} \leq e^\mathrm{b}.

Note that :math:`e^\mathrm{b}` can be infinite, in which case the constraint is skipped. :math:`E_y^\mathrm{o}` is the cumulative carbon emission overshoot and allows exceeding the carbon emission budget :math:`e^\mathrm{b}`. However, exceeding the carbon emission budget in the last year of the planning horizon :math:`\mathrm{Y}=\max(y)` (i.e., :math:`E_\mathrm{Y}^\mathrm{o}>0`) is penalized with the carbon emissions budget overshoot price :math:`\mu^\mathrm{bo}` in the objective function (:eq:`opex_c`).

By setting the carbon emission budget overshoot price to infinite, we enforce that the cumulative carbon emissions stay below the carbon emission budget :math:`e^\mathrm{b}` across all years (i.e.,:math:`E_\mathrm{y}^\mathrm{o}=0, \forall y\in\mathcal{Y}`). By setting the carbon emission budget overshoot price to a real number, we allow overshooting a carbon emission budget overshoot throughout the transition, with only the carbon budget overshoot in the last year of the planning horizon being penalized in the objective function.

.. _operational_constraints:
Operational constraints
-----------------------

The conversion factor :math:`\eta_{i,c,t,y}` describes the ratio between the carrier flow :math:`c\in\mathcal{C}` and the reference carrier flow :math:`G_{i,n,t,y}^\mathrm{r}` of a conversion technology :math:`i\in\mathcal{I}`. If the carrier flow is an input carrier, i.e. :math:`c\in\underline{\mathcal{C}}_i`:

.. math::

    \eta_{i,c,t,y} = \frac{\underline{G}_{c,i,n,t,y}}{G_{i,n,t,y}^\mathrm{r}}.

If the carrier flow is an output carrier, i.e. :math:`c\in\overline{\mathcal{C}}_i`:

.. math::

    \eta_{i,c,t,y} = \frac{\overline{G}_{c,i,n,t,y}}{G_{i,n,t,y}^\mathrm{r}}.

The transport flow losses :math:`F_{j,e,t,y}^\mathrm{l}` through a transport technology :math:`j\in\mathcal{J}` on edge :math:`e\in\mathcal{E}` are expressed by the loss function :math:`\rho_{j,e}` and the transported quantity:

.. math::

    F_{j,e,t,y}^\mathrm{l} = \rho_{j,e} h_{j,e} F_{j,e,t,y}.

The loss function is described through a linear or exponential loss factor, :math:`\rho^\mathrm{lin}_{j}` and :math:`\rho^\mathrm{exp}_{j}`, respectively. The loss factor is applied to the transport distance :math:`h_{j,e}``. For transport technologies where transport flow losses are approximated by a linear loss factor it follows:

.. math::

    \rho_{j,e} =  h_{j,e}^{\rho^\mathrm{exp}_{j,e}}

For transport technologies where transport flow losses are approximated by an exponential loss factor it follows:

.. math::

    \rho_{j,e} = h_{j,e} \rho^\mathrm{exp}_{j,e}

**JM START**
The temporal representation of storage technologies :math:`k\in\mathcal{K}` is particular because the storage constraints are time-coupled and the sequence of time steps must be preserved. To enable both the modeling of short- and medium-term storage, e.g., pumped hydro storage, and long-term storage, e.g., natural gas storage, we present a novel formulation, where the energy-rated storage variables are resolved on a different time sequence. In particular, each change in the aggregated time sequence for power-rated variables yields an additional time step for the energy-rated storage variables. Assume the representation of the exemplary full time index :math:`\mathcal{T}^\mathrm{full}=[0,...,9]` by four representative time steps :math:`\mathcal{T}=[0,...,3]` with the sequence :math:`\sigma` for power-rated variables:

.. math::

    \sigma = [0,0,1,2,1,1,3,3,2,0].

The resulting sequence for energy-rated storage variables :math:`\sigma^\mathrm{k}:math:` of the storage time steps :math:`\mathcal{T}^\mathrm{k}=[0,...,6]` is then:

.. math::

    \sigma^\mathrm{k} = [0,0,1,2,3,3,4,4,5,6].

While this formulation enables both the short-term and long-term operation of storages, it increases the number of time steps :math:`\vert \mathcal{T}^\mathrm{k}\vert` and thus the number of variables.

For sake of simplicity, let :math:`\sigma:\mathcal{T}^\mathrm{k}\to \mathcal{T}` denote the unique mapping of a storage level time step :math:`t^\mathrm{k}` to a power-rated time step :math:`t`.
The time-coupled equation for the storage level :math:`L_{k,n,t^\mathrm{k},y}` of storage technology :math:`k` at node :math:`n` is formulated for each storage level time step except the first :math:`t^\mathrm{k}\in\mathcal{T}^\mathrm{k}\setminus\{0\}` as:

.. math::
    :label: storage_level

    L_{k,n,t^\mathrm{k},y} = L_{k,n,t^\mathrm{k}-1,y}\left(1-\varphi_k\right)^{\tau^\mathrm{k}_{t^\mathrm{k}}}+\left(\underline{\eta}_k\underline{H}_{k,n,\sigma(t^\mathrm{k}),y}-\frac{\overline{H}_{k,n,\sigma(t^\mathrm{k}),y}}{\overline{\eta}_k}\right)\sum_{\tilde{t}^\mathrm{k}=0}^{\tau^\mathrm{k}_{t^\mathrm{k}}-1}\left(1-\varphi_k\right)^{\tilde{t}^\mathrm{k}},

with the self-discharge rate :math:`\varphi_k`, the charge and discharge efficiency :math:`\underline{\eta}_k` and :math:`\overline{\eta}_k` and the duration of a storage level time step :math:`\tau^\mathrm{k}_{t^\mathrm{k}}`.
If storage periodicity is enforced, the storage level at :math:`t^\mathrm{k}=0` is coupled with the level in the last time step of the period
:math:`t^\mathrm{k}=T^\mathrm{k}`:

.. math::

    L_{k,n,0,y} = L_{k,n,T^\mathrm{k},y}\left(1-\varphi_k\right)^{\tau^\mathrm{k}_{t^\mathrm{k}}}+\left(\underline{\eta}_k\underline{H}_{k,n,\sigma(0),y}-\frac{\overline{H}_{k,n,\sigma(0),y}}{\overline{\eta}_k}\right)\sum_{\tilde{t}^\mathrm{k}=0}^{\tau^\mathrm{k}_{t^\mathrm{k}}-1}\left(1-\varphi_k\right)^{\tilde{t}^\mathrm{k}}.

The non-negative :math:`L_{k,n,t^\mathrm{k},y}` is constrained by the energy-rated storage capacity :math:`S^\mathrm{e}_{k,n,y}`:

.. math::
    :label:limit_storage_level

    0\leq L_{k,n,t^\mathrm{k},y}\leq S^\mathrm{e}_{k,n,y}.

:math:`L_{k,n,t^\mathrm{k},y}` is monotonous between :math:`t^\mathrm{k}` and :math:`t^\mathrm{k}+1`. Hence, :math:`L_{k,n,t^\mathrm{k},y}` and :math:`L_{k,n,t^\mathrm{k}+1,y}` are the local extreme values and :eq:`limit_storage_level` constrains the entire time interval between :math:`t^\mathrm{k}` and :math:`t^\mathrm{k}+1`. We prove this in :eq:`subsec:proof_storage`.

The storage level at :math:`t^\mathrm{k}=0` can be set to an initial storage level :math:`\chi_{k,n}` as a share of :math:`S^\mathrm{e}_{k,n,y}`:

.. math::

    L_{k,n,0,y} = \chi_{k,n}S^\mathrm{e}_{k,n,y}.

**JM STOPP**

The flow of the reference carrier :math:`c_h^\mathrm{r}` of all technologies :math:`h\in\mathcal{H}` is constrained by the maximum load :math:`m^\mathrm{max}_{h,p,t,y}` and the installed capacity :math:`S_{h,p,y}`. For conversion technologies :math:`i\in\mathcal{I}`, it follows:

.. math::

    0 \leq G_{i,n,t,y}^\mathrm{r} \leq m^\mathrm{max}_{i,n,t,y}S_{i,n,y}.

Analogously for transport technologies :math:`j\in\mathcal{J}` it follows:

.. math::

    0 \leq F_{j,e,t,y} \leq m^\mathrm{max}_{j,e,t,y}S_{j,e,y}.

Since a storage technology does not charge (:math:`\underline{H}_{k,n,t,y}`) and discharge (:math:`\overline{H}_{k,n,t,y}`) at the same time, the sum of both flows is constrained by the maximum load:

.. math::

    0 \leq \underline{H}_{k,n,t,y}+\overline{H}_{k,n,t,y}\leq m_{k,n,t,y}S_{k,n,y}.

In addition, minimum load constraints can be added. Please note, that adding minimum load constraints :math:`m^\mathrm{min}_{h,p,t,y}` introduces binary variables, which can increase the computational complexity of the optimization problem substantially. The min-load constraints are described in :ref:`_min_load_constraints`.

Finally, the reference flow of retrofitting technologies is linked to the reference flow of their base technology. The set of base technologies links each retrofitting technology :math:`rt` to their base technology :math:`bt`. The retrofit flow coupling factor can be interpreted as a conversion factor :math:`\eta^\mathrm{retrofit}_{rt,bt}` that describes the ratio between the reference flow of the retrofitting technology and the reference flow of the base technology:

.. math::

    G_{i,n,rt,y}^\mathrm{r} = \eta^\mathrm{retrofit}_{rt,bt} G_{i,n,bt,y}^\mathrm{r}.

Investment constraints
----------------------

The capacity :math:`S_{h,p,y}` of a technology :math:`h\in\mathcal{H}` at a position :math:`p\in\mathcal{P}` in period :math:`y` is the sum of all previous capacity additions :math:`\Delta S_{h,p,y}` and existing capacities :math:`\Delta s^\mathrm{ex}_{h,p,y}`, that are still within their usable technical lifetime :math:`l_h` (compare :eq:`annuity`):

.. math::
    :label: capacity

    S_{h,p,y}=\sum_{\tilde{y}=\max\left(y_0,y-\left\lceil\frac{l_h}{\Delta^\mathrm{y}}\right\rceil+1\right)}^y \Delta S_{h,p,\tilde{y}}+\sum_{\hat{y}=\psi\left(\min\left(y_0-1,y-\left\lceil\frac{l_h}{\Delta^\mathrm{y}}\right\rceil+1\right)\right)}^{\psi(y_0)} \Delta s^\mathrm{ex}_{h,p,\hat{y}}.

:math:`S_{h,p,y}` is constrained by the capacity limit :math:`s^\mathrm{max}_{h,p,y}`:

.. math::

    S_{h,p,y} \leq s^\mathrm{max}_{h,p,y}.

The capacity addition is constrained by the minimum and maximum capacity addition :math:`\Delta s^\mathrm{min}_{h,p,y}` and :math:`\Delta s^\mathrm{max}_{h,p,y}`:

.. math::

    \Delta s^\mathrm{min}_{h,p,y} \leq \Delta S_{h,p,y}

.. math::

    \Delta S_{h,p,y} \leq \Delta s^\mathrm{max}_{h,p,y}

.. note::

    You can skip the maximum capacity addition constraint for a technology by setting the maximum capacity addition to infinity.

Furthermore, for storage technologies the ratios of the energy- and power rated capacity additions are constrained by the energy-to-power ratio :math:`\rho_{k}`. Minimum and maximum energy-to-power ratios can be defined. For infinite power ratios, the constraints are skiped.

.. math::

    \rho^\mathrm{min}_{k} S_{k,\mathrm{energy},n,y} \leq \S_{k,\mathrm{power},n,y}

.. math::

    \rho^\mathrm{max}_{k} S_{k,\mathrm{energy},n,y} \geq \S_{k,\mathrm{power},n,y}

To account for technology construction times :math:`dy^mathrm{construction}` we introduce an auxiliary variable, :math:`S_{h,p,y` representing the technology investments. The following constraint ensures that the new technology capacities do not become available before the construction time has passed:

.. math::

    \Delta S_{h,p,y} = S_{h,p,y-dy^mathrm{construction}}^\mathrm{invest}

Furthermore, if :math:`y-dy^mathrm{construction}}^\mathrm{invest}<0`:

.. math::

    S_{h,p,y} = 0

**JM START**
In case you are using constrained technology deployment, :math:`\Delta S_{h,p,y}` is constrained by the existing knowledge of how to install the technology :math:`K_{h,p,y}` with the technology diffusion rate :math:`\vartheta_h`. For node-based technologies, i.e., conversion and storage technologies, spillover effects from other nodes :math:`\tilde{\mathcal{N}} = \mathcal{N}\setminus\{n\}` can be utilized (knowledge spillover rate :math:`\omega`). To allow for an entry into a niche market, we add an unbounded market share :math:`\xi` of the total capacity of all other technologies with the same reference carrier:

.. math::

    \tilde{\mathcal{H}}=\Set{\tilde{h}\in\mathcal{H}\setminus\{h\} \mid c_{\tilde{h}}^\mathrm{r} = c_{h}^\mathrm{r}}

With the unbounded capacity addition :math:`\zeta_h`, it follows for the conversion technologies :math:`i\in\mathcal{I}`:

.. math::

    0 \leq \Delta S_{i,n,y}\leq \left((1+\vartheta_i)^{\Delta^\mathrm{y}}-1\right)\left(K_{i,n,y}+\omega\sum_{\tilde{n}\in\tilde{\mathcal{N}}}K_{i,\tilde{n},y}\right)+\Delta^\mathrm{y}\left(\xi\sum_{\tilde{i}\in\tilde{\mathcal{I}}}S_{\tilde{i},n,y} + \zeta_i\right).

Analogously, it follows for the storage technologies :math:`k\in\mathcal{K}`:

.. math::

    0 \leq \Delta S_{k,n,y}\leq \left((1+\vartheta_k)^{\Delta^\mathrm{y}}-1\right)\left(K_{k,n,y}+\omega\sum_{\tilde{n}\in\tilde{\mathcal{N}}}K_{k,\tilde{n},y}\right)+\Delta^\mathrm{y}\left(\xi\sum_{\tilde{k}\in\tilde{\mathcal{K}}}S_{\tilde{k},n,y} + \zeta_k\right).


We prohibit spillover effects for transport technologies :math:`j\in\mathcal{J}` from other edges:

.. math::

    0 \leq \Delta S_{j,e,y}\leq \left((1+\vartheta_j)^{\Delta^\mathrm{y}}-1\right)K_{j,e,y}+\Delta^\mathrm{y}\left(\xi\sum_{\tilde{j}\in\tilde{\mathcal{J}}}S_{\tilde{j},e,y} + \zeta_j\right).


To avoid the unrealistically excessive use of spillover effects, we constrain the capacity additions in all positions as follows:

.. math::

    \sum_{p\in\mathcal{P}}\Delta S_{h,p,y}\leq \sum_{p\in\mathcal{P}}\Bigg(\left((1+\vartheta_h)^{\Delta^\mathrm{y}}-1\right)K_{h,p,y}+\Delta^\mathrm{y}\left(\xi\sum_{\tilde{h}\in\tilde{\mathcal{H}}}S_{\tilde{h},p,y} + \zeta_h\right)\Bigg).


:math:`K_{h,p,y}` is a function of the previous capacity additions :math:`\Delta S_{h,p,y}` and :math:`\Delta s^\mathrm{ex}_{h,p,y}` as it represents the expertise and knowledge of the industry on how to install a certain amount of capacity. This knowledge is depreciated over time with the knowledge depreciation rate :math:`\delta`:

.. math::

    K_{h,p,y} = \sum_{\tilde{y}=y_0}^{y-1}\left(1-\delta\right)^{\Delta^\mathrm{y}(y-\tilde{y})}\Delta S_{h,p,\tilde{y}} + \sum_{\hat{y}=-\infty}^{\psi(y_0)}\left(1-\delta\right)^{\left(\Delta^\mathrm{y}(y-y_0) + (\psi(y_0)-\hat{y})\right)}\Delta s^\mathrm{ex}_{h,p,\hat{y}}.

.. _storage_level_monotony:
**Proof of storage level monotony**

We prove that :eq:`storage_level` is monotonous on the entire time interval that is aggregated to a single storage time step :math:`t^\mathrm{k}`.
Consider :eq:`storage_level` for one storage time step :math:`t^\mathrm{k}`, during which :math:`\underline{H}_{k,n,\sigma(t^\mathrm{k}),y}` and :math:`\overline{H}_{k,n,\sigma(t^\mathrm{k}),y}` are constant.
Neglecting all further indices without loss of generality, the storage level :math:`L(t)` for the intermediate time steps :math:`t\in[1,\tau^\mathrm{k}_{t^\mathrm{k}}]` follows as:

.. math::
    :label: storage_level_simpl

    L(t) = L_0\kappa^t + \Delta H\sum_{\tilde{t}=0}^{t-1}\kappa^{\tilde{t}},

with :math:`\kappa=1-\varphi` and :math:`\Delta H=\left(\underline{\eta}\underline{H}-\frac{\overline{H}}{\overline{\eta}}\right)`. :math:`L_0` is the storage level at the end of the previous storage time step :math:`t^\mathrm{k}-1`.
Without self-discharge (:math:`\varphi=0\Rightarrow\kappa=1`), it follows:

.. math::

    L(t) = L_0 + \Delta Ht \Rightarrow \dv{L(t)}{t}=\Delta H.

Since :math:`\dv*{L(t)}{t}` is independent of :math:`t`, :eq:`storage_level_simpl` is monotonous for :math:`\varphi=0`.

For :math:`0<\varphi<1`, :math:`\sum_{\tilde{t}=0}^{t-1}\kappa^{\tilde{t}}` is reformulated as the partial geometric series:

.. math::

    \sum_{\tilde{t}=0}^{t-1}\kappa^{\tilde{t}} = \frac{1-\kappa^t}{1-\kappa}.

:eq:`storage_level_simpl` is reformulated to:

.. math::

    label: storage_level_selfdisch
    L(t) = L_0\kappa^t + \Delta H\frac{1-\kappa^t}{1-\kappa} = \frac{\Delta H}{1-\kappa}+\left(L_0-\frac{\Delta H}{1-\kappa}\right)\kappa^t.

The derivative of :eq:`storage_level_selfdisch` follows as:

.. math::

    \dv{L(t)}{t} = \underbrace{\left(L_0-\frac{\Delta H}{1-\kappa}\right)\ln(\kappa)}_{= \text{ constant }\forall t\in[1,\tau^\mathrm{k}_{t^\mathrm{k}}]}\kappa^t.

With :math:`\kappa^t>0`, it follows that \cref{eq:storage_level_simpl} is monotonous for :math:`0<\varphi<1`.
**JM STOPP**

.. _min_load_constraints:
Minimum load constraints
------------------------

For conversion technologies, the minimum load constraint can be formulated as follows:

.. math::
    :label: min_load_conversion_bilinear

    b_{h,p,t} m^\mathrm{min}_{h,n,t,y} S_{h,n,y} \leq G_{h,n,t,y}^\mathrm{r}

where :math:`b_{h,p,t}` represents a binary variable that is one if the technology is on and zero if the technology is off. However, this constraint would introduce a bilinearity. To resolve the bilinearity, we use a big-M formulation and approximate the capacity variable by :math:`S^\mathrm{approx}_{h,p,y}`. With that, :eq:`min_load_conversion_bilinear` is rewritten as:

.. math::
    :label: min_load_conversion

    G_{h,n,t,y}^\mathrm{r} \geq m^\mathrm{min}_{h,n,t,y} S^\mathrm{approx}_{h,n,y}

Furthermore, the following two constraints are added to ensure that the approximated capacity equals the installed capacity if the technology is on (i.e., :math:`b_{h,p,t}=1`), and is zero if the technology is off (i.e., :math:`b_{h,p,t}=0`):

.. math::
    :label: binary_constraint_on

    S^\mathrm{approx}_{i,n,y} \leq S_{i,n,y
    S^\mathrm{approx}_{i,n,y} \geq (1-b_{h,p,t}) M + S_{i,n,t}

Similarly, for transport technologies it follows:

.. math::
    :label: min_load_transport

    F_{j,e,t,y}^\mathrm{r} \geq m^\mathrm{min}_{j,n,t,y} S^\mathrm{approx}_{j,e,y}

For storage technologies, the minimum load constraint is formulated as the sum of the charge and discharge flows as storage technologies do not charge and discharge at the same time:

.. math::
    :label: min_load_storage

    \underline{H}_{k,n,t,y} + \overline{H}_{k,n,t,y} \geq m^\mathrm{min}_{k,n,t,y} S^\mathrm{approx}_{k,e,y}

.. _PWA_constraints:
Piecewise affine approximation of captial expenditures
-----------------------------------------------------

The capital expenditures of the conversion technologies can be approximated by a piecewise affine (PWA) function to account for non-linearities and e.g., represent economies of scale. To this end, the captial investment unit costs are approximated by segments with different capital unit investment costs. The number of segments is described via the set of segments :math:`x\in\mathcal{X}`. To model the segement selection, the binary variable :math:`d_{i,n,y,x}` is introduced. :math:`d_{i,n,y,x}` equals one if segement :math:`x` is active, otherwise :math:`d_{i,n,y,x}` equals zero. At most, one segment can be active at a time:

.. math::

    \sum_{x\in\mathcal{X}} d_{i,n,y,x} \leq 1

Each segment is valid withing a minimum and maximum capacity :math:`s^\mathrm{pwa}_{i,n,y,x}`. Furthermore, :math:`A_{i,p,y}^\mathrm{approx}` is introduced to approximate the capacity variable and avoid non-linearities.

For the first segment (i.e., :math:`x=0`), the technology capacitiy is described by the following constraints:

.. math:: 

    0 \leq S_{i,n,y,x}^\mathrm{approx}
    S_{i,n,y}^\mathrm{approx} \geq d_{i,n,y,x} s^\mathrm{pwa}_{i,n,y,x}

For all other segments (i.e., :math:`x>0`), the following constraints must hold:

.. math:: 

    d_{i,n,y,x} s^\mathrm{pwa}_{i,n,y,x-1} \leq S_{i,n,y,x}^\mathrm{approx}
    S_{i,n,y}^\mathrm{approx} \geq d_{i,n,y,x} s^\mathrm{pwa}_{i,n,y,x}

The capacity approximation variable :math:`S_{i,n,y,x}^\mathrm{approx}` and the capacity variables :math:`S_{i,n,y}` are linked:

.. math::

    \sum_{x\in\mathcal{X}} S_{i,n,y,x}^\mathrm{approx} = S_{i,n,y}

Finally, the captial expenditures are described by the following constraint:

.. math::
    
    A_{i,p,y} = \sum_{x\in\mathcal{X}} \alpha_{i,y,x} S_{i,n,y,x}^\mathrm{approx}