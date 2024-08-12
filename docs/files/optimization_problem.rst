################
Optimization problem
################

ZEN-garden optimizes the design and operation of energy system models to investigate transition pathways towards decarbonization.
The optimization problem is formulated as a mixed-integer linear program (MILP). In the following, we provide a brief overview of the optimization problem and the main components of the model.

.. _objective-function:
Objective function
=================
The objective of the optimization problem :math:`J` is to minimize the net present cost :math:`NPC_y` over the entire planning horizon :math:`y \in {\mathcal{Y}}`. :math:`NPC_y` includes the annual capital expenditures :math:`CAPEX_y` and the operational expenditures :math:`OPEX_y`, and accounts for the interval between planning periods :math:`\Delta^y`. The last period of the planning horizon :math:`Y=\max(y)` is only counted as a single year since we assume that the optimization is only conducted until the end of the first year of the last planning period. The future cash flows are discounted with a constant discount rate of :math:`r=6 \%`:

.. math::

    \begin{align}
        J = &\sum_{y\in\mathcal{Y}}NPC_y \\
        =&\sum_{y\in\mathcal{Y}}^{Y-1}\sum_{\tilde{y} = 0}^{\Delta^\mathrm{y}-1}\left(\frac{1}{1+r}\right)^{\Delta^\mathrm{y}(y-y_0)+\tilde{y}}\left(CAPEX_y+OPEX_y\right)+\nonumber\\
        &\left(\frac{1}{1+r}\right)^{\Delta^\mathrm{y}(Y-y_0)}\left(CAPEX_Y+OPEX_Y\right).\nonumber
    \end{align}`

.. _technologies:


