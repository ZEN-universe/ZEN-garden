.. _t_infeasibilities.t_infeasibilities:

###############
Infeasibilities
###############

.. admonition:: At a glance
   :class: note

   | **You will** make a model infeasible on purpose, read the termination condition, 
        and use the irreducible inconsistent subsystem to find the cause.
   | **You need** the setup from :ref:`tutorials_intro.setup`.
        Gurobi is required for the IIS section.

An optimization problem is infeasible when no solution satisfies all
constraints at once — some constraints are mutually exclusive. For example:

.. code-block:: text

    x <= y    (I)
    x >= 5    (II)
    y <= 0    (III)

is infeasible, because it requires both ``x >= 5`` and ``x <= y <= 0``.

.. note::
    The objective function does not affect feasibility. Whether you minimize
    ``x``, maximize ``y`` or anything else, ``x >= 5`` and ``x <= 0`` remain
    impossible together. In practice this means **your cost assumptions are
    almost never the source of an infeasibility**, because they only enter the
    objective.


Exercises
=========

.. _t_infeasibilities.break_it:

Step 1: make the model infeasible
----------------------------------

Work on a copy of ``4_multiple_time_steps_per_year``.

Heat demand in this model can only be met by burning natural gas or running a
heat pump, and natural gas has to be imported. Remove the import and the heat
demand cannot be supplied.

1. Open ``set_carriers/natural_gas/attributes.yaml``.
2. Set the default value of ``availability_import`` to ``0``:

   .. code:: yaml

       availability_import:
         default_value: 0
         unit: GW
3. Set the price for shedding heat demand to ``inf`` in
    ``set_carriers/heat/attributes.yaml``:

   .. code:: yaml

       price_shed_demand:
         default_value: inf
         unit: kEuro/GWh
4. Delete ``set_carriers/natural_gas/availability_import.csv``, so
   that nothing overwrites the new default.
5. Remove ``heat_pump`` from ``set_conversion_technologies`` in
   ``system.yaml``, so that the gas boiler is the only source of heat.
6. Set the number of time steps to 1 in ``system.yaml`` to make the problem smaller
   and easier to read:

   .. code:: yaml

       unaggregated_time_steps_per_year: 1

7. Change the solver to ``gurobi`` in ``config.yaml`` to use the IIS feature.
8. Run the model.

.. _t_infeasibilities.read_it:

Step 2: read the termination condition
======================================

A successful run ends with:

.. code-block:: text

    Optimization successful:
    Status: ok
    Termination condition: optimal

Your run should instead end with:

.. code-block:: text

    Optimization failed:
    Status: warning
    Termination condition: infeasible

Sometimes you will see a third case:

.. code-block:: text

    Optimization failed:
    Status: warning
    Termination condition: infeasible_or_unbounded

This means the solver could not determine whether the problem was infeasible or
`unbounded <https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/HTML/chapter3.html?scroll=section3002>`_,
which is often caused by `bad numerics
<https://gurobi.com/documentation/current/refman/guidelines_for_numerical_i.html>`_.

If you are using Gurobi, disable `DualReductions
<https://www.gurobi.com/documentation/current/refman/dualreductions.html>`_ to
get a definite answer. Add to ``config.yaml``:

.. code-block:: yaml

    solver:
      solver_options:
        DualReductions: 0

If the problem then reports ``infeasible``, it really is infeasible. If not,
you most likely have numerical issues — see :ref:`t_scaling.t_scaling`.


.. _t_infeasibilities.iis:

Step 3: find the conflicting constraints
========================================

Finding the source of an infeasibility in a large model is hard: the solver
knows which constraints conflict, but not which parameter value is "right".

Gurobi can compute an `irreducible inconsistent subsystem
<https://docs.gurobi.com/projects/optimizer/en/current/concepts/logging/iis.html>`_
(IIS): a subproblem that is

1. still infeasible, and
2. feasible again as soon as any single constraint or bound is removed.

Suppose the full model contained:

.. code-block:: text

    x <= y    (I)
    x >= 5    (II)
    y <= 0    (III)
    x >= -5          (IV)
    x + y <= 100     (V)
    x + y >= -50     (VI)

Constraints IV–VI do not constrain the problem further, and I–III are already
infeasible on their own. I–VI is the original problem; I–III is the IIS.
Reducing the problem to that subset makes the error far easier to find.

**ZEN-garden writes the IIS automatically** when you use Gurobi and the
termination condition is ``infeasible`` — not ``infeasible_or_unbounded``. Look
for ``infeasible_model_IIS.ilp`` in the output folder of your dataset (it is
the only file written there when the run is infeasible).

The IIS for a minimal two-node model with the same defect looks like this:

.. code-block:: text

    constraint_availability_import:
        [heat, CH, 0]:    1.0 flow_import[heat, CH, 0] <= 0
        [heat, DE, 0]:    1.0 flow_import[heat, DE, 0] <= 0
        [natural_gas, CH, 0]:    1.0 flow_import[natural_gas, CH, 0] <= 0
        [natural_gas, DE, 0]:    1.0 flow_import[natural_gas, DE, 0] <= 0

    constraint_cost_shed_demand:
        [heat, CH, 0]:	1.0 shed_demand[heat, CH, 0] = 0
        [heat, DE, 0]:	1.0 shed_demand[heat, DE, 0] = 0

    constraint_nodal_energy_balance:
        [heat, CH, 0]:	1.0 flow_conversion_output[natural_gas_boiler, heat, CH, 0] + 1.0 flow_import[heat, CH, 0] - 1.0 flow_export[heat, CH, 0] + 1.0 shed_demand[heat, CH, 0] = 10
        [heat, DE, 0]:	1.0 flow_conversion_output[natural_gas_boiler, heat, DE, 0] + 1.0 flow_import[heat, DE, 0] - 1.0 flow_export[heat, DE, 0] + 1.0 shed_demand[heat, DE, 0] = 100

    constraint_carrier_conversion:
        [natural_gas_boiler, natural_gas, CH, 0]:	1.0 flow_conversion_input[natural_gas_boiler, natural_gas, CH, 0] - 1.1 flow_conversion_output[natural_gas_boiler, heat, CH, 0] = 0
        [natural_gas_boiler, natural_gas, DE, 0]:	1.0 flow_conversion_input[natural_gas_boiler, natural_gas, DE, 0] - 1.1 flow_conversion_output[natural_gas_boiler, heat, DE, 0] = 0

Your own IIS will be larger, because the tutorial dataset has transport and storage
technologies. But they do not change that the problem is infeasible.

Read it by asking, for each block: **which of these constraints would I be
willing to relax?**

* Relaxing ``constraint_nodal_energy_balance`` would let demand go unmet —
  possible, but it is the whole point of the model.
* Relaxing ``constraint_carrier_conversion`` would let the boiler produce heat
  from nothing. That constraint is behaving as intended.
* ``constraint_availability_import`` says neither heat nor natural gas may be
  imported (``flow_import <= 0``). That is the constraint you changed, and it
  is the one that is wrong.

The IIS never tells you which constraint is at fault. It narrows the search
from the whole model to a handful of constraints; the modelling judgement is
still yours.

1. **Restore the import availability and confirm the model solves again.**

   *Expected result: the termination condition returns to ``optimal`` and no
   IIS file is written.*

2. **Make the model infeasible a second way: leave imports available, but set
   the** ``capacity_limit`` **of the natural gas boiler to 0 while the heat pump
   is still removed from** ``system.yaml``\ **. Which constraint family appears
   in the IIS this time?**

   *Expected result: the IIS now involves the capacity limit and the technology
   capacity constraints rather than ``constraint_availability_import``. The
   energy balance still cannot be satisfied, but the binding restriction has
   moved from the carrier to the technology, which is exactly the distinction
   the IIS is there to make.*

3. **Use demand shedding to turn an infeasibility into a diagnosis.** Restore
   the boiler capacity limit, remove the natural gas import again, and set
   ``price_shed_demand`` to a large finite value for ``heat``.

   *Expected result: the model becomes feasible, because at worst the optimizer
   can shed all demand. Reading which carrier sheds demand, at which node and
   time step, tells you where the bottleneck is. This is often faster than
   reading an IIS on a large model.*
