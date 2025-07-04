###########################
Tutorial 4: Infeasibilities
###########################

Here, we briefly want to introduce several important concepts around 
infeasibility.

What is an infeasible model?
-----------------------------

An optimization problem is infeasible if no feasible solution can be found. That 
means that some constraints are mutually exclusive. Assume an optimization 
problem with the variables ``x`` and ``y`` and the following constraints:

.. code-block::

    x <= y    (I)
    x >= 5    (II)
    y <= 0    (III)

This problem is obviously infeasible because it enforces ``x >= 5`` and 
``x <= y <= 0``.

.. note::
    It is important to note that the objective function does not impact 
    feasibility! It does not matter if our objective function was ``min x``, 
    ``max y``, or ``min x**2-2*y**5``; ``x >= 5`` and ``x <= 0`` is always 
    impossible. In practical terms, this means that your cost assumptions are 
    most probably not the source of the infeasibility, because they mainly 
    impact the objective function.

How do I see that my model is infeasible?
------------------------------------------

The output of your optimization tells you the termination condition of your 
problem. A successful optimization looks like this:

.. code-block::

    Optimization successful:
    Status: ok
    Termination condition: optimal

whereas an infeasible problem has this output:

.. code-block::

    Optimization failed:
    Status: warning
    Termination condition: infeasible

Sometimes you will see this message:

.. code-block::

    Optimization failed:
    Status: warning
    Termination condition: infeasible_or_unbounded

That indicates that the optimizer could not determine whether the problem was, 
in fact, infeasible or `unbounded <https://www.fico.com/fico-xpress-optimization/docs/latest/solver/optimizer/HTML/chapter3.html?scroll=section3002>`_.
This can be due to `bad numerics <https://gurobi.com/documentation/current/refman/guidelines_for_numerical_i.html>`_.

To find out whether your problem is actually infeasible or has bad numerics and 
**you are using Gurobi**, you can disable the 
`DualReductions <https://www.gurobi.com/documentation/8.1/refman/dualreductions.html#parameter:DualReductions>`_ parameter. Just add this line to the ``config.json``:

.. code-block::

    "solver": {
        "solver_options": {
          "DualReductions": 0
        }
    }

This should give you a definite answer if your problem is infeasible. If not, 
you most probably have numerical issues.

How can I find out what constraint led to the infeasibility?
------------------------------------------------------------

Finding the source of the infeasibility can become tricky, especially in large 
models with a lot of parameters and constraints. You will need to use your 
knowledge of your own model to understand where you made a mistake. 
Unfortunately, the solver does not know which parameter value is "right" and 
which one is "wrong", it only knows that some constraints are conflicting.

Fortunately, Gurobi has a fantastic tool that is helpful in finding the 
conflicting constraints that make the problem infeasible: The `Irreducible 
Inconsistent Subsystem <https://www.gurobi.com/documentation/current/refman/py_model_computeiis.html>`_ (IIS). 
The IIS is a subproblem of the original problem with two properties:

1. It is still infeasible, and
2. If a single constraint or bound is removed, the subsystem becomes feasible.

Take the original example from the beginning and let's assume there were 
additional constraints:

.. code-block::

    x <= y    (I)
    x >= 5    (II)
    y <= 0    (III)
    ...
    x >= -5    (IV)
    x + y <= 100     (V)
    x + y >= -50     (VI)

Constraints (IV, V, and VI) do not further constrain the problem, and (I, II, 
and III) are already infeasible. So, Constraints I-VI could be the original 
problem and I-III the corresponding IIS. Reducing the problem to this subset of 
constraints makes finding the error significantly easier. Always ask yourself  
the question: Which of these constraints do I want to remove to make the problem 
feasible again?

How is the IIS handled in ZEN-garden?
--------------------------------------

In ZEN-garden, we automatically write the IIS if you are using Gurobi and the 
termination condition is infeasible (not infeasible_or_unbounded!). **It is 
written to the output folder of the dataset.**

Take the following example, which is ``test_1a`` but without the option to 
import natural gas. Clearly, if no gas can be imported, the heat demand cannot 
be supplied and the problem becomes infeasible. The resulting IIS is the 
following:

.. code-block::

    constraint_availability_import:
        [heat, CH, 0]:    1.0 flow_import[heat, CH, 0] <= 0
        [heat, DE, 0]:    1.0 flow_import[heat, DE, 0] <= 0
        [natural_gas, CH, 0]:    1.0 flow_import[natural_gas, CH, 0] <= 0
        [natural_gas, DE, 0]:    1.0 flow_import[natural_gas, DE, 0] <= 0

    constraint_cost_shed_demand:
        [heat, CH, 0]:	1.0 shed_demand[heat, CH, 0] = 0
        [heat, DE, 0]:	1.0 shed_demand[heat, DE, 0] = 0
        [natural_gas, CH, 0]:	1.0 shed_demand[natural_gas, CH, 0] = 0
        [natural_gas, DE, 0]:	1.0 shed_demand[natural_gas, DE, 0] = 0

    constraint_nodal_energy_balance:
        [heat, CH, 0]:	1.0 flow_conversion_output[natural_gas_boiler, heat, CH, 0] + 1.0 flow_import[heat, CH, 0] - 1.0 flow_export[heat, CH, 0] + 1.0 shed_demand[heat, CH, 0] = 10
        [heat, DE, 0]:	1.0 flow_conversion_output[natural_gas_boiler, heat, DE, 0] + 1.0 flow_import[heat, DE, 0] - 1.0 flow_export[heat, DE, 0] + 1.0 shed_demand[heat, DE, 0] = 100
        [natural_gas, CH, 0]:	-1.0 flow_conversion_input[natural_gas_boiler, natural_gas, CH, 0] + 1.0 flow_transport[natural_gas_pipeline, DE-CH, 0] - 1.0 flow_transport_loss[natural_gas_pipeline, CH-DE, 0] - 1.0 flow_transport[natural_gas_pipeline, CH-DE, 0] - 1.0 flow_storage_charge[natural_gas_storage, CH, 0] + 1.0 flow_storage_discharge[natural_gas_storage, CH, 0] + 1.0 flow_import[natural_gas, CH, 0] - 1.0 flow_export[natural_gas, CH, 0] + 1.0 shed_demand[natural_gas, CH, 0] = 0
        [natural_gas, DE, 0]:	-1.0 flow_conversion_input[natural_gas_boiler, natural_gas, DE, 0] + 1.0 flow_transport[natural_gas_pipeline, CH-DE, 0] - 1.0 flow_transport_loss[natural_gas_pipeline, DE-CH, 0] - 1.0 flow_transport[natural_gas_pipeline, DE-CH, 0] - 1.0 flow_storage_charge[natural_gas_storage, DE, 0] + 1.0 flow_storage_discharge[natural_gas_storage, DE, 0] + 1.0 flow_import[natural_gas, DE, 0] - 1.0 flow_export[natural_gas, DE, 0] + 1.0 shed_demand[natural_gas, DE, 0] = 0

    constraint_carrier_conversion:
        [natural_gas_boiler, natural_gas, CH, 0]:	1.0 flow_conversion_input[natural_gas_boiler, natural_gas, CH, 0] - 1.1 flow_conversion_output[natural_gas_boiler, heat, CH, 0] = 0
        [natural_gas_boiler, natural_gas, DE, 0]:	1.0 flow_conversion_input[natural_gas_boiler, natural_gas, DE, 0] - 1.1 flow_conversion_output[natural_gas_boiler, heat, DE, 0] = 0

    constraint_couple_storage_level:
        [natural_gas_storage, CH, 0]:	1.0 storage_level[natural_gas_storage, CH, 0] - 1.0 storage_level[natural_gas_storage, CH, 0] - 0.9747 flow_storage_charge[natural_gas_storage, CH, 0] + 1.026 flow_storage_discharge[natural_gas_storage, CH, 0] = 0
        [natural_gas_storage, DE, 0]:	1.0 storage_level[natural_gas_storage, DE, 0] - 1.0 storage_level[natural_gas_storage, DE, 0] - 0.9747 flow_storage_charge[natural_gas_storage, DE, 0] + 1.026 flow_storage_discharge[natural_gas_storage, DE, 0] = 0

    constraint_transport_technology_losses_flow:
        [natural_gas_pipeline, CH-DE, 0]:	1.0 flow_transport_loss[natural_gas_pipeline, CH-DE, 0] - 0.0255 flow_transport[natural_gas_pipeline, CH-DE, 0] = 0
        [natural_gas_pipeline, DE-CH, 0]:	1.0 flow_transport_loss[natural_gas_pipeline, DE-CH, 0] - 0.0255 flow_transport[natural_gas_pipeline, DE-CH, 0] = 0

The IIS doesn't tell you which constraint is "wrong"; you have to figure that 
out yourself. If you were to relax any of the constraints, the problem would be 
feasible again. Intuitively, relaxing the ``constraint_nodal_energy_balance`` 
makes it feasible. Then, if you would relax the technology constraints 
``constraint_carrier_conversion``, ``constraint_couple_storage_level``, or 
``constraint_transport_technology_losses_flow``, you could produce heat without 
consuming any other carrier. All of these constraints behave as expected and 
desired. Now, if you consider ``constraint_availability_import``, you see that 
neither heat nor natural gas can be imported ``(flow_import <= 0)``, and so the 
problem becomes infeasible. This simple example can help you to understand the 
IIS and thereby find infeasibilities in your problem.



