.. _plugins.available_events:

Available events
====================

.. _plugins.available_events.after_model_construction:

Event ``after_model_construction``
------------------------------------

The event ``after_model_construction`` is triggered after model construction, but before scaling and solving the model:
The ``optimization_setup`` object is passed to the event handler, so you are able to modify the optimization problem
before scaling and solving. The trigger is placed in ``runner.py``:

.. code-block:: python

    # create optimization problem
    optimization_setup.construct_optimization_problem()
    EventPublisher.trigger(
        Event.after_model_construction, optimization_setup=optimization_setup
    )

    if optimization_setup.solver.use_scaling:
        optimization_setup.scaling.run_scaling()
    elif (
        optimization_setup.solver.analyze_numerics
        or optimization_setup.solver.run_diagnostics
    ):
        optimization_setup.scaling.analyze_numerics()
    # SOLVE THE OPTIMIZATION PROBLEM
    optimization_setup.solve()

Exemplary use cases for this event include:

- Defining a new objective function. You can first delete the already defined objective function
  with ``optimization_setup.model.remove_objective()`` and then add a new one. For instance, you can
  think of adding a bias term to the objective function which co-optimizes cost and another metric.
- Define new variables to be used in new constraints.
- Define new constraints. E.g. policy targets for certain capacity expansions, generation constraints of
  technologies,...

