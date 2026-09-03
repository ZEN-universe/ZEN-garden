.. _linopy.linopy:

#############################
Writing constraints in Linopy
#############################

ZEN-garden uses the `Linopy <https://linopy.readthedocs.io/en/latest/>`_ package
to build the optimization problem. In comparison to other packages, such as
`Pyomo <https://www.pyomo.org/>`_, Linopy reduces model construction time by
using vectorized constraint formulations. While this significantly speeds up the
model construction, the vectorization of complex constraints is not always
intuitive and straightforward.

.. note::

    In Linopy, vectorization is fast and individual data lookups (``for-loops``,
    ``.loc``, etc.) are incredibly slow. This should be avoided at all cost.

This guide is intended to help you write constraints in Linopy. Of course, it
cannot cover everything, but we want to provide you with a shopping cart of tips
and tricks in Linopy. Writing complex constraints in Linopy is an art in itself,
and we are happy to help you with it. Linopy uses the framework `xarray
<https://docs.xarray.dev/en/stable/>`_, so many of the presented concepts
involve the manipulation of ``xarray DataSets``.


General structure of constraints
--------------------------------

Our constraints are always set up in the same way:

1. Each constraint is its own class, e.g. ``NodalEnergyBalanceConstraint`` in
   ``zen_garden/elements/carrier/constraints/nodal_energy_balance_constraint.py``.
   The class inherits from
   :py:class:`GenericConstraint <zen_garden.model.component_types.constraint.GenericConstraint>`
2. The class is added to the element's constraint list, e.g. ``CARRIER_CONSTRAINTS`` in
   ``zen_garden/elements/carrier/constraints/__init__.py``, which the
   ``Carrier`` element class then exposes through its ``constraints`` attribute
   (see :ref:`adding_elements.adding_constraints`).
3. The class name is descriptive and ends in ``Constraint``. Its single entry
   point is a ``build`` classmethod that takes ``model_constructor`` as its
   argument. Through ``model_constructor`` the constraint reaches
   ``model_constructor.optimization_model.sets``, ``.parameters``,
   ``.variables``, and ``.add_constraint()``, as well as
   ``model_constructor.element_registry``, ``.network_topology``, and
   ``.time_steps`` for anything else it needs.
4. We write the mathematical formulation of the constraint in the docstring of
   ``build``. This is important for documentation purposes and helps us
   understand the constraint when we come back to it later. The docstring is
   split into a ``Summary:``, a ``Formulation:`` (the mathematical formulation
   in LaTeX notation), and a ``Notation:`` section, and is inserted into the
   mathematical formulation pages by the docstring extension (see
   :ref:`documentation.docstrings`).
5. We define the left-hand side ``lhs`` and the right-hand side ``rhs`` of the
   constraint, then compare them directly with a Python operator (``<=``,
   ``>=``, or ``==``) to build the constraint, e.g. ``constraints = lhs <=
   rhs``. The ``lhs`` only contains variables or products of variables and
   parameters. The ``rhs`` can only contain parameters.
6. For the sake of readability, we call intermediate constraint expressions,
   i.e., multiple variable terms or products of variables and parameters,
   ``term_<descriptive_name_for_term>``.
7. You can add multiple constraints in the same ``build`` method. This is
   useful if you have multiple constraints that are similar or related to each
   other, for example the cost and the limit of shedding demand in
   ``CostLimitShedDemandConstraint``. We recommend to only add multiple
   constraints in one method if the constraints are very similar so that we
   can save redundant code.

.. code-block:: python

    class ExampleConstraint(GenericConstraint):
        @classmethod
        def build(cls, model_constructor):
            """Summary:
            This is an example constraint.

            Formulation:

            .. math::
                x_{a,b} + y_{a,b} <= p_{a,b}
                x_{a,b} + y_{a,b} >= q_{a,b}

            Notation:

            :math:`x_{a,b}`: Indexed variable for :math:`a` and :math:`b`.
            :math:`y_{a,b}`: Indexed variable for :math:`a` and :math:`b`.
            :math:`p_{a,b}`: Indexed parameter for :math:`a` and :math:`b`.
            :math:`q_{a,b}`: Indexed parameter for :math:`a` and :math:`b`.
            """
            term_product_xp = (
                model_constructor.optimization_model.variables["x"]
                * model_constructor.optimization_model.parameters.p
            )
            lhs = term_product_xp + model_constructor.optimization_model.variables["y"]
            rhs = model_constructor.optimization_model.parameters.p
            constraints = lhs <= rhs
            model_constructor.optimization_model.add_constraint(
                "constraint_example_constraint_upper", constraints
            )
            rhs = model_constructor.optimization_model.parameters.q
            constraints = lhs >= rhs
            model_constructor.optimization_model.add_constraint(
                "constraint_example_constraint_lower", constraints
            )


Constraints with same dimensionality
------------------------------------

The easiest constraints to write are those that have the same dimensionality on
both sides of the equation. For example, let's say we want to limit the
``shed_demand`` by the ``demand``. Let's look at the dimensionality:

.. code-block::

    variables["shed_demand"]:

    Variable (set_carriers: 2, set_nodes: 2, set_time_steps_operation: 2)
    ---------------------------------------------------------------------
    [heat, CH, 0]: shed_demand[heat, CH, 0] ∈ [0, inf]
    [heat, CH, 1]: shed_demand[heat, CH, 1] ∈ [0, inf]
    [heat, DE, 0]: shed_demand[heat, DE, 0] ∈ [0, inf]
    [heat, DE, 1]: shed_demand[heat, DE, 1] ∈ [0, inf]
    [natural_gas, CH, 0]: shed_demand[natural_gas, CH, 0] ∈ [0, inf]
    [natural_gas, CH, 1]: shed_demand[natural_gas, CH, 1] ∈ [0, inf]
    [natural_gas, DE, 0]: shed_demand[natural_gas, DE, 0] ∈ [0, inf]
    [natural_gas, DE, 1]: shed_demand[natural_gas, DE, 1] ∈ [0, inf]

    parameters.demand:

    <xarray.DataArray (set_carriers: 2, set_nodes: 2, set_time_steps_operation: 2)> Size: 64B
    array([[[ 10.,  10.],
            [100., 100.]],
           [[  0.,   0.],
            [  0.,   0.]]])
    Coordinates:
      * set_carriers              (set_carriers) <U11 88B 'heat' 'natural_gas'
      * set_nodes                 (set_nodes) <U2 16B 'CH' 'DE'
      * set_time_steps_operation  (set_time_steps_operation) int64 16B 0 1

We can see that the ``shed_demand`` and the ``demand`` have the same
dimensionality, as they are defined for 2 carriers ('heat' and 'natural_gas')
in 2 nodes ('CH' and 'DE') for 2 time steps (0 and 1).

We can now write the constraint as follows:

.. code-block:: python

    lhs = model_constructor.optimization_model.variables["shed_demand"]
    rhs = model_constructor.optimization_model.parameters.demand
    constraints = lhs <= rhs
    model_constructor.optimization_model.add_constraint(
        "constraint_shed_demand", constraints
    )

Masking
-------
If we look at the actual constraint that limits ``shed_demand`` in
:py:meth:`CostLimitShedDemandConstraint.build <zen_garden.elements.carrier.constraints.cost_limit_shed_demand_constraint.CostLimitShedDemandConstraint.build>`,
we see that we actually want to add another condition, which is that
``shed_demand = 0`` if the ``price_shed_demand`` is ``np.inf``. We can achieve
this by masking the ``demand`` parameter for the corresponding carriers:

.. code-block:: python

    mask = model_constructor.optimization_model.parameters.price_shed_demand != np.inf
    lhs = model_constructor.optimization_model.variables["shed_demand"]
    rhs = model_constructor.optimization_model.parameters.demand.where(mask, 0.0)
    constraints = lhs <= rhs
    model_constructor.optimization_model.add_constraint(
        "constraint_limit_shed_demand", constraints
    )

This will overwrite the rhs and thereby set the ``shed_demand`` to 0 if the
``price_shed_demand`` is ``np.inf``.

Analogously, we can mask variables or entire expressions. Let's say
(hypothetically - this is not a real constraint) that we only want to formulate
the constraint for those ``carriers`` that have a ``price_shed_demand != 0``:

.. code-block:: python

    mask_0 = model_constructor.optimization_model.parameters.price_shed_demand != 0
    mask_inf = model_constructor.optimization_model.parameters.price_shed_demand != np.inf
    lhs = model_constructor.optimization_model.variables["shed_demand"].where(mask_0)
    rhs = model_constructor.optimization_model.parameters.demand.where(mask_inf, 0.0)
    constraints = lhs <= rhs
    model_constructor.optimization_model.add_constraint(
        "constraint_shed_demand", constraints
    )

Masks are boolean arrays and therefore substitute ``if-statements``.


Broadcasting
------------

An important concept in ``xarray`` is broadcasting.
Briefly said, broadcasting is the process of making arrays with different shapes
compatible for arithmetic operations. Much broadcasting is done implicitly: in
the example above, the ``price_shed_demand`` has only one dimension,
``set_carriers``, while the ``shed_demand`` has three dimensions:

.. code-block::

    parameters.price_shed_demand:

    <xarray.DataArray (set_carriers: 2)> Size: 16B
    array([inf, inf])
    Coordinates:
      * set_carriers  (set_carriers) <U11 88B 'heat' 'natural_gas'

Therefore, the ``mask`` also has only one dimension.
When we use the ``where`` method, the ``mask`` is automatically broadcasted to
the shape of the ``shed_demand``.

Sometimes, we have to help linopy a bit to broadcast. This is especially the
case when none of the dimensions overlap. We can expand the dimensionality of,
e.g., a variable by the dimensions of a parameter by using ``broadcast_like``.
Let's take the example from
:py:meth:`CostCarrierTotalConstraint.build <zen_garden.elements.carrier.constraints.cost_carrier_total_constraint.CostCarrierTotalConstraint.build>`:
We want to multiply the hourly carrier cost with the duration of the time step
and sum over all time steps of the year. ``get_year_time_step_duration_array``
is one of the helper methods that
:py:class:`GenericConstraint <zen_garden.model.component_types.constraint.GenericConstraint>`
provides to every constraint class; call it as ``cls.<helper_name>(...)`` from
within ``build``.

.. code-block:: python

    times = cls.get_year_time_step_duration_array(model_constructor)
    term_expanded_cost_carrier = model_constructor.optimization_model.variables[
        "cost_carrier"
    ].broadcast_like(times)

.. code-block::

    variables["cost_carrier"]:

    Variable (set_carriers: 2, set_nodes: 2, set_time_steps_operation: 2)
    ---------------------------------------------------------------------
    [heat, CH, 0]: cost_carrier[heat, CH, 0] ∈ [-inf, inf]
    [heat, CH, 1]: cost_carrier[heat, CH, 1] ∈ [-inf, inf]
    [heat, DE, 0]: cost_carrier[heat, DE, 0] ∈ [-inf, inf]
    [heat, DE, 1]: cost_carrier[heat, DE, 1] ∈ [-inf, inf]
    [natural_gas, CH, 0]: cost_carrier[natural_gas, CH, 0] ∈ [-inf, inf]
    [natural_gas, CH, 1]: cost_carrier[natural_gas, CH, 1] ∈ [-inf, inf]
    [natural_gas, DE, 0]: cost_carrier[natural_gas, DE, 0] ∈ [-inf, inf]
    [natural_gas, DE, 1]: cost_carrier[natural_gas, DE, 1] ∈ [-inf, inf]

    times:

    <xarray.DataArray (set_years: 1, set_time_steps_operation: 2)> Size: 16B
    array([[1., 1.]])
    Coordinates:
      * set_years     (set_years) int64 8B 0
      * set_time_steps_operation  (set_time_steps_operation) int32 8B 0 1

The resulting ``term_expanded_cost_carrier`` has the shape
``(set_years: 1, set_time_steps_operation: 2, set_carriers: 2, set_nodes: 2)``.


Summing over dimensions
-----------------------

With the expanded variable, we can now sum over the time steps of the year, the
nodes, and the carriers:

.. code-block:: python

    term_summed_cost_carrier = term_expanded_cost_carrier.sum(["set_carriers", "set_nodes", "set_time_steps_operation"])

The resulting ``term_summed_cost_carrier`` has the shape
``(set_years: 1)``. Broadcasting and summing over dimensions is a
powerful tool to manipulate the dimensionality of variables and parameters. In
many situations it can substitute ``for-loops``.


Renaming dimensions
-------------------

Broadcasting works on the names of the dimensions. If the names of the
dimensions are not the same, we can rename them using ``rename``.

For example, the variable ``capacity`` is defined for all ``set_technologies``
and ``set_location``, whereas ``storage_level`` is only defined for
``set_storage_technologies`` and ``set_nodes``. To build a constraint with the
two variables, we must rename the dimensions of ``capacity`` (see
:py:meth:`StorageLevelMaxConstraint.build <zen_garden.elements.storage_technology.constraints.storage_level_max_constraint.StorageLevelMaxConstraint.build>`):

.. code-block:: python

    capacity = model_constructor.optimization_model.variables["capacity"].rename(
        {"set_technologies": "set_storage_technologies", "set_location": "set_nodes"}
    )
    capacity = capacity.sel(
        {
            "set_nodes": model_constructor.optimization_model.sets["set_nodes"],
            "set_storage_technologies": model_constructor.optimization_model.sets[
                "set_storage_technologies"
            ],
        }
    )

The ``.sel({<dimension>: <values>})`` is a fast method to select the values of a
dimension.


Map and expand
--------------
ZEN-garden has an implemented function that broadcasts, selects, and renames
variables and parameters in one go. It is called ``map_and_expand``, and, like
``get_year_time_step_duration_array``, it is one of the helper methods on
:py:class:`GenericConstraint <zen_garden.model.component_types.constraint.GenericConstraint>`,
called as ``cls.map_and_expand(...)``. For example, in the previous example, we
want to restructure ``capacity`` to fit the ``set_time_steps_storage`` so that
we can use it as a constraint for the ``storage_level``:

.. code-block:: python

    times = cls.get_storage2year_time_step_array(model_constructor)
    capacity = cls.map_and_expand(
        model_constructor.optimization_model.variables["capacity"], times
    )

.. code-block::

    variables["capacity"]

    Variable (set_time_steps_storage: 2,
        set_technologies: 3,
        set_capacity_types: 2,
        set_location: 4,
        set_years: 1) - 32 masked entries

    times

    set_time_steps_storage
    0    0
    1    0
    Name: set_years, dtype: int64

    Output

    Variable (set_technologies: 3,
        set_capacity_types: 2,
        set_location: 4,
        set_time_steps_storage: 2) - 32 masked entries

Note that the ``mapper`` (in this case ``times``) must be a ``pd.Series``, whose
name is the dimension we want to replace and whose index name is the target
dimension.


Align and mask
--------------
Sometimes, the dimensions of a mask are not the same as the dimensions of the
variable or expression that we want to mask. In other cases, the order of the
indices of dimensions are not aligned. Therefore, ``GenericConstraint`` also
provides ``align_and_mask``, which aligns the dimensions of the mask and
variable/expression, and sets the mask.

One such application, from
:py:meth:`LinearCapexConstraint.build <zen_garden.elements.conversion_technology.constraints.linear_capex_constraint.LinearCapexConstraint.build>`,
looks like this:

.. code-block:: python

    lhs = cls.align_and_mask(lhs, mask)

We use ``align_and_mask`` analogously to ``.where(mask)`` but for more complex
(= many dimensions) expressions, where ``.where(mask)`` does not work. You will
get a feeling for when to use ``.where(mask)`` and where you have to use
``align_and_mask``.


Merging expressions
-------------------

When we want to merge two expressions in one, we can use the ``lp.merge``
function (here from
:py:meth:`NodalEnergyBalanceConstraint.build <zen_garden.elements.carrier.constraints.nodal_energy_balance_constraint.NodalEnergyBalanceConstraint.build>`):

.. code-block:: python

    lhs = lp.merge(
        [
            term_carrier_conversion_out,
            -term_carrier_conversion_in,
            term_flow_transport_in,
            -term_flow_transport_out,
            -term_flow_storage_charge,
            term_flow_storage_discharge,
            term_carrier_import,
            -term_carrier_export,
            term_carrier_shed_demand,
        ],
        compat="broadcast_equals",
        join="outer",
        cls=LinearExpression,
    )

This function merges the expressions and makes sure that the dimensions are
compatible. This is often faster and more reliable than using ``+`` or ``-``.
Passing ``cls=LinearExpression`` (imported from ``linopy.expressions``) keeps
the result a linear expression rather than a plain ``xarray`` object.
