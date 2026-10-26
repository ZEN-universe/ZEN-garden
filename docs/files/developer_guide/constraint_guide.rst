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
3. The class name is descriptive and ends in ``Constraint``. It declares the
   stable registry ``name`` and ``indices``, and implements exactly one
   ``get_constraint`` classmethod. ``GenericConstraint.build`` is final,
   validates the returned dimensions against ``indices``, and registers the
   relation. Through ``model_constructor`` the constraint reaches
   ``model_constructor.optimization_model.sets``, ``.parameters``,
   ``.variables``, and ``.expressions``, as well as
   ``model_constructor.element_registry``, ``.network_topology``, and
   ``.time_steps`` for anything else it needs.
4. We write the mathematical formulation of the constraint in the docstring of
   ``get_constraint``. This is important for documentation purposes and helps us
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
6. Reused or independently meaningful algebra belongs in a named
   ``GenericExpression`` class. Expressions declare dependencies by name and
   are constructed in global topological order before constraints. Keep a
   comparison local when it is short and specific to one constraint.
7. A module contains one concrete constraint class and that class returns one
   relation. Related equations use separate modules and share expressions or
   ordinary utility functions instead of sharing a constraint builder.

.. code-block:: python

    class ExampleConstraint(GenericConstraint):
        name = "constraint_example_upper"
        indices = ("set_assets", "set_years")
        masks = (AvailableAssetMask,)

        @classmethod
        def should_construct(cls, model_constructor):
            return bool(model_constructor.optimization_model.sets["set_assets"])

        @classmethod
        def get_constraint(cls, model_constructor):
            """Summary:
            This is an example constraint.

            Formulation:

            .. math::
                x_{a,b} + y_{a,b} <= p_{a,b}

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
            return lhs <= rhs


Constraints with same dimensionality
------------------------------------

The easiest constraints to write are those that have the same dimensionality on
both sides of the equation. For example, let's say we want to limit the
``shed_demand`` by the ``demand``. Let's look at the dimensionality:

.. code-block::

    variables["shed_demand"]:

    Variable (set_carriers: 2, set_nodes: 2, set_years: 1,
              set_representative_hours: 2)
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

    <xarray.DataArray (set_carriers: 2, set_nodes: 2, set_years: 1,
                      set_representative_hours: 2)> Size: 64B
    array([[[ 10.,  10.],
            [100., 100.]],
           [[  0.,   0.],
            [  0.,   0.]]])
    Coordinates:
      * set_carriers              (set_carriers) <U11 88B 'heat' 'natural_gas'
      * set_nodes                 (set_nodes) <U2 16B 'CH' 'DE'
      * set_years                 (set_years) int64 8B 0
      * set_representative_hours  (set_representative_hours) int64 16B 0 1

We can see that the ``shed_demand`` and the ``demand`` have the same
dimensionality, as they are defined for 2 carriers ('heat' and 'natural_gas')
in 2 nodes ('CH' and 'DE'), one year, and 2 representative hours (0 and 1).

We can now write the constraint as follows:

.. code-block:: python

    lhs = model_constructor.optimization_model.variables["shed_demand"]
    rhs = model_constructor.optimization_model.parameters.demand
    return lhs <= rhs

Masking
-------
If we look at the actual constraint that limits ``shed_demand`` in
:py:meth:`LimitShedDemandConstraint.get_constraint <zen_garden.elements.carrier.constraints.limit_shed_demand_constraint.LimitShedDemandConstraint.get_constraint>`,
we see that we actually want to add another condition, which is that
``shed_demand = 0`` if the ``price_shed_demand`` is ``np.inf``. We can achieve
this by masking the ``demand`` parameter for the corresponding carriers:

.. code-block:: python

    mask = model_constructor.optimization_model.parameters.price_shed_demand != np.inf
    lhs = model_constructor.optimization_model.variables["shed_demand"]
    rhs = model_constructor.optimization_model.parameters.demand.where(mask, 0.0)
    return lhs <= rhs

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
    return lhs <= rhs

Masks are boolean arrays and therefore substitute ``if-statements``.


Broadcasting
------------

An important concept in ``xarray`` is broadcasting.
Briefly said, broadcasting is the process of making arrays with different shapes
compatible for arithmetic operations. Much broadcasting is done implicitly: in
the example above, the ``price_shed_demand`` has only one dimension,
``set_carriers``, while the ``shed_demand`` has four dimensions:

.. code-block::

    parameters.price_shed_demand:

    <xarray.DataArray (set_carriers: 2)> Size: 16B
    array([inf, inf])
    Coordinates:
      * set_carriers  (set_carriers) <U11 88B 'heat' 'natural_gas'

Therefore, the ``mask`` also has only one dimension.
When we use the ``where`` method, the ``mask`` is automatically broadcasted to
the shape of the ``shed_demand``.

The explicit year and representative-hour dimensions make annual constraints
direct. Let's take the example from
:py:meth:`CostCarrierTotalConstraint.build <zen_garden.elements.carrier.constraints.cost_carrier_total_constraint.CostCarrierTotalConstraint.build>`:
We multiply hourly carrier cost by the year-specific representative-hour
weight and sum over all representative hours of the year. ``xarray`` and
Linopy align the shared ``set_years`` and ``set_representative_hours``
dimensions automatically.

.. code-block:: python

    weight = (
        model_constructor.optimization_model.parameters
        .weight_representative_hours
    )
    weighted_cost = (
        model_constructor.optimization_model.variables["cost_carrier"]
        * weight
    )

.. code-block::

    variables["cost_carrier"]:

    Variable (set_carriers: 2, set_nodes: 2, set_years: 1,
              set_representative_hours: 2)
    ---------------------------------------------------------------------
    [heat, CH, 0]: cost_carrier[heat, CH, 0] ∈ [-inf, inf]
    [heat, CH, 1]: cost_carrier[heat, CH, 1] ∈ [-inf, inf]
    [heat, DE, 0]: cost_carrier[heat, DE, 0] ∈ [-inf, inf]
    [heat, DE, 1]: cost_carrier[heat, DE, 1] ∈ [-inf, inf]
    [natural_gas, CH, 0]: cost_carrier[natural_gas, CH, 0] ∈ [-inf, inf]
    [natural_gas, CH, 1]: cost_carrier[natural_gas, CH, 1] ∈ [-inf, inf]
    [natural_gas, DE, 0]: cost_carrier[natural_gas, DE, 0] ∈ [-inf, inf]
    [natural_gas, DE, 1]: cost_carrier[natural_gas, DE, 1] ∈ [-inf, inf]

    weight:

    <xarray.DataArray (set_years: 1, set_representative_hours: 2)> Size: 16B
    array([[1., 1.]])
    Coordinates:
      * set_years     (set_years) int64 8B 0
      * set_representative_hours  (set_representative_hours) int32 8B 0 1

The resulting ``weighted_cost`` has the dimensions ``set_carriers``,
``set_nodes``, ``set_years``, and ``set_representative_hours``.


Summing over dimensions
-----------------------

We can now sum over the representative hours, nodes, and carriers while
retaining the explicit year dimension:

.. code-block:: python

    term_summed_cost_carrier = weighted_cost.sum(
        ["set_carriers", "set_nodes", "set_representative_hours"]
    )

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
ZEN-garden has an implemented function that selects and replaces a time
dimension in one operation. ``map_and_expand`` lives in
``zen_garden.model.linear_expression``. Storage-state expressions use it to select
the operational hour corresponding to each year-specific storage hour:

.. code-block:: python

    mapping = model_constructor.time_steps.storage_to_operation
    years = list(model_constructor.optimization_model.sets["set_years"])
    times_power2energy = mapping.loc[
        mapping.index.get_level_values("set_years").isin(years)
    ]
    storage_flow = map_and_expand(
        operational_storage_flow, times_power2energy
    )

The mapper is a ``pd.Series`` with a ``(set_years,
set_representative_hours_storage)`` MultiIndex. Its values are members of
``set_representative_hours``. The year dimension is therefore preserved while
only the within-year time dimension is mapped.


Align and mask
--------------
Sometimes, the dimensions of a mask are not the same as the dimensions of the
variable or expression that we want to mask. In other cases, the order of the
indices of dimensions are not aligned. ``align_and_mask`` in
``zen_garden.model.linear_expression`` aligns the dimensions of the mask and
variable/expression, and sets the mask.

One such application, from
:py:meth:`LinearCapexConstraint.build <zen_garden.elements.conversion_technology.constraints.linear_capex_constraint.LinearCapexConstraint.build>`,
looks like this:

.. code-block:: python

    lhs = align_and_mask(lhs, mask)

We use ``align_and_mask`` analogously to ``.where(mask)`` but for more complex
(= many dimensions) expressions, where ``.where(mask)`` does not work. You will
get a feeling for when to use ``.where(mask)`` and where you have to use
``align_and_mask``.


Merging expressions
-------------------

When we want to merge two expressions in one, we can use the ``lp.merge``
function (here illustrated by the ``nodal_net_supply`` expression):

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
