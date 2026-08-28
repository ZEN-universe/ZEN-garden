.. _results_api.results_api:

##################
Results API
##################

This page is the reference for working with a ``Results`` object. For a guided
introduction, see the tutorial on
:ref:`analyzing and comparing results <t_analyze.t_analyze>`.

.. code:: python

    from zen_garden import Results
    r = Results(path='<result_folder>')


.. _results_api.readers:

Reading components
==================

Four component types are available: ``'parameter'``, ``'variable'``, ``'dual'``
and ``'sets'``. List the names of one type with:

.. code:: python

    r.get_component_names('variable')

Descriptions of all components are given in :ref:`notation.notation`.

Once you know the name of a component, the following methods return its values:

.. list-table::
   :widths: 34 66
   :header-rows: 1

   * - Method
     - Returns
   * - ``r.get_full_ts(<name>)``
     - Full time series. For hourly-resolved data the series has a length of
       8760 times the number of years simulated.
   * - ``r.get_total(<name>)``
     - Annual totals. Values within the same year are summed.
   * - ``r.get_dual(<name>)``
     - Dual values of a constraint.
   * - ``r.get_unit(<name>)``
     - Unit of the component.
   * - ``r.get_doc(<name>)``
     - Documentation string of the component.

.. note::
    A ``Results`` object can only see components that were actually written to
    the result files. If a component you need is missing, add it to
    ``selected_saved_parameters`` or ``selected_saved_variables`` and re-run.
    See :ref:`t_output.t_output`.


.. _results_api.filtering:

Filtering arguments
===================

``get_full_ts``, ``get_total``, ``get_dual``, ``get_unit`` and ``get_doc``
accept optional arguments that narrow the result:

1. ``year``: a single optimization period (0, 1, 2, ...). Not available for
   ``r.get_unit()``.
2. ``scenario_name``: a single scenario name. Only relevant when the scenario
   tool is used; see :ref:`t_scenario.t_scenario`.
3. ``index``: a slicing index, i.e. the indices of the dataframe for which
   results should be returned.

There are four ways to pass an index:

1. **A single index**, e.g. ``r.get_total('capacity', index="heat_pump")``.
   Returns the capacity of the heat pump for all other indices (nodes, years).
   The index must correspond to the first index of the component.
2. **A list of indices**, e.g.
   ``r.get_total('capacity', index=["heat_pump", "photovoltaics"])``. Returns
   both technologies for all other indices. The index must correspond to the
   first index of the component.
3. **A tuple of indices**, e.g.
   ``r.get_total('capacity', index=("heat_pump", None, ["DE", "CH"]))``.
   Returns the heat pump capacity in the nodes DE and CH. The order of index
   levels matters. Each entry may be a single index, ``None``, or a list of
   indices; ``None`` returns all indices of that level.
4. **A dictionary**, e.g.
   ``r.get_total('capacity', index={"node": ["DE", "CH"], "technology": "heat_pump"})``.
   Returns the heat pump capacity in DE and CH. Because the keys are named, the
   order does not matter.


.. _results_api.units:

Units
=====

``r.get_unit()`` takes the additional argument ``convert_to_yearly_unit``
(default ``False``). If set to ``True``, the unit is converted to a yearly
unit, i.e. the unit string of components with an operational time step type is
multiplied by ``hour``.

``r.get_unit()`` can also read the unit of the objective function with
``r.get_unit('objective')``.


.. _t_analyze.compare:

Comparing results
=================

ZEN-garden provides functions to compare two ``Results`` objects. This helps to
understand why two results differ, and is a fast way to spot errors in a
dataset. The most useful application is comparing the configuration
(:ref:`configuration.configuration`) and the parameter values of two datasets.
Comparing variable values is often less informative, because results usually
differ in a large number of variables at once.

.. code:: python

    from zen_garden import Results, compare_model_values, compare_configs

    r1 = Results(path='<result_folder_1>')
    r2 = Results(path='<result_folder_2>')

    compare_parameters = compare_model_values([r1, r2], component_type='parameter')
    compare_variables  = compare_model_values([r1, r2], component_type='variable')
    compare_config     = compare_configs([r1, r2])

By default, ``compare_model_values`` compares the annual totals of components.
Pass ``compare_total=False`` to compare the full time series instead.
``compare_model_values`` also accepts ``component_type="dual"`` and
``component_type="sets"``. ``compare_configs`` compares the configurations of
the two datasets.
