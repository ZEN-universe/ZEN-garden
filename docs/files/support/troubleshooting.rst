.. _troubleshooting.troubleshooting:

###############
Troubleshooting
###############

This page collects the mistakes users hit most often, and what to do about
them. If your model does not solve at all, start with the tutorial on
:ref:`infeasibilities <t_infeasibilities.t_infeasibilities>`.


.. _troubleshooting.frequent_mistakes:

Frequently made mistakes
========================

ZEN-garden tries to make its error messages as helpful as possible, but errors
that occur inside other packages can still be hard to read.


Indentation errors in YAML
---------------------------

YAML uses indentation, not braces, to define structure. Mixing tabs and
spaces, or shifting the indentation level of a line so it no longer lines up
with its siblings, produces a parsing error. The resulting error message
points at the offending line and column, for example::

    yaml.scanner.ScannerError: while scanning for the next token
    found character '\t' that cannot start any token

Check ``system.yaml``, ``config.yaml``, ``scenarios.yaml`` and the
``attributes.yaml`` files for inconsistent indentation. Scrolling up in the
error message usually reveals which file caused it.


Unit consistency errors
-----------------------

The dataset example ``14_unit_consistency_expected_error`` intentionally
contains inconsistent units. Run it following the instructions for
:ref:`using dataset examples <building.examples>` and use the error message to
locate the inconsistent ``unit`` entries in the ``attributes.yaml`` files.
See :ref:`t_units.t_units` and :ref:`input_structure.attribute_files` for the
unit conventions.

For this example, the affected units include:

1. ``CHP_plant/conversion_factor[natural_gas]``: ``kilotons/GWh``
2. ``natural_gas_pipeline/capacity_investment_existing``: ``kilotons/hour``
3. The energy-related units of ``natural_gas_storage``: use the kiloton basis
   instead of the GWh/MWh basis.

After correcting the units, run the dataset again. ZEN-garden should complete
without a unit-consistency error.


Special characters and long paths
---------------------------------

Special characters in file or folder names lead to errors. On Windows, total
path lengths above 260 characters may also be rejected; keep carrier and
technology names short, or enable long paths as described in
:ref:`building.building`.


.. _troubleshooting.smaller_models:

Building smaller test models
============================

If you have a large model and you are struggling with infeasibilities or
unclear problems, it helps to build a smaller test model. You can then quickly
identify the source of the problem and fix it. Once the small model works, add
complexity back gradually until you have the full model again.

The easiest way to build a smaller model is to reduce the number of time steps,
years, nodes, or technologies. If you are using time series aggregation, reduce
``aggregated_time_steps_per_year`` (see :ref:`t_tsa.t_tsa`). Refer to
:ref:`configuration.system` for the relevant settings.


.. _troubleshooting.solution_times:

Improving solution times
========================

If you are struggling with long solution times:

1. Build a smaller model (see above).
2. Remove constraints that make the problem harder to solve. Binary variables
   are the usual culprit: ``min_load``, ``capacity_addition_min``,
   ``double_capex_transport`` and ``storage_charge_discharge_binary`` each turn
   the problem into a mixed-integer program. Technology expansion constraints
   (:ref:`t_expansion.t_expansion`) couple years and nodes and are also
   expensive.
3. Improve the numerics by scaling the model. See :ref:`t_scaling.t_scaling`
   for the algorithms and :ref:`t_scaling.scaling_recommendations` for which
   configuration to start from.
4. Check the numerical range with ``analyze_numerics: true`` in the ``solver``
   section of ``config.yaml``, and
   if you are using Gurobi, consult the `guidelines for numerical issues
   <https://www.gurobi.com/documentation/current/refman/guidelines_for_numerical_i.html>`_.
5. Reduce what is written to disk; large output files can dominate the runtime
   of small models. See :ref:`t_output.t_output`.
