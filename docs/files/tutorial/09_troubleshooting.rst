.. _t_troubleshooting._t_troubleshooting:


############################
Tutorial 9: Troubleshooting
############################

Frequently made mistakes
========================

We try to make ZEN-garden's error messages as helpful as possible, but sometimes 
it can be hard to understand what went wrong, especially when the errors occur 
in other packages.

Here are some common mistakes that can lead to errors:

Invalid YAML indentation
------------------------

YAML uses indentation to define nested mappings and lists. Tabs or incorrectly
aligned keys can cause a parser error. Check ``system.yaml``, ``config.yaml``,
and ``attributes.yaml`` near the line and column reported in the error message.


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


Building smaller test models
============================

If you have a large model and you are struggling with infeasibilities or unclear 
problems, it can be helpful to build a smaller test model. This way, you can 
quickly identify the source of the infeasibility or problem and fix it. Once you 
have a working small model, you can gradually add more complexity until you have 
the full model again.

The easiest way to build a smaller model is to reduce the number of time steps, 
years, regions, or technologies. If you are using time series aggregation 
(see :ref:`t_tsa.t_tsa`), reduce the number of 
``aggregated_time_steps_per_year``. Refer to :ref:`configuration.system` for the 
relevant settings.

Improving solution times
========================

If you are struggling with long solution times, there are several ways to 
improve them:

1. build a smaller model
2. remove constraints that make the problem harder to solve through parameter 
   selection, such as technology expansion constraints, binary constraints, or 
   storage constraints
3. improve your numerics by scaling your model (see :ref:`t_scaling.t_scaling`)
4. improve your numerics by selecting other solver options (if you are using 
   Gurobi see `Guidelines for Numerical Issues 
   <https://www.gurobi.com/documentation/current/refman/guidelines_for_numerical_i.html>`_)
