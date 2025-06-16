######################
Troubleshooting
######################

Frequently made mistakes
========================

We try to make ZEN-garden's error messages as helpful as possible, but sometimes 
it can be hard to understand what went wrong, especially when the errors occur 
in other packages.

Here are some common mistakes that can lead to errors:

Comma at the end of a list in json
==================================

 ``"list": [1, 2, 3,]`` is wrong, it should be ``"list": [1, 2, 3]``. This is a 
 common mistake because Python allows it, but JSON does not. The cryptic error 
 message is ``json.decoder.JSONDecodeError: Expecting value: [...]``. Fix: check 
 ``system.json``, ``config.json``, and ``attributes.json`` for commas at the end 
 of lists. When you scroll up in the error message, you can guess what file 
 caused the error.


Building smaller test models
============================

If you have a large model and you are struggling with infeasibilities or unclear 
problems, it can be helpful to build a smaller test model. This way, you can 
quickly identify the source of the infeasibility or problem and fix it. Once you 
have a working small model, you can gradually add more complexity until you have 
the full model again.

The easiest way to build a smaller model is to reduce the number of time steps, 
years, regions, or technologies. If you are using time series aggregation 
(see :ref:`tsa.tsa`), reduce the number of 
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
3. improve your numerics by scaling your model (see :ref:`input_handling.scaling`)
4. improve your numerics by selecting other solver options (if you are using 
   Gurobi see `Guidelines for Numerical Issues 
   <https://www.gurobi.com/documentation/current/refman/guidelines_for_numerical_i.html>`_)


