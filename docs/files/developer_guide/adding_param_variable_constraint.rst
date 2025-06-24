.. _adding_elements.structure:

###################################################
Adding Sets, Parameters, Variables, and Constraints
###################################################

An optimization problem is defined by a set of elements, specifically sets, parameters, variables, and constraints.
Variables are the values that are optimized, i.e., decided by the optimizer.
Parameters are the fixed input values that are used in the optimization process, such as demands or specific costs.
Constraints are the rules that the solution must adhere to, such as energy balances, capacity limits, or resource availability.
Sets are the indices that define the scope of the problem, such as locations or time periods.
Most variables, parameters, and constraints are indexed by sets, meaning they are defined for each index in the set.

This section provides a guide on how to add these elements to ZEN-garden.
Generally, the elements will be added to the class where they logically belong.
For example, if you wanted to add a constraint to set a minimum import flow of a carrier, you would add it to the `Carrier` class.

.. note::

    This guide assumes you have a good understanding of Python and the ZEN-garden framework.
    ZEN-garden already has plenty of functionalities, so check out the :ref:`math_formulation.math_formulation` and
    :ref:`notation.notation` for more information on how to use the existing functionalities.

.. _adding_elements.adding_sets:


Adding Sets
-----------

Sets can be added in the `construct_sets` method of the element classes,
i.e., `EnergySystem`, `Carrier`, or `Technology` (including all subtechnology classes, such as `ConversionTechnology`).
The new set is added to the `optimization_setup.sets` through the method `optimization_setup.sets.add_set`.

The `add_set` method takes the following parameters:
- `name`: The name of the set, which should be unique.
- `data`: The data for the set, which can be a list or a dictionary.
- `doc`: The documentation for the set, which should be a string describing the set.
- `index_set`: The set that is used as the index for the new set, if applicable.

Two examples for adding a set is shown below (from the `Technology` class):

.. code-block:: python

    optimization_setup.sets.add_set(
        name="set_conversion_technologies",
        data=energy_system.set_conversion_technologies,
        doc="Set of conversion technologies")

    optimization_setup.sets.add_set(
        name="set_reference_carriers",
        data=optimization_setup.get_attribute_of_all_elements(cls, "reference_carrier"),
        doc="set of all reference carriers correspondent to a technology. Indexed by set_technologies",
        index_set="set_technologies")

The first example is not indexed by any set, while the second example is indexed by the `set_technologies` set.
That means that each technology from the `set_technologies` set will have a corresponding entry in the `set_reference_carriers` set.

Adding Parameters
-----------------

Parameters can be added in the `construct_params` method of the element classes.
But first, the data has to be imported in the `store_input_data` method of the element classes.