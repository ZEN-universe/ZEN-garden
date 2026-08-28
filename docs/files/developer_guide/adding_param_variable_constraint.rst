.. _adding_elements.structure:

###################################################
Adding Sets, Parameters, Variables, and Constraints
###################################################

An optimization problem consists of sets, parameters, variables, and constraints.
Variables are the values that are optimized, i.e., decided by the optimizer.
Parameters are the fixed input values that are used in the optimization process, such as demands or specific costs.
Constraints are the rules that the solution must adhere to,
such as energy balances, capacity limits, or resource availability.
Sets are the indices that define the scope of the problem, such as locations or time periods.
Most variables, parameters, and constraints are indexed by sets, meaning they are defined for each index in the set.

This section provides a guide on how to add these elements to ZEN-garden.
Generally, the elements will be added to the class where they logically belong.
For example, if you wanted to add a constraint to set a minimum import flow of a carrier,
you would add it to the ``Carrier`` class.

.. tip::

    This guide assumes you have a good understanding of Python and the ZEN-garden framework.
    ZEN-garden already has plenty of functionalities, so check out the :ref:`mathematical_formulation.mathematical_formulation` and
    :ref:`notation.notation` for more information on how to use the existing functionalities.

.. _adding_elements.adding_sets:

Adding Sets
-----------

Sets can be added in the ``construct_sets`` method of the element consturctor class,
e.g., :py:class:`EnergySystemConstructor <zen_garden.elements.energy_system.energy_system_constructor.EnergySystemConstructor>`,
:py:class:`CarrierConstructor <zen_garden.elements.carrier.carrier_constructor.CarrierConstructor>`,
or :py:class:`TechnologyConstructor <zen_garden.elements.technology.technology_constructor.TechnologyConstructor>`.
The new set is added to ``self.zen_model.sets`` through the method ``self.zen_model.add_set()``.

The :py:meth:`add_set <zen_garden.model.components.set_registry.SetRegistry.add_set>` method takes the following parameters:

- ``name``: The name of the set, which should be unique.
- ``data``: The data for the set, which can be a list or a dictionary.
- ``doc``: The documentation for the set, which should be a string describing the set.
- ``index_set``: The set that is used as the index for the new set, if applicable.

Two examples for adding a set is shown below (from the ``Technology`` class):

.. code-block:: python

    self.zen_model.add_set(
        name="set_conversion_technologies",
        data=energy_system.set_conversion_technologies,
        doc="Set of conversion technologies")

    self.zen_model.add_set(
        name="set_reference_carriers",
        data=self.element_registry.get_attribute_of_all_elements(
            self.element_class, "reference_carrier"
        ),
        doc="set of all reference carriers correspondent to a technology. Indexed by set_technologies",
        index_set="set_technologies")

The first example is not indexed by any set, while the second example is indexed by the ``set_technologies`` set.
That means that each technology from the ``set_technologies`` set
will have a corresponding entry in the ``set_reference_carriers`` set.

.. _adding_elements.adding_parameters:

Adding Parameters
-----------------

Parameters can be added in the ``construct_params`` method of the element constructor class.
But first, the data has to be imported in the ``store_input_data`` method of the corresponding element classes
or the energy system class with the ``extract_input_data`` method of the
:py:class:``ElementDataLoader <zen_garden.preprocess.element_data_loader.ElementDataLoader>`` class.
For example, if you want to add a parameter for the yearly import availability of a carrier, the code looks like this:

.. code-block:: python

    self.availability_import_yearly = self.element_data_loader.extract_input_data(
        "availability_import_yearly",
        index_sets=["set_nodes", "set_years"],
        unit_category={"energy_quantity": 1}
    )

.. note::

    If the parameter is hourly resolved, it must be added to the ``self.raw_time_series`` attribute
    to be correctly handled in the time series aggregation:
    ``self.raw_time_series["demand"] = ...``

First, the name of the parameter is defined, in this case ``availability_import_yearly``.
Then, the ``index_sets``, i.e., the sets that the parameter is indexed by, are defined.
In this case, the parameter is indexed by ``set_nodes`` and ``set_years``
(which are the years of the optimization problem).

Finally, the ``unit_category`` parameter is set to the unit category of the parameter,
which is used for unit conversion and validation.
The ``unit_category`` is a dictionary with the categories of the unit and their power (+1 or -1).
For example ``{"energy_quantity": 1}`` means that the parameter is in energy quantity units (e.g., MWh, m^3, kg, etc.).
As discussed in :ref:`t_units.t_units`,
the concrete unit of the energy quantity is determined through the input data and is not predefined in the code.
What is predefined is how the unit dimensionalities build the parameter unit.
A parameter with emissions per distance and energy, for example,
would have a ``unit_category`` of ``{"emissions": 1, "distance": -1, "energy_quantity": -1}``.

.. note::

    Each element class has a corresponding ``<Classname>Constructor`` class that is responsible
    for constructing the sets, parameters, variables, and constraints of the element.
    The method `store_input_data` is part of the element class,
    while the method `construct_params` is in the constructor class.

After the input data is read, it can be added in the ``construct_params`` method
through the method ``optimization_setup.parameters.add_parameter``.
The ``add_parameter`` method is called in the following way:

.. code-block:: python

    self.add_parameter(
        name="availability_import_yearly",
        index_names=["set_carriers", "set_nodes", "set_years"],
        doc='Parameter which specifies the maximum energy that can be imported from outside the system boundaries for the entire year',
    )

The name must be the same as the name defined in the ``store_input_data`` method.
Note that the ``index_names`` now include ``set_carriers``, as the parameter is defined for all carriers.
Furthermore, the ``calling_class`` parameter is set to the class that is calling the method.

.. note::

    The parameters are available in the constraint rules through the ``self.parameters.<parameter_name>`` attribute.

Logging new and changed parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you add a new parameter or change the name of an existing one, please document that in
:py:data:`PARAMETER_CHANGE_LOG <zen_garden.preprocess.element_data_loader.PARAMETER_CHANGE_LOG>`.
The reason to add the name is that the new or changed parameters will be searched for in the input data,
but are not available in the datasets of others. To avoid breaking changes, the new or changed parameters
are documented in the log file and then equivalent parameters are found without breaking the code.

There are two possible options:

1. You change the name of an existing parameter, e.g.,
   from ``outdated_name`` to ``updated_name``.
   In this case, you add the old name to the log file and the new name as the current name.
   The code will then search for the old name in the input data and use the new name in the optimization.

.. code-block::

    log_dict = {
        "outdated_name": "updated_name",
        # other parameters...
    }

2. You add a new parameter that had not existed before, e.g., ``new_parameter``.
   In addition to the new name, you also provide the ``default_value`` (only `0`, `1`, or `inf` are allowed),
   and another parameter with the same unit category that is used to infer the unit of the new parameter.

.. code-block::

    log_dict = {
        "new_parameter": {
            "default_value": 0,
            "unit": "existing_parameter_name_with_same_unit"
        },
        # other parameters...
    }

In every major release, the log file is cleared, so users must update their input data accordingly.

.. _adding_elements.adding_variables:

Adding Variables
----------------

Variables can be added in the ``construct_variables`` method of the element constructor classes.
The ``add_variable`` method is called in the following way:

.. code-block:: python

    self.zen_model.add_variable(
        name="flow_import",
        index_sets=self.create_custom_set(["set_carriers", "set_nodes", "set_time_steps_operation"]),
        bounds=(0,np.inf),
        doc="node- and time-dependent carrier import from the grid",
        unit_category={"energy_quantity": 1, "time": -1},
    )

First, the ``model`` parameter is passed, which is the linopy model that the variable will be added to.
Then, the ``name`` of the variable is defined, in this case ``flow_import``.
The ``index_sets`` parameter is set to a custom set that is created with the ``create_custom_set`` method.
In case that a single set is used, it can be passed directly: ``index_sets=sets["set_years"]``.
The ``bounds`` parameter is set to ``(0, np.inf)``, which means that the variable can take any non-negative value.
If you do not specify the bounds, the variable will be unbounded.
The ``unit_category`` parameter is a dictionary that defines the unit of the variable.
Thereby, we can infer the unit of the variable from the unit categories of the parameters.

.. tip::

    Binary and integer variables can be added in the same way,
    but with the ``binary=True`` or ``integer=True`` parameter, respectively.
    Compare for example the ``technology_installation`` variable in the ``Technology`` class.

.. note::

    The variables are available in the constraint rules through the ``self.zen_model.variables[<variable_name>]`` attribute.

.. _adding_elements.adding_constraints:

Adding Constraints
------------------

Constraints can be added in the ``construct_constraints`` method of the element constructor classes.
Each class has a corresponding ``<Classname>Rules`` class that contains the rules for the constraints.
A rule is called with the corresponding rule name, e.g., ``rules.constraint_availability_import_export()``.

Please follow the constraint guide in :ref:`linopy.linopy`.

.. tip::

    You can add multiple constraints in the same rule,
    for example ``constraint_availability_import`` and ``constraint_availability_export`` in
    :py:class:`AvailabilityImportExportConstraint <zen_garden.elements.carrier.constraints.availability_import_export_constraint.AvailabilityImportExportConstraint>`.
    The rule of thumb is to add all constraints that are related to the same topic in the same rule
    to reuse the code and avoid duplication. If the constraints are too different, it is better to create a new constraint method.
