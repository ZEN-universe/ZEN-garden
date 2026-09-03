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

Each of the four component types is added the same way: as its own small,
declarative class in its own file, added to a list that the owning element
class (``Carrier``, ``Technology``, ``ConversionTechnology``, etc.) exposes as
a class attribute (``own_sets``, ``own_parameters``, ``variables``,
``constraints``). There is a single generic
:py:class:`ModelConstructor <zen_garden.model.constructor.ModelConstructor>`
that reads these lists off the element class and builds whatever they
declare; you never subclass the constructor itself.

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

A set is a class that inherits from
:py:class:`GenericSet <zen_garden.model.component_types.set.GenericSet>` and
declares:

- ``name``: the name of the set, which should be unique.
- ``doc``: the documentation for the set, which should be a string describing the set.
- ``index_set``: the set that is used as the index for the new set, if applicable.
- ``get_data(cls, model_constructor)``: a classmethod that returns the data for the set,
  which can be a list or a dictionary.

The class is added to the element's set list, e.g.
:py:data:`CONVERSION_TECHNOLOGY_SETS <zen_garden.elements.conversion_technology.sets.CONVERSION_TECHNOLOGY_SETS>`
in ``zen_garden/elements/conversion_technology/sets/__init__.py``, which the
``ConversionTechnology`` element class exposes through its ``own_sets``
attribute.

Two examples, one without and one with an ``index_set``:

.. code-block:: python

    # zen_garden/elements/energy_system/sets/set_technologies.py
    class SetTechnologies(GenericSet):
        name, doc = "set_technologies", "Set of technologies"

        @classmethod
        def get_data(cls, model_constructor):
            return model_constructor.model_schema.set_technologies

.. code-block:: python

    # zen_garden/elements/technology/sets/set_reference_carriers.py
    class SetReferenceCarriers(GenericSet):
        name, doc, index_set = (
            "set_reference_carriers",
            "Reference carriers indexed by technology",
            "set_technologies",
        )

        @classmethod
        def get_data(cls, model_constructor):
            return model_constructor.element_registry.get_attribute_of_all_elements(
                model_constructor.element_class, "reference_carrier"
            )

``SetTechnologies`` is not indexed by any set, while ``SetReferenceCarriers``
is indexed by the ``set_technologies`` set. That means that each technology
from the ``set_technologies`` set will have a corresponding entry in the
``set_reference_carriers`` set.

.. _adding_elements.adding_parameters:

Adding Parameters
-----------------

A parameter is a class that inherits from
:py:class:`GenericParameter <zen_garden.model.component_types.parameter.GenericParameter>`
and declares:

- ``name``: the name of the parameter, matched against the parameter names used
  in the ``attributes.yaml``/``.csv`` input files (see :ref:`input_structure.attribute_files`).
- ``indices``: the sets that the parameter is indexed by in the optimization model.
- ``doc``: the documentation for the parameter.
- ``unit_category``: a dictionary with the unit-dimensionality categories of the
  parameter and their power (``+1`` or ``-1``), used for unit conversion and
  validation (see :ref:`t_units.t_units`). For example ``{"energy_quantity": 1}``
  means the parameter is in energy-quantity units (e.g., MWh, m^3, kg). What is
  predefined is how the unit dimensionalities build the parameter unit, not the
  concrete unit itself, which comes from the input data.
- ``time_series``: set to ``True`` if the parameter is hourly resolved, so it is
  routed through time series aggregation instead of being read as a plain value.

For a parameter that is read directly from the input data, this is the entire
class:

.. code-block:: python

    # zen_garden/elements/carrier/parameters/availability_import_yearly.py
    class AvailabilityImportYearly(GenericParameter):
        """Parameter which specifies the yearly availability of carrier import."""

        name = "availability_import_yearly"
        indices = ("set_carriers", "set_nodes", "set_years")
        doc = "Parameter which specifies the yearly availability of carrier import"
        unit_category = {"energy_quantity": 1}

The class is added to the element's parameter list, e.g.
:py:data:`CARRIER_PARAMETERS <zen_garden.elements.carrier.parameters.CARRIER_PARAMETERS>`
in ``zen_garden/elements/carrier/parameters/__init__.py``, which the
``Carrier`` element class exposes through its ``own_parameters`` attribute.

You do not need to call anything yourself: the base class's
``store_input_data`` classmethod reads the values from the element's input
files with
:py:meth:`ElementDataLoader.extract_input_data <zen_garden.input.element_data_loader.ElementDataLoader.extract_input_data>`
during preprocessing, and its ``build`` classmethod registers them on the
optimization model. When resolving which indices to read per element,
``store_input_data`` automatically drops the dimension that is already implied
by the element itself (e.g. ``set_carriers`` for a parameter on the ``Carrier``
class, since each carrier's input files already live in that carrier's own
folder).

.. note::

    Parameters that are *derived* from other, already-registered model
    parameters instead of read from input files override ``build`` directly
    and leave ``store_input_data`` empty. See, for example,
    :py:class:`ExistingCapex <zen_garden.elements.technology.parameters.existing_capex.ExistingCapex>`,
    which aggregates existing capacities into a capex figure instead of
    reading one from the input data.

.. note::

    The parameters are available in the constraint's ``build`` method through
    the ``model_constructor.optimization_model.parameters.<parameter_name>`` attribute.

Logging new and changed parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you add a new parameter or change the name of an existing one, please document 
that in
:py:data:`PARAMETER_CHANGE_LOG <zen_garden.input.element_data_loader.PARAMETER_CHANGE_LOG>`.
The reason to add the name is that the new or changed parameters will be searched for 
in the input data, but are not available in the datasets of others. 
To avoid breaking changes, the new or changed parameters are documented in the log file 
and then equivalent parameters are found without breaking the code.

The keys of ``PARAMETER_CHANGE_LOG`` are always the new, current parameter name.
There are two possible options:

1. You change the name of an existing parameter, e.g.,
   from ``outdated_name`` to ``updated_name``.
   In this case, you add the new name as the key and the old name as the value.
   The code will then search for the old name in the input data and use the new name in 
   the optimization.

.. code-block:: python

    PARAMETER_CHANGE_LOG = {
        "updated_name": "outdated_name",
        # other parameters...
    }

2. You add a new parameter that had not existed before, e.g., ``new_parameter``.
   In addition to the new name, you also provide the ``default_value`` 
   (only `0`, `1`, or `inf` are allowed), and another parameter with the same 
   unit category that is used to infer the unit of the new parameter.

.. code-block:: python

    PARAMETER_CHANGE_LOG = {
        "new_parameter": {
            "default_value": 0,
            "unit": "existing_parameter_name_with_same_unit"
        },
        # other parameters...
    }

In every major release, the log file is cleared, so users must update their input data 
accordingly.

.. _adding_elements.adding_variables:

Adding Variables
----------------

A variable is a class that inherits from
:py:class:`GenericVariable <zen_garden.model.component_types.variable.GenericVariable>`
and declares:

- ``name``: the name of the variable.
- ``indices``: the sets that the variable is indexed by.
- ``doc``: the documentation for the variable.
- ``unit_category``: a dictionary that defines the unit of the variable, in the
  same form as for parameters (see :ref:`adding_elements.adding_parameters`).
  This lets ZEN-garden infer the unit of the variable from the unit categories
  of the parameters.
- ``get_bounds(cls, model_constructor, index_sets)``: an optional classmethod
  returning a ``(lower, upper)`` tuple. If omitted, the variable is unbounded.

.. code-block:: python

    # zen_garden/elements/carrier/variables/flow_import.py
    class FlowImport(GenericVariable):
        """Variable for import flow."""

        name = "flow_import"
        indices = ["set_carriers", "set_nodes", "set_time_steps_operation"]
        doc = "Variable for node- and time-dependent carrier import from the grid"
        unit_category = {"energy_quantity": 1, "time": -1}

        @classmethod
        def get_bounds(cls, model_constructor, index_sets):
            return 0.0, np.inf

The class is added to the element's variable list, e.g.
:py:data:`CARRIER_VARIABLES <zen_garden.elements.carrier.variables.CARRIER_VARIABLES>`
in ``zen_garden/elements/carrier/variables/__init__.py``, which the ``Carrier``
element class exposes through its ``variables`` attribute.

The base class's ``build`` classmethod resolves ``indices`` into the model's
index sets by calling ``model_constructor.create_custom_set(cls.indices)`` (in
case a single set is used, pass its name directly, e.g. ``indices =
["set_years"]``), then registers the variable through
``model_constructor.optimization_model.add_variable()``.

.. tip::

    Binary and integer variables can be added in the same way,
    but with the ``binary = True`` or ``integer = True`` class attribute, respectively.
    Compare for example the ``TechnologyInstallation`` variable in
    ``zen_garden/elements/technology/variables/technology_installation.py``.

.. tip::

    A variable can also restrict *which* indices are actually constructed by
    overriding ``get_mask`` (a boolean array over ``index_sets``) or skip
    construction entirely for the current configuration by overriding
    ``should_construct``. ``TechnologyInstallation`` uses both.

.. note::

    The variables are available in the constraint's ``build`` method through
    the ``model_constructor.optimization_model.variables[<variable_name>]`` attribute.

.. _adding_elements.adding_constraints:

Adding Constraints
------------------

A constraint is a class that inherits from
:py:class:`GenericConstraint <zen_garden.model.component_types.constraint.GenericConstraint>`
and implements a single ``build(cls, model_constructor)`` classmethod. The
class is added to the element's constraint list, e.g.
:py:data:`CARRIER_CONSTRAINTS <zen_garden.elements.carrier.constraints.CARRIER_CONSTRAINTS>`
in ``zen_garden/elements/carrier/constraints/__init__.py``, which the
``Carrier`` element class exposes through its ``constraints`` attribute.

Please follow the constraint guide in :ref:`linopy.linopy`.

.. tip::

    You can add multiple constraints in the same ``build`` method,
    for example ``constraint_availability_import`` and ``constraint_availability_export`` in
    :py:class:`AvailabilityImportExportConstraint <zen_garden.elements.carrier.constraints.availability_import_export_constraint.AvailabilityImportExportConstraint>`.
    The rule of thumb is to add all constraints that are related to the same topic in the same method
    to reuse the code and avoid duplication. If the constraints are too different, it is better to create a new constraint class.
