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

Each component type is added the same way: as its own small,
declarative class in its own file, added to a list that the owning element
class (``Carrier``, ``Technology``, ``ConversionTechnology``, etc.) exposes as
a class attribute (``own_sets``, ``own_parameters``, ``variables``,
``constraints``). There is a single generic
:py:class:`ModelConstructor <zen_garden.model.constructor.ModelConstructor>`
that reads these lists off the element class and builds whatever they
declare; you never subclass the constructor itself.

All component classes share ``name``, ``doc``, and an optional immutable
``dependencies`` tuple. Dependencies contain component classes rather than
strings, making them easy to navigate in an editor. The framework orders
components topologically and rejects duplicate names, missing dependencies, and
cycles. The public ``build`` lifecycle is final; scientific model code provides
the focused hook documented for each component type instead of overriding
registration logic.

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
- ``symbol``: the raw LaTeX notation displayed in the generated notation table.
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
        name = "set_technologies"
        symbol = r"h \in \mathcal{H}"
        doc = "Technologies available to the energy system."

        @classmethod
        def get_data(cls, model_constructor):
            return model_constructor.model_schema.set_technologies

.. code-block:: python

    # zen_garden/elements/technology/sets/set_reference_carriers.py
    class SetReferenceCarriers(GenericSet):
        name = "set_reference_carriers"
        symbol = r"c_h^{\mathrm{ref}} \in \mathcal{C}"
        doc = "Reference carrier used by each technology."
        index_set = SetTechnologies

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
- ``symbol``: the raw LaTeX notation displayed in the generated notation table.
- ``indices``: the sets that the parameter is indexed by in the optimization model.
- ``doc``: the documentation for the parameter.
- ``unit_category``: a dictionary with the unit-dimensionality categories of the
  parameter and their power (``+1`` or ``-1``), used for unit conversion and
  validation (see :ref:`t_units.t_units`). For example ``{"energy_quantity": 1}``
  means the parameter is in energy-quantity units (e.g., MWh, m^3, kg). What is
  predefined is how the unit dimensionalities build the parameter unit, not the
  concrete unit itself, which comes from the input data.
- ``dependencies``: an optional tuple of other ``GenericParameter`` classes that
  must be registered first.
- ``default_value`` and ``default_unit``: optional metadata used together when an
  input may be omitted. ``default_unit`` names a parameter whose unit is reused.
- ``time_series``: set to ``True`` if the parameter is hourly resolved, so it is
  routed through time series aggregation instead of being read as a plain value.

For an hourly parameter that is read directly from the input data, this is the
entire class:

.. code-block:: python

    # zen_garden/elements/carrier/parameters/demand.py
    class Demand(GenericParameter):
        """Parameter which specifies the carrier demand."""

        name = "demand"
        symbol = r"d_{c,n,y,r}"
        indices = (
            "set_carriers",
            "set_nodes",
            "set_years",
            "set_representative_hours",
        )
        doc = "Carrier demand at each node and representative hour."
        unit_category = {"energy_quantity": 1, "time": -1}
        time_series = True

Hourly input files remain indexed by ``set_hours``. For a ``time_series``
parameter, preprocessing converts that input index into the separate
``set_years`` and ``set_representative_hours`` model indices declared above.

The class is added to the element's parameter list, e.g.
:py:data:`CARRIER_PARAMETERS <zen_garden.elements.carrier.parameters.CARRIER_PARAMETERS>`
in ``zen_garden/elements/carrier/parameters/__init__.py``, which the
``Carrier`` element class exposes through its ``own_parameters`` attribute.

Every scalar or array used as a mathematical coefficient must have a registered
``GenericParameter`` specification. Build switches remain configuration, while
index sets, predecessor relationships, and other index mappings remain
structural metadata.

The final base-class ``build`` method registers the parameter. Most parameters
need no methods: the defaults read the named input and collect its stored values.
Override only one of these hooks when needed:

- ``get_input_data(element) -> ParameterInputData | None`` controls preprocessing.
  Return the raw values and units that ZEN-garden should store on that element.
  Return ``None`` for a parameter calculated only during model construction.
- ``get_parameter_value(model_constructor) -> ParameterData`` controls the final
  value, dimensions, and units registered on the optimization model. Override it
  for a derived parameter calculated from already-registered parameters.

``ParameterInputData`` contains exactly ``values`` and ``units``.
``ParameterData`` contains the final ``values``, ``indices``, and optional
``units``.

.. note::

    Parameters that are *derived* from other registered parameters return
    ``None`` from ``get_input_data`` and override ``get_parameter_value``. See,
    for example,
    :py:class:`ExistingCapex <zen_garden.elements.technology.parameters.existing_capex.ExistingCapex>`,
    which aggregates existing capacities into a capex figure instead of
    reading one from the input data.

.. note::

    Parameters are available in ``get_expression`` and ``get_constraint`` through
    the ``model_constructor.optimization_model.parameters.<parameter_name>`` attribute.

Handling new and renamed parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you change the name of an existing parameter, document that in
:py:data:`PARAMETER_CHANGE_LOG <zen_garden.input.element_data_loader.PARAMETER_CHANGE_LOG>`.
This allows existing datasets to continue using the old name.

The keys of ``PARAMETER_CHANGE_LOG`` are always the new, current parameter name.
Add the new name as the key and the old name as the value. The code will then
search for the old name in the input data and use the new name in the optimization.

.. code-block:: python

    PARAMETER_CHANGE_LOG = {
        "updated_name": "outdated_name",
        # other parameters...
    }

For a new parameter that is missing from existing datasets, define
``default_value`` (only ``0``, ``1``, or ``"inf"`` are allowed) and
``default_unit`` on its :py:class:`GenericParameter
<zen_garden.model.component_types.parameter.GenericParameter>` subclass.
``default_unit`` names an existing parameter whose input unit should be reused.

.. code-block:: python

    class NewParameter(GenericParameter):
        # other parameter metadata...
        default_value = 0
        default_unit = "existing_parameter_name_with_same_unit"

In every major release, the log file is cleared, so users must update their input data 
accordingly.

.. _adding_elements.adding_variables:

Adding Variables
----------------

A variable is a class that inherits from
:py:class:`GenericVariable <zen_garden.model.component_types.variable.GenericVariable>`
and declares:

- ``name``: the name of the variable.
- ``symbol``: the raw LaTeX notation displayed in the generated notation table.
- ``indices``: the sets that the variable is indexed by.
- ``doc``: the documentation for the variable.
- ``unit_category``: a dictionary that defines the unit of the variable, in the
  same form as for parameters (see :ref:`adding_elements.adding_parameters`).
  This lets ZEN-garden infer the unit of the variable from the unit categories
  of the parameters.
- ``variable_type``: ``"continuous"`` (the default), ``"binary"``, or
  ``"integer"``.
- ``bounds``: optional constant bounds, such as ``(0.0, np.inf)``.
- ``get_bounds(cls, model_constructor, index_sets)``: an optional classmethod
  for bounds that depend on model data. If omitted, ``bounds`` is used.
- ``masks``: an optional tuple of
  :py:class:`GenericMask <zen_garden.model.component_types.mask.GenericMask>`
  subclasses. Their structural masks are aligned to the variable and combined
  with logical AND.

.. code-block:: python

    # zen_garden/elements/carrier/variables/flow_import.py
    class FlowImport(GenericVariable):
        """Variable for import flow."""

        name = "flow_import"
        symbol = r"F^{\mathrm{imp}}_{c,n,y,r}"
        indices = (
            "set_carriers",
            "set_nodes",
            "set_years",
            "set_representative_hours",
        )
        doc = "Carrier imported from the grid at each node and representative hour."
        unit_category = {"energy_quantity": 1, "time": -1}

        bounds = (0.0, np.inf)

The class is added to the element's variable list, e.g.
:py:data:`CARRIER_VARIABLES <zen_garden.elements.carrier.variables.CARRIER_VARIABLES>`
in ``zen_garden/elements/carrier/variables/__init__.py``, which the ``Carrier``
element class exposes through its ``variables`` attribute.

The base class's ``build`` classmethod resolves ``indices`` into the model's
index sets by calling ``model_constructor.create_custom_set(cls.indices)`` (for
a single set, use a one-item tuple such as ``indices = ("set_years",)``), then
registers the variable through
``model_constructor.optimization_model.add_variable()``.

.. tip::

    Binary and integer variables use ``variable_type = "binary"`` or
    ``variable_type = "integer"``, respectively.
    Compare for example the ``TechnologyInstallation`` variable in
    ``zen_garden/elements/technology/variables/technology_installation.py``.

.. tip::

    A variable can restrict *which* indices are actually constructed by listing
    reusable structural masks. For example, technology operation variables use
    ``masks = (PowerOperationMask,)``. A completely false mask keeps an
    addressable variable in the model API but creates no solver columns.
    Each reusable mask is a ``GenericMask`` subclass. Construction can be
    skipped entirely for the current configuration by overriding
    ``should_construct``.

The notation page uses ``.. model-component-table::`` with ``sets``,
``parameters``, or ``variables`` as its argument. The directive reads the
registered component classes directly, so updating ``symbol``, ``doc``,
``indices``, or ``unit_category`` updates the published table automatically.

Structural masks
^^^^^^^^^^^^^^^^

A structural mask describes indices at which a variable or constraint can
affect a feasible solution. It is distinct from a boolean condition used to
select a branch of an equation. Create reusable masks by subclassing
``GenericMask``:

.. code-block:: python

    class PowerOperationMask(GenericMask):
        name = "power_operation"
        indices = (
            "set_technologies",
            "set_location",
            "set_years",
            "set_representative_hours",
        )
        dependencies = (CapacityAvailableMask,)

        @classmethod
        def get_mask(cls, model_constructor, registry):
            capacity = registry[CapacityAvailableMask]
            representative_hours = list(
                model_constructor.optimization_model.sets[
                    "set_representative_hours"
                ]
            )
            return capacity.expand_dims(
                {"set_representative_hours": representative_hours}, axis=-1
            )

This mask keeps the existing yearly capacity dimension and expands it over the
common within-year operational hours. Dependencies are class references, which
support direct editor navigation. They are constructed in topological order
after all parameters and before variables. Cycles and
duplicate names raise errors. A mask tuple assigned to a variable or constraint is
always combined with AND; implement intentional OR logic inside a named
composite mask. Masks may depend on sets,
parameters, configuration, time mappings, and other masks, but not on variables
or constraints.

Masks use canonical ``set_technologies`` and ``set_location`` dimensions. The
``MaskRegistry`` projects these onto technology subsets and nodes or edges, and
broadcasts across additional target dimensions such as carriers.

Like parameters, variables, and constraints, every mask is defined in its own
module inside the ``masks`` package of the element that owns it. Shared
technology masks therefore live in ``technology/masks``, while storage- and
transport-specific masks live in their respective element packages. Each
package re-exports its mask classes and provides a typed ``*_MASKS`` list for
discoverability. Mask construction remains usage-driven: variables,
constraints, and mask dependencies determine which masks the registry builds.
Specialized elements may depend on masks from their parent element, but parent
elements must not import masks from specialized elements.

.. note::

    The variables are available in the constraint's ``get_constraint`` method through
    the ``model_constructor.optimization_model.variables[<variable_name>]`` attribute.

.. _adding_elements.adding_constraints:

Adding Constraints
------------------

A constraint is a class that inherits from
:py:class:`GenericConstraint <zen_garden.model.component_types.constraint.GenericConstraint>`,
declares a unique ``name``, and implements one ``get_constraint`` classmethod.
The final base-class ``build`` method omits empty index domains, calls
``should_construct``, obtains the single relation, applies the class's masks,
and registers it. The
class is added to the element's constraint list, e.g.
:py:data:`CARRIER_CONSTRAINTS <zen_garden.elements.carrier.constraints.CARRIER_CONSTRAINTS>`
in ``zen_garden/elements/carrier/constraints/__init__.py``, which the
``Carrier`` element class exposes through its ``constraints`` attribute.

Please follow the constraint guide in :ref:`linopy.linopy`.

A constraint class can declare an immutable tuple of structural masks:

.. code-block:: python

    class ExampleConstraint(GenericConstraint):
        name = "constraint_example"
        indices = ("set_assets", "set_years")
        masks = (AvailableAssetMask,)

        @classmethod
        def get_constraint(cls, model_constructor):
            return ...

An empty registered index set automatically omits the constraint, so
``should_construct`` is only needed for other configuration-dependent behavior.
``GenericConstraint.build`` aligns and combines these masks with the
constraint's own Linopy mask and finite right-hand side. Use
``should_construct`` only for a side-effect-free applicability check. Conditions
that formulate different mathematical branches should remain in
``get_constraint`` rather than being extracted into structural masks. The final
``build`` method verifies that the returned constraint dimensions exactly match
the declared ``indices`` tuple before registration.

Reusable Expressions
--------------------

Algebra reused by constraints or objectives, and logically independent parts of
a complicated equation, should be a named ``GenericExpression`` in the owning
element's ``expressions`` package. Each expression module contains one class:

.. code-block:: python

    class AnnualImportQuantity(GenericExpression):
        """Annual imported energy by carrier, node, and year.

        Dimensions: set_carriers, set_nodes, set_years.
        Dependencies: none.
        """

        name = "annual_import_quantity"
        indices = ("set_carriers", "set_nodes", "set_years")
        dependencies = ()

        @classmethod
        def get_expression(cls, model_constructor):
            model = model_constructor.optimization_model
            return (
                model.variables["flow_import"]
                * model.parameters.weight_representative_hours
            ).sum("set_representative_hours")

List the class in the owning element's ``*_EXPRESSIONS`` registry. Dependencies
refer to expression classes and may cross element packages. Construction rejects
duplicate names, missing dependencies, and dependency cycles. Consumers access
the result through ``optimization_model.expressions[<name>]``. ``indices`` lists
the returned public dimensions and omits Linopy's internal ``_term`` dimension;
the final ``build`` method verifies this declaration before registration. An
expression whose declared registered index set is empty is omitted automatically.
Document the expression's physical meaning in its class docstring and declare
its indices and named expression dependencies on the class.

Keep the complete formulation owned by one expression in ``get_expression``.
When a small class-local calculation improves readability, implement it as a
private static method. Use a module-level utility only when multiple components
consume the same independently meaningful domain operation; selecting different
formulations through a helper argument is not considered reuse.

Every concrete constraint module must contain exactly one class, one ``name``,
and one ``get_constraint``. Splitting related equations is intentional; share
their algebra through expressions rather than emitting multiple constraints
from one class.
