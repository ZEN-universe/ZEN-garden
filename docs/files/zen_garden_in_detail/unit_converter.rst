.. _t_units.t_units:

#############################
Unit conversion and consistency
#############################

ZEN-garden models describe physical processes, where numerical values are
associated with physical units. Input data may use different units for the
same physical quantity, while the optimization problem itself only accepts
numeric values. ZEN-garden therefore converts input values to a common set of
base units and checks that the units of parameters and elements are consistent.


Convert units to common base units
==================================

ZEN-garden defines a set of base units, which can be combined to represent each
dimensionality in the model. For example:

.. code-block:: text

    km => [distance]
    hour => [time]
    Euro => [currency]
    GW => [mass]^1 [length]^2 [time]^-3

Units with the same dimensionality can be converted by calculating a multiplier.
For example:

.. code-block:: text

    Euro/MWh => Euro/GW/hour
    (Euro/GW/hour)/(Euro/MWh) = (MW)/(GW) = 0.001

The numeric value is multiplied by this conversion factor.

Base units are defined in ``<dataset>/energy_system/base_units.yaml``. The
canonical format is:

.. code-block:: yaml

    unit:
      - hour
      - GW
      - km
      - megatons
      - megaEuro

The deprecated JSON format is still accepted for compatibility, but YAML should
be used for new datasets.

Input units are specified in the ``unit`` field next to ``default_value`` in
each ``attributes.yaml`` file (see :ref:`input_structure.attribute_files`).
Values in CSV input files are assumed to use the same unit as the corresponding
attribute.


Defining new units
==================

ZEN-garden uses `Pint <https://pint.readthedocs.io/en/stable/>`_, which already
defines most common units. Additional units can be defined in
``<dataset>/energy_system/unit_definitions.txt``:

.. code-block:: text

    Euro = [currency] = EURO = Eur
    pkm = [mileage] = passenger_km = passenger_kilometer

The first definition uses the existing ``[currency]`` dimensionality. A new
dimensionality can be introduced in the same way, for example
``pkm = [mileage]``.

When choosing base units:

1. All input units must be representable as a combination of the base units.
   Each base unit should only have an exponent of ``1``, ``-1``, or ``0``.
2. Base units must not be linearly dependent. For example, ``GW``, ``hour``,
   and ``GJ`` cannot all be selected as base units.
3. Base units must have unique dimensionalities. For example, do not use both
   ``MW`` and ``GW`` as base units.


Enforcing unit consistency
==========================

Converting values to common base units makes their magnitudes comparable, but
does not ensure that units are consistent across parameters and elements. For
example, an electrolyzer capacity could be defined in ``GW`` while its
investment cost uses ``Euro/(ton/hour)``.

ZEN-garden checks the units of parameters and elements. It connects
technologies to their reference carriers and checks whether the carrier units
are compatible with the technology parameters. If an inconsistency is found,
ZEN-garden reports the error and tries to identify the least common unit that
caused the mismatch.

After the consistency check, ZEN-garden derives the units of optimization
variables from the parameter units. In custom model components,
``variable.add_variable()`` uses ``unit_category`` to define the dimensionality,
for example:

.. code-block:: python

    unit_category={"energy_quantity": 1, "time": -1}

In results, the unit of a parameter or variable can be retrieved with
``r.get_unit(<variable/parameter name>)``; see
:ref:`t_analyze.results_code`.


Known issues with Pint
======================

* ``ton``: Pint normally uses ``ton`` for an imperial ton. ZEN-garden
  overwrites this definition to use the metric ton by default, so ``ton`` and
  ``tonne`` can be used interchangeably. To use imperial tons, set
  ``solver.define_ton_as_metric_ton: false`` in ``config.yaml``.

* ``h``: If ``h`` is interpreted as the Planck constant instead of hour, update
  the Pint version in the environment.
