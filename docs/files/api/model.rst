=====
Model
=====

General
=======
.. autosummary::
   :toctree: generated

   zen_garden.workflow.optimization_workflow
   zen_garden.workflow.optimization_step

.. toctree::
   :maxdepth: 1

   modules/zen_garden.config


Elements
========
.. autosummary::
    :toctree: generated

    zen_garden.model.element
    zen_garden.elements.energy_system
    zen_garden.elements.carrier
    zen_garden.elements.technology
    zen_garden.elements.conversion_technology
    zen_garden.elements.storage_technology
    zen_garden.elements.transport_technology
    zen_garden.elements.retrofitting_technology


Model construction
==================
.. autosummary::
    :toctree: generated

    zen_garden.model.schema
    zen_garden.model.constructor
    zen_garden.model.construction_service
    zen_garden.model.element_factory
    zen_garden.model.element_registry


Component types
===============
.. autosummary::
    :toctree: generated

    zen_garden.model.component_types.set
    zen_garden.model.component_types.parameter
    zen_garden.model.component_types.variable
    zen_garden.model.component_types.expression
    zen_garden.model.component_types.constraint


Constraints
===========
.. autosummary::
    :toctree: generated

    zen_garden.elements.carrier.constraints
    zen_garden.elements.conversion_technology.constraints
    zen_garden.elements.energy_system.constraints
    zen_garden.elements.retrofitting_technology.constraints
    zen_garden.elements.storage_technology.constraints
    zen_garden.elements.technology.constraints
    zen_garden.elements.transport_technology.constraints




Optimization model
==================
.. autosummary::
    :toctree: generated

    zen_garden.model.optimization_model
    zen_garden.model.registries.base
    zen_garden.model.registries.constraint
    zen_garden.model.registries.multi_index_helper
    zen_garden.model.registries.parameter
    zen_garden.model.registries.set
    zen_garden.model.registries.set_registry
    zen_garden.model.registries.variable
    zen_garden.model.time_steps
