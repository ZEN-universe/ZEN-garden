.. _Building a model:
################
Building a model
################
You can build a new or adapt an existing model in ZEN-garden.
All ZEN-garden models follow the same structure of folders, ``.json`` files, and ``.csv`` files. The detailed structure is described in :ref:`Input data structure`.

Broadly speaking, the ``.json`` files define the default values of an element, which can be overwritten by the ``.csv`` files to specify data in more detail.
The ``.csv`` files are optional and can be omitted if the default values are sufficient. More detail on the use of default values and overwriting them can be found in :ref:`Attribute.json files` and :ref:`Overwriting default values`.

Note that you do not need to change any code when working with a new model. You only need to provide the necessary files in the correct format.

.. _Working with existing models:
Working with existing models
============================
The easiest way to get started with ZEN-garden is to use an existing model, obtained either from a collaborator or from one of the ZEN-garden dataset examples.

To run ZEN-garden with an example dataset (see `Dataset Examples <dataset_examples.rst>`_), execute the following line::

  python -m zen_garden --example=<example_name>

More information on how to run existing models, can be found in :ref:`Running a model`.

.. _Building a new model:
Building a new model
====================
Building entirely new models from scratch is also possible in ZEN-garden, but requires more work than using an existing model.
The following sections describe the necessary steps to build a new model.

1. Define the technology and carrier sets (:ref:`Technologies` and :ref:`Carriers`)
2. Build the folder structure with ``energy_system``, ``set_technologies``, and ``set_carriers`` folders (:ref:`Input data structure`)
3. Fill the energy system folder (:ref:`Energy system`)
4. Create the ``attributes.json`` file for each element (energy system, technology, carrier; :ref:`Attribute.json files`). The ``attributes.json`` files must contain a default value for all parameters of this element type (:ref`optimization_problem`)
5. If necessary, create the ``.csv`` files to specify the data in more detail (:ref:`Overwriting default values`)
6. Create the ``config.json`` and ``system.json`` files (:ref:`System, analysis, solver settings`)