################
Building a model
################

A model in ZEN-garden consists entirely of input data files in the ``.json`` and ``.csv`` file formats. Building a model therefore refers to creating the
input files, in the proper input format, for ZEN-garden to read. The code uses these input files to construct, solve, and analyze the desired model. 
You can build a new or adapt an existing model in ZEN-garden. Note that you do not need to change any code when working with a new model. You only need to provide 
the necessary files in the correct format. 

At the highest level, two components are necessary in a ZEN-garden model:

1. A 'config.json' file which names dataset to be used, specifies solver configuration, and sets high-level analysis options. 
2. A folder (henceforth referred to as the ``dataset``) which contains detailed information about the desired model's network topology, conversion technologies, and transport technologies.

.. code-block::

    <data_folder>/
    |--<dataset>/
    |   |--energy_system/...
    |   |--set_carriers/...
    |   |--set_technologies/...
    |   `--system.json
    |
    `--config.json

The required file setup is shown on a high-level by the image above. The ``<data_folder>`` for ZEN-Garden contains the two core elements described above: a `config.json` file 
and a ``<dataset>`` folder. The dataset folder contains further files and subdirectories which specify model details. The detailed structure of this dataset is described 
in :ref:`Input data structure`. Broadly speaking, ``.json`` files within the dataset define the default values of an element, which can be overwritten by the ``.csv`` files to specify data in more detail.
The ``.csv`` files are optional and can be omitted if the default values are sufficient. More detail on the use of default values and overwriting them can be found in :ref:`Attribute.json files` and :ref:`Overwriting default values`.


.. _Working with existing models:
Working with existing models
============================

The easiest way to get started with ZEN-garden is to use an existing model, obtained either from a collaborator or from one of the ZEN-garden dataset examples.

To run ZEN-garden with an example dataset (see :ref:`dataset_examples`), execute the following line::

  python -m zen_garden --example=<example_name>

More information on how to run existing models, can be found in :ref:`Running a model`.

.. _Building a new model:
Building a new model
====================

Building entirely new models from scratch is more work than using an existing model. The existing model examples can be used as a template for the new model.
The following sections describe the necessary steps to build a new model.

1. Create a  ``config.json`` file. 
2. Create a dataset folder with the structure as shown above.
   
   - Define the technology and carrier sets (:ref:`Technologies` and :ref:`Carriers`)
   - Build the folder structure with ``energy_system``, ``set_technologies``, and ``set_carriers`` folders (:ref:`Input data structure`)
   - Fill the energy system folder (`Energy System <https://zen-garden.readthedocs.io/en/latest/files/zen_garden_in_detail/input_structure.html#energy-system>`_)
   - Create the ``attributes.json`` file for each element (energy system, technology, carrier; :ref:`Attribute.json files`). The ``attributes.json`` files must contain a default value for all parameters of this element type (:ref:`Sets, parameters, variables, and constraints`)
   - If necessary, create the ``.csv`` files to specify the data in more detail (:ref:`Overwriting default values`)
   - Create the ``system.json`` files (:ref:`System, analysis, solver settings`)

