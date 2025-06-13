.. _building.building:

################
Building a model
################

A model in ZEN-garden consists entirely of input data files in the ``.json`` and 
``.csv`` file formats. Building a model therefore refers to creating the input 
files, in the proper input format, for ZEN-garden to read. The code uses these 
input files to construct, solve, and analyze the desired model. You can build a 
new or adapt an existing model in ZEN-garden. Note that you do not need to 
change any code when working with a new model. You only need to provide the 
necessary files in the correct format. 

At the highest level, two components are necessary in a ZEN-garden model:

1. A 'config.json' file which names dataset to be used, specifies solver 
   configuration, and sets high-level analysis options. 
2. A folder (henceforth referred to as the ``dataset``) which contains detailed 
   information about the desired model's network topology, conversion 
   technologies, and transport technologies. The name of this folder determines 
   the name of the dataset.


.. _building.file_structure_basic:

.. code-block:: text

    <data>/
    |--<dataset>/
    |   |--energy_system/...
    |   |--set_carriers/...
    |   |--set_technologies/...
    |   `--system.json
    |
    `--config.json

The required file setup is shown on a high-level by the image above. The 
``<data>`` folder for ZEN-Garden contains the two core elements described 
above: a `config.json` file and a ``<dataset>`` folder. The dataset folder 
contains further files and subdirectories which specify model details. The 
detailed structure of this dataset is described in :ref:`input_structure.input_structure`. 
Broadly speaking, ``.json`` files within the dataset define the default values 
of an element, which can be overwritten by the ``.csv`` files to specify data in 
more detail. The ``.csv`` files are optional and can be omitted if the default 
values are sufficient. More detail on the use of default values and overwriting 
them can be found in :ref:`input_handling.attribute_files` and :ref:`input_handling.overwrite_defaults`.

.. tip::
    If you create the ``data`` folder in the ZEN-garden root folder, it will not 
    be uploaded to Github (it is in the ``.gitignore`` file). That said, we 
    recommend keeping the data folder in a different location from the source 
    code to ensure continued separation between the model and input data.


.. _building.first_model:

Creating a first model
======================


.. _building.existing_model:

Existing models
^^^^^^^^^^^^^^^

The easiest way to build a new model is to start from an existing model. 
Existing models have all required files and file-formats to run ZEN-garden. 
Existing models can be obtained either from the dataset examples, the ZEN-models 
GitHub, or Collaborators. 

.. tip::
    Dataset examples are an easy and quick way to get started with your model. 
    Use the descriptions in :ref:`dataset_examples.dataset_examples` to find an example that best 
    suits your needs and use it as a template to build your own model. The 
    section below describes how to access the dataset examples.


.. _building.examples:

Using dataset examples
----------------------

The dataset examples provide small test datasets which are particularly useful 
for first-time model users. The dataset examples are described and documented in 
detail in :ref:`dataset_examples.dataset_examples`. To download one of the example datasets (e.g. 
"1_base_case"), use the following steps:

1. Create a new folder where to store the data (i.e. the ``data`` folder).
2. In a terminal or the command prompt, navigate to the newly created folder.
3. Activate the ZEN-garden python environment (see :ref:`instructions 
   <installation.activate>`).
4. Download the desired example data set using the command: 

   .. code-block::

       python -m zen_garden --download_example="<example name>"


   The desired example data set will be downloaded to the current working 
   directory. A full list of example data sets can be found in :ref:`dataset_examples.dataset_examples`. 

.. note::
    If done correctly, you should now see a new directory in the ``data`` folder
    whose name matches the ``<example_name>`` that you entered. This directory
    should have a file structure that matches the  :ref:`basic ZEN-garden input 
    structure <building.file_structure_basic>`

.. tip::
    The dataset examples, once downloaded, include a Jupyter notebook called 
    ``example_notebook.ipynb``. This notebook provides a tutorial and code 
    for quickly accessing the results of ZEN-garden. Use this notebook after 
    running the model to begin your data analysis. 


.. _building.zen_models:

ZEN-models repository
----------------------

Full-scale models from past studies are available on the `ZEN-models GitHUB Page 
<https://github.com/ZEN-universe/ZEN-models>`_. Each branch of this repository 
contains a new dataset. The version of ZEN-garden required to run this model 
is indicated in the README file of that branch. The ZEN-models page, for instance, 
contains a fully functional model of the European energy system. 

To use models from the ZEN-models repository, simply select the desired branch 
and download the `data`` folder from the repository.


.. _building.from_scratch:

Starting from scratch
^^^^^^^^^^^^^^^^^^^^^

Building entirely new models from scratch is more work than using an existing 
model. The existing model examples can be used as a template for the new model.
The following sections describe the necessary steps to build a new model.

1. Create a  ``config.json`` file. 
2. Create a dataset folder with the structure as shown above.
   
   - Define the technology and carrier sets (:ref:`input_structure.technologies` and 
     :ref:`input_structure.carriers`)
   - Build the folder structure with ``energy_system``, ``set_technologies``, 
     and ``set_carriers`` folders (:ref:`input_structure.input_structure`)
   - Fill the energy system folder (`Energy System 
     <https://zen-garden.readthedocs.io/en/latest/files/zen_garden_in_detail/input_structure.html#energy-system>`_)
   - Create the ``attributes.json`` file for each element (energy system, 
     technology, carrier; :ref:`input_handling.attribute_files`). The ``attributes.json`` 
     files must contain a default value for all parameters of this element type 
     (:ref:`notation.notation`)
   - If necessary, create the ``.csv`` files to specify the data in more detail 
     (:ref:`input_handling.overwrite_defaults`)
   - Create the ``system.json`` files (:ref:`configuration.configuration`)