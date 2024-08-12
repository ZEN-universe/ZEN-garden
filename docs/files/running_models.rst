################
Running a model
################

Running ZEN-garden from a terminal
==============

To run ZEN-garden from a terminal, activate the environment where you installed ZEN-garden. Per default, the environment will be called ``zen-garden-env``::

  conda activate zen-garden-env

Make sure to change your path to the working directory, i.e. the directory that contains the ``config.json``. This directory will also be used to save the results::

  cd /path/to/your/data

.. note::
    You can create the data folder in the ZEN-garden root folder, but it will not be uploaded to Github (it is in the ``.gitignore`` file).
    This way, you can keep your data separate from the repository.

Execute the following lines to run ZEN-garden as a module::

  python -m zen_garden

When running the previous line, ZEN-garden will attempt to run the dataset specified in ``analyis/dataset`` in ``config.json``. You can change the dataset via the dataset argument::

  python -m zen_garden --dataset=<my_dataset>

If you have multiple ``config.json`` files in your working directory, you can specify the file you want to use with the ``config`` argument::

  python -m zen_garden --config=<my_config.json> --dataset=<my_dataset>

To test if the setup is working correctly, you can copy a dataset example and the ``config.json`` from the ``dataset_examples`` folder to the data folder and run the following command::

  python -m zen_garden --dataset=<example_dataset>
PyCharm configurations
==============

To execute ZEN-garden with the PyCharm IDE you can use the configuration setup which can be found next to the run button, and click on "Edit configurations.." to edit or add a configuration.

.. image:: ../images/pycharm_configuration.png
    :alt: creating zen-garden configurations in pycharm

Add a new configuration by clicking on the "+" button on the top left corner of the window. Choose ´´Python´´ as a type. You can name the configuration however you like. The important settings are:

- Change "Script Path" to "Module name" and set it to "zen_garden"
- Set the Python interpreter to the Conda environment that was used to install the requirements and ZEN-garden as a package. Per default, the environment will be named "zen-garden-env". **Important**: This setup will only work for Conda environments that were also declared as such in PyCharm; if you set the path to the Python executable, you will have to create a new PyCharm interpreter first.
- Set the "Working directory" to the path that contains the ``config.json``. This directory will also be used to save the results.

In the end, your configuration to run ZEN-garden as a module should look similar to this:

.. image:: ../images/pycharm_run_module.png
    :alt: run module

VS code configuations
==============

To run ZEN-garden as a module in VS code follow these steps:

- select the correct interpreter: Press ctrl + shift + p to open the command palette (if you're on Windows or Linux), and enter ``Python: Select interpreter`` and make sure that the correct conda environment is selected. Per default, the conda enivronment will be called ``zen-garden-env``.
- Create a new file in the folder ``./.vscode/`` called ``launch.json`` with the following content:

.. code-block:: JSON

  {"configurations": [
      {
        "name": "Python: ZEN-Garden", 
        "type": "python", 
        "cwd":"<path to folder with config.py>", 
        "request": "launch", "module": "zen_garden", 
        "console": "integratedTerminal"
      }
    ]
  }






