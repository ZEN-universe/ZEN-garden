#########
Debugging
#########


This section requires that ZEN-Garden repository is forked as described in (see :ref:`Install for developers`). When the repository is forked, the code from the fork will be used to execute the module whenever ZEN-garden is called. This section describers how to run and debug the ZEN-garden module after modifying the code base.

There are two ways to run/debug ZEN-garden as a developer: 

1. Import and run the module the module in a python script.
2. Use terminal commands, as in :ref:`Running ZEN-garden as a Model-User`. In this case, the python IDE (e.g. VSCode or PyCharm) need to be specially configured to enable debugging.  
  
  
The sections below describe each of these two methods in detail. 

.. note::
    When running ZEN-Garden as a developer, always make sure that the python environment is up to date with the current model version. This means that the ZEN-garden environment should be re-installed each time you pull a new version from the repository. Otherwise, dependency changes may result in errors.

Run ZEN-garden using a Python Script
====================================

ZEN-Garden can be run and debugged by importing the module in a python script and calling the Run_Module function. This can be done via the following code:

.. code-block:: python

  from zen_garden.__main__ import run_module
  import os

  os.chdir("<path\to\data>")
  run_module(dataset = "<dataset_name>")


The script reflects two core requirements for running ZEN-garden. First, ZEN-garden must be executed from the directory in which the model input data is located. Second, the dataset which to use in the model needs to be specified. 

The ``run_module`` is equivalent to running the model via the command line. All command line flags can be specified as optional keyword arguments in the function. The two codes below demonstrate, for instance how to run an example problem and specify a config file, respectively.

.. code-block:: python

  from zen_garden.__main__ import run_module
  import os

  os.chdir("<path\to\data>")
  run_module(example="1_base_case")

.. code-block:: python

  from zen_garden.__main__ import run_module
  import os

  os.chdir("<path\to\data>")
  run_module(dataset = "<dataset_name>", 
             config="<my_config.json>")



The advantage of running ZEN-garden as a script is that standard debug functionalities can be easily applied to the model.


Run ZEN-garden from the Terminal
================================

Alternatively, developers may also run ZEN-garden using terminal commands as described in :ref:`Running ZEN-garden as a Model-User`. In this case, special configurations need to be set in the python IDE being used to enable debugging. These configurations are described below for two common IDEs: PyCharm and VSCode.

PyCharm configurations
----------------------

To execute ZEN-garden with the PyCharm IDE you can use the configuration setup which can be found next to the run button, and click on "Edit configurations.." to edit or add a configuration.

.. image:: images/pycharm_configuration.png
    :alt: creating zen-garden configurations in pycharm

Add a new configuration by clicking on the "+" button on the top left corner of the window. Choose ´´Python´´ as a type. You can name the configuration however you like. The important settings are:

- Change "Script Path" to "Module name" and set it to "zen_garden"
- Set the Python interpreter to the Conda environment that was used to install the requirements and ZEN-garden as a package. Per default, the environment will be called ``zen-garden-env``. **Important**: This setup will only work for Conda environments that were also declared as such in PyCharm; if you set the path to the Python executable, you will have to create a new PyCharm interpreter first.
- Set the "Working directory" to the path that contains the ``config.json``. This directory will also be used to save the results.

In the end, your configuration to run ZEN-garden as a module should look similar to this:

.. image:: images/pycharm_run_module.png
    :alt: run module

Once these configurations are set, the standard ``run`` and ``debug`` buttons of the PyCharm IDE can be used. When pressed, these buttons will create and execute the appropriate terminal commands for running and debugging ZEN-Garden, respectively. Command line flags can be typed into the ``Script Parameters`` field of the Run/Debug configurations.

VS code configurations
----------------------

To debug ZEN-garden with VSCode, follow these steps:

- select the correct interpreter: Press ctrl + shift + p to open the command palette (if you're on Windows or Linux), and enter ``Python: Select interpreter`` and make sure that the correct conda environment is selected. Per default, the conda environment will be called ``zen-garden``.
- Create a new file in the folder ``./.vscode/`` called ``launch.json`` with the following content:

.. code-block:: JSON

    {   
        "version": "0.2.0",
        "configurations": [
        {
          "name": "Python: ZEN-Garden",
          "type": "debugpy",
          "cwd":"<path to folder with dataset>",
          "request": "launch", "module": "zen_garden",
          "console": "integratedTerminal"
        }
      ]
    }


To debug ZEN-Garden, select ``Python Debugger: Debug using launch.json`` from the debug menu as shown in the figure. Note that no command line flags can be entered. The dataset must therefore be specified in the config.json file which is located in the dataset folder.

.. image:: images/VSCode_Debug.png
    :alt: VSCode Debug





