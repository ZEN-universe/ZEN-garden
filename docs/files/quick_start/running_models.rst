.. _running.running:

###############
Running a model
###############

.. _running.run_model:

Run ZEN-garden
--------------

1. In a terminal or command prompt, change your directory to the ``data`` folder 
   (i.e. the directory that contains your model and the ``config.json``)::
      
      cd /path/to/your/data

2. Activate the ZEN-garden environment (see :ref:`instructions 
   <installation.activate>`).

3. Execute the following lines to run ZEN-garden::
      
      python -m zen_garden --dataset="<dataset_name>"

   Here, replace ``<dataset_name>`` with the name of the dataset that you 
   wish to run (e.g. ``1_base_case``). Once the model finishes, the output files 
   will be stored in a new directory ``outputs`` in your current folder.


.. tip::
   ZEN-garden provides tools for easily reading and visualizing the outputs.
   For an introduction on how to analyze and visualize model outputs, see the
   :ref:`tutorial on analyzing outputs <t_analyze.t_analyze>`.


.. _running.additional_remarks:

Additional Remarks and Tips
---------------------------

1. ZEN-garden can also be run without providing any command line flags:

   .. code-block::
    
      python -m zen_garden
   
   In this case, ZEN-garden will look for the name of the dataset in the  
   ``analysis/dataset`` entry of ``config.json``. 

2. If you have multiple ``config.json`` files in your working directory, you can 
   specify the file you want to use with the ``config`` argument::

     python -m zen_garden --config=<my_config.json> --dataset=<my_dataset>


3. ZEN-garden can also be run from with a python script. You may find this to be 
   more familiar than using the command line. The following python code can be 
   used to run ZEN-garden, provided that it is run from within an environment 
   in which ZEN-garden is installed:

   .. code-block:: python
    
      from zen_garden.__main__ import run_module
      import os

      os.chdir("<path\to\data>")
      run_module(dataset = "<dataset_name>")

   In this code, replace ``<path\to\data>`` with the path to your data folder 
   (i.e. the directory that contains the ``config.json``) and replace 
   ``<dataset_name>`` with the name of the dataset you would like to run. 
   
   Note that the ``run_module`` function can take as an optional input arguments 
   any flags which can also be specified in the command line. For instance, in 
   order to specify a config file, you can use the code:


   .. code-block:: python
    
      from zen_garden.__main__ import run_module
      import os

      os.chdir("<path\to\data>")
      run_module(dataset = "<dataset_name>",
                 config="<my_config.json>")