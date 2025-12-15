.. _running.running:

###############
Running a model
###############

.. _running.run_model:

Run ZEN-garden
--------------

1. In a terminal or command prompt, change your directory to the ``<data>`` 
   folder (i.e. the directory that contains your model and 
   the ``config.json``)

   .. code:: shell
      
      cd <data>

2. Activate the ZEN-garden environment (see :ref:`instructions 
   <installation.activate>`).

3. Execute the following lines to run ZEN-garden::
      
      zen-garden --dataset="1_base_case"

   This will run the ZEN-garden model ``1_base_case``, provide that it was 
   downloaded as described in :ref:`Builing a model 
   <building.building>`. To run other datasets, replace ``"1_base_case"`` with 
   the name of (or path to) the dataset that you wish to run. Once the model 
   finishes, the output files will be stored in a new directory ``outputs`` 
   in your current working directory.


.. tip::
   ZEN-garden provides tools for easily reading and visualizing the outputs.
   For an introduction on how to analyze and visualize model outputs, see the
   :ref:`tutorial on analyzing outputs <t_analyze.t_analyze>`.


.. _running.additional_remarks:

Additional Remarks and Tips
---------------------------

1. ZEN-garden can also be run without providing any command line flags:

   .. code-block:: shell
    
      zen-garden
   
   In this case, ZEN-garden will look for the name of the dataset in the  
   ``analysis/dataset`` entry of ``config.json``. 

2. If you have multiple ``config.json`` files in your working directory, you can 
   specify the file you want to use with the ``config`` argument:

   .. code-block:: shell

      zen-garden --config="config.json" --dataset="1_base_case"


3. ZEN-garden can also be run from with a python script. You may find this to be 
   more familiar than using the command line. The following python code can be 
   used to run ZEN-garden, provided that it is run from within an environment 
   in which ZEN-garden is installed:

   .. code-block:: python
    
      from zen_garden import run
      import os

      os.chdir("<data>")
      run(dataset = "1_base_case")

   In this code, replace ``<data>`` with the path to your data folder 
   (i.e. the directory that contains the ``config.json``). The 
   ``dataset`` argument of the function ``run`` can be used to specify the 
   name of the dataset that should be run. 
   
   Note that the ``run`` function can take as an optional input arguments 
   any flags which can also be specified in the command line. For instance, in 
   order to specify a config file, you can use the code below. See 
   :func:`zen_garden.runner.run` for full documentation of the ``run`` 
   function.


   .. code-block:: python
    
      from zen_garden import run
      import os

      os.chdir("<data>")
      run(dataset = "1_base_case", config="<my_config.json>")