.. _qs_summary.qs_summary:

############
Summary
############

The full procedure for running a ZEN-garden model is shown in the two scripts 
below. ZEN-garden can be used (1) with the command-line interface, or (2) 
by importing the package in a python script. 

The two examples below are equivalent. They both download the dataset examples
``1_base_case`` to the current working directory. Then, they run ZEN-garden 
on that dataset. Finally, they provide a simple example of how to analyze the
resulting model outputs. In the command-line interface, the model outputs 
can be explored using the zen-garden visualization platform. In a python script,
the zen-garden results class can be used to extract variable values and units.
Both types of analyses are described in detail in the 
:ref:`tutorial on analyzing outputs<t_analyze.t_analyze>`.


.. _qs_summary.cli:

Command line interface
-----------------------

The following commands can be used to run an example dataset using the 
command-line interface.

.. code:: bash

    zen-example --dataset="1_base_case"
    zen-garden --dataset="1_base_case"
    zen-visualization

The first line downloads the example dataset `1_base_cae` to the current 
working directory. The second line then runs ZEN-garden on this dataset.
Finally, the third line opens the visualization platform, which allows users
to explore the results of the model. If the visualization platform does not 
open automatically, you can open it manually by typing  http://localhost:8000/ 
in any browser of your choice.


.. _qs_summary.python:

Python script
--------------

The following script can be used to run ZEN-garden in python. This script 
is equivalent to the one above except that it uses the ``Results`` class for
analyzing the results rather than the visualization platform. 


.. code:: python

    from zen_garden.dataset_examples import download_example_dataset
    from zen_garden.__main__ import run_module
    from zen_garden.postprocess.results.results import Results

    # dataset name
    dataset = "1_base_case"

    # download example dataset to current working directory
    download_example_dataset(dataset)

    # run ZEN-garden
    run_module(dataset=dataset)

    # load results
    r = Results(f"./outputs/{dataset}")

    # extract optimal capacities
    print(r.get_total("capacity"))


