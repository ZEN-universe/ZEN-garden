.. _tutorials_intro.intro: 

#############################
Tutorials: Overview and Setup
#############################


.. _tutorials_intro.overview:

Overview
--------

The ZEN-garden tutorials provide interactive, hands-on instructions for
the most important skills required to master ZEN-garden. All tutorials are based 
on the example model ``1_base_case`` from the included dataset examples.
:ref:`Tutorials Setup <tutorials_intro.setup>` describes how to install
and run this example model. All tutorials start from this point. 

We recommend new users to start with the  
:ref:`Tutorial 1: Analyze Results <t_analyze.t_analyze>`. This tutorial helps
users to explore and access the outputs that ZEN-garden. All subsequent 
tutorials are fully independent of one-another. Users are thus free to explore
these tutorials in any order. 

The provided tutorials are:

1.  :ref:`Tutorial 1: Analyze Results <t_analyze.t_analyze>` 


.. _tutorials_intro.setup:


Tutorials Setup
---------------

This module describes all the setup steps required in order to run the
tutorials. Each tutorial starts from the same starting point and can therefore
be run independently of all other tutorials.

To begin each tutorial:

1. Install ZEN-garden by following the instructions in the :ref:`installation 
   guide <installation.installation>`. 

2. Download the example dataset ``5_multiple_time_steps_per_year`` by following 
   the instructions for :ref:`using dataset examples <building.examples>`. When 
   following these instructions, replace  ``<example name>`` with 
   ``5_multiple_time_steps_per_year`` to download the appropriate data set.

3. Run the dataset example using the instructions for :ref:`running a model 
   <running.run_model>`. When following the instructions, replace 
   ``<dataset_name>`` with ``5_multiple_time_steps_per_year``.

Once run, ZEN-garden will begin printing output into the command window. Upon
successful completion, the following line will be printed:  

.. code:: text
   
   --- Optimization finished ---

A new directory called ``output``, which contains the ZEN-garden output files,
will be added in the current ``data`` directory.

The dataset ``5_multiple_time_steps_per_year`` simulates electricity and heat 
supply for a two-node system. The two nodes are Germany (DE) and Switzerland 
(CH). Electricity can only be supplied by solar photovoltaics, while heat can 
only be supplied by a natural gas boiler. Natural gas can be freely imported at 
each node. The model simulates three years (2023, 2024, and 2025) and 96 hours 
(4 days) per year. More detailed descriptions of the dataset, 
and all other example datasets, can be found in 
:ref:`dataset examples <dataset_examples.dataset_examples>`.
