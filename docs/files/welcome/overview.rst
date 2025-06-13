.. _overview.overview:

############
Overview
############

**Welcome to the ZEN-garden: Zero emissions Energy Networks!**

ZEN-garden is an open-source optimization framework designed to model long-term 
energy system transition pathways. To support current research focused on the 
transition of sector-coupled energy systems toward decarbonization, ZEN-garden 
is built upon two paradigms: Navigating the high dimensionality of 
sector-coupled transition pathway models and allowing users to design small, 
flexible, and robust input datasets.

ZEN-garden strictly separates the codebase from the input data to allow for 
strongly diverse case studies. Lightweight and intuitive input datasets and unit 
consistency checks reduce the possibility of user errors and facilitate the use 
of ZEN-garden for both new and experienced energy system modelers.

ZEN-garden is developed in Python and uses the Linopy package to formulate the 
optimization problem. The optimization problem can be solved with open-source 
and commercial solvers, such as HiGHs and Gurobi. ZEN-garden is licensed under 
the `MIT license <https://github.com/ZEN-universe/ZEN-garden/blob/main/LICENSE.txt>`_.

The key features of ZEN-garden are:

**Transition pathways**

* Greenfield and brownfield energy system transition pathways with a focus on 
  sector-coupled systems
* Detailed investigation of time coupling of transition pathways (myopic vs. 
  perfect foresight, technology expansion constraints, annual vs. cumulative 
  emission targets)
* Time series aggregation to reduce model complexity while maintaining 
  the representation of long- and short-term storage operation
* Flexible visualization of the results to support the interpretation of the 
  results

**Input data handling**

* Flexible modification and extension of input data with intuitive structure 
  and default values
* Unit conversion and consistency checks
* Scaling algorithm to improve the numerics of the optimization problem
* Parallelizable scenario analysis to investigate the impact of parameter 
  changes on the results
* Detailed dataset tutorials

ZEN-garden is developed and maintained by the `Reliability and Risk Engineering Laboratory <https://rre.ethz.ch/>`_ 
at `Eidgenössische Technische Hochschule Zürich (ETHZ) <https://ethz.ch/de.html>`_.


.. _overview.target:

Target group
===============
ZEN-garden can be an effective platform for energy system modelers, educators, 
and industrial and organizational users who are interested in planning the 
energy transition. ZEN-garden is not restricted to be used in a specific sector, 
such as the power system, since it does not include any industry-specific 
constraints. Furthermore, the separation of framework and input data and the 
intuitive visualization platform invite less coding-experienced users.