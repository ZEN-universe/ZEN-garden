.. _features.features:

########
Features
########

In the following, we provide an overview of the high-level features of 
ZEN-garden. For each feature, we provide a reference to the section in the model 
description where the feature is described in detail.

The core features are:

1. :ref:`Multiple pre-defined objective functions <features.objectives>`
2. :ref:`Flexible spatial resolution <features.spatial>`
3. :ref:`Flexible temporal resolution <features.temporal>`
4. :ref:`Variable foresight horizon <features.foresight>`
5. :ref:`Emission limits and budgets <features.emissions>`
6. :ref:`Detailed technology representations <features.technology>`
7. :ref:`Clean input formats <features.inputs>`
8. :ref:`Scenario creation tool <features.scenario>`
9. :ref:`Unit transformations and consistency checks <features.units>`
10. :ref:`Multiple supported solvers <features.solvers>` 
11. :ref:`input_handling.scaling algorithms for enhanced numerical stability <features.scaling>`
12. :ref:`Analysis and visualization tools <features.analysis>`


.. _features.objectives:

Multiple pre-defined objective functions
----------------------------------------

The user can flexibly choose between the available objective functions 
(:ref:`math_forumlation.objective`).

The available objective functions are:

* Minimization of the cumulative net-present cost of the energy system over the 
  entire planning horizon.
* Minimization of the cumulative carbon emissions of the energy system over the 
  entire planning horizon.

Additional objective functions can be added to the optimization problem.


.. _features.spatial:

Flexible spatial resolution
---------------------------

**Spatial resolution:**

The user can flexible define the spatial resolution of their model,
where each geographical regions is represented by a single node. The set of 
nodes is defined in the input data. The :ref:`system settings <configuration.system>` (``system.json``) allows 
users to flexibly select subsets of the set of nodes included in the input data.

**Network:**

Edges connect the nodes. Per default, edge distances are computed as the 
haversine distance between the nodes they connect. For each transport technology 
the default values can be overwritten with technology-specific edge distances.


.. _features.temporal:

Flexible temporal resolution
----------------------------

**Interyearly resolution:**

ZEN-garden optimizes the design and operation of energy systems over 
multi-year time horizons. The reference year, the number of years, and the 
interyearly resolution of the model can be flexibly modified in the system 
configuration (``system.json``). 

**Intrayearly resolution:**

Per default, the intrayearly resolution is set at to an hourly resolution and 
considers 8760 h/a. Time-series aggregation allows users to change the 
intrayearly resolution to reduce model complexity (see :ref:`tsa.tsa`). 
Timeseries aggregation methods are available in ZEN-garden via the tsam package. 
Timeseries which should not impact the clustering can be excluded by the user.
Moreover, a novel formulation of the constraints describing the storage levels 
enables users to capture both long- and short-term storage operation despite 
applying aggregation methods.


.. _features.foresight:

Variable foresight horizon
---------------------------

The transition pathway can be optimized with perfect foresight, i.e., all years 
optimized together, or myopic foresight, i.e., the optimization horizon is 
reduced. The foresight and decision horizon lengths can be flexibly defined in 
the system configuration (``system.json``).


.. _features.emissions:

Emission limits and budgets
----------------------------

Emissions are determined based on the carrier- and technology-specific carbon 
intensities defined in the input data (:ref:`math_formulation.emissions_objective`).

**Decarbonization pathway:**

The decarbonization of the energy system can be modelled via annual carbon 
emission targets or a carbon emission budget. It is also possible, to combine 
annual carbon emission targets with a carbon emission budget. Furthermore, a 
carbon emission price can be introduced as a market-based instrument to reduce 
carbon emissions.

The annual emission target and the emission budget can be relaxed by introducing 
a carbon emission overshoot price for the annual carbon emissions targets, or 
the carbon emission budget, respectively. The overshoot price determines the 
penalty term that is added to the objective function.

For more information see :ref:`math_formulation.emissions_constraints`.


.. _features.technology:

Detailed technology representations
-----------------------------------

The modular structure of ZEN-garden allows for a flexible definition of the 
technology-specific characteristics. General technology features are defined in 
the technology class. Technology-specific characteristics are defined in the 
corresponding child-classes.

Some technology functionalities requires binary variables; however, if the 
functionalities are not selected, the binary variables are not required and the 
optimization problem is a linear program. We highlight the binary variables in 
the following functionalities with the keyword "binary".

Three technology child classes are available to capture the behavior of 
conversion, storage, and transport technologies. Conversion technologies convert 
0-n input carriers into 0-m output carriers. Conversion factors describe the 
conversion of the input and output carriers with respect to the 
technology-specific unique reference carrier. Storage technologies store 
carriers over multiple time-steps; and transport technologies transport carriers 
between nodes via edges. Technology retrofitting is modeled via retrofitting 
technologies, a child class of conversion technologies. For more detailed 
information on the available technology types see :ref:`input_structure.technologies`.

**Technology features:**

* technology expansion constraints (minimum ("binary") and maximum capacity 
  addition, capacity limits, etc.)
* construction times
* option to account for existing technology capacities (brownfield optimization)
* option to include technology capacities which will be available in the future

**Conversion technology features:**

* flexible definition of multiple in- and output carriers
* minimum ("binary") and maximum load behavior
* option to model the capital expenditures via a piecewise-affine approximation 
  of non-linear cost-curves ("binary")
* retrofitting, e.g., with carbon capture units (:ref:`input_structure.conversion_technologies`)
* fuel substitution or fuel replacement (:ref:`input_structure.conversion_technologies`)

**Storage technology features:**

* natural inflow
* separate investment in power and energy capacity; option to set a fixed ratio 
  between power and energy capacity
* time series representation of short- and long-term storage operation with 
  self-discharge

**Transport technology features:**

* capital expenditures of transport technologies can be split into distance- and 
  capacity-dependent components ("binary")


.. _features.inputs:

Clean input formats
-----------------------

ZEN-garden completely separates the model code and the input data. No input data 
is **ever** hard-coded into the model code. Instead, all inputs are structured 
in the form of human-readable csv/json files (:ref:`input_handling.input_handling`).
Inputs are designed to be minimalistic, with no redundant values. At minimum, 
each parameter of every element must have a user-specified default value 
(:ref:`input_handling.attribute_files`). Default values are always set in the ``.json`` 
files and apply uniformly to all dimensions of the parameter (i.e. nodes, time 
steps, years, etc.). Default values can be overwritten to account variation of 
the parameter across dimensions using the  ``.csv`` files 
(:ref:`input_handling.overwrite_defaults`). 


.. _features.scenario:

Scenario creation tool
----------------------------

The scenario tool allows users to repeatedly run ZEN-garden using variations
of some base dataset. The desired variations for each scenario are specified in 
a ``.json`` scenario file (:ref:`t_scenario.t_scenario`). Scenarios created using
the scenario tool are fully parallelizable on high-performance computing 
clusters.


.. _features.units:

Unit transformations consistency checks
---------------------------------------

Raw data for energy system models may come in inconsistent units 
(e.g. megawatt vs. gigawatt) and a failure to properly convert these
will lead to modeling errors. To minimize errors, ZEN-garden requires
users to input units along with parameter values. At the start of each
model run, all units are converted to pre-defined base units base units 
(:ref:`input_handling.unit_consistancy`). Then, unit consistency checks ensure that the 
units are consistent throughout all parameters of all elements. When 
mismatches occur, the most probable wrong unit is stated when the consistency 
check fails. In the outputs, the units of all variables are inferred from the 
input parameters.


.. _features.solvers:

Multiple supported solvers
-------------------------------------

The optimization problem is formulated using linopy and is known to be 
compatible with the following solvers:

* HiGHs (open-source, ZEN-garden default)
* GLPK (open-source)
* Gurobi (commercial solver, but free academic licenses are available)


.. _features.scaling:

Scaling algorithms for enhanced numerical stability
---------------------------------------------------

A scaling algorithm is available which can be applied to reduce the matrix range 
(LHS) and the parameter range (RHS) of the optimization problem. Scaling is 
known to significantly reduce solution times, efficiently reducing numerical 
issues. Several scaling parameters are available to fine tune the algorithm and 
improve the algorithm performance (see :ref:`input_handling.scaling`).


.. _features.analysis:

Analysis and visualization tools
-----------------------------------------

The results of the optimization can be analyzed and visualized with the 
following functionalities:

1. Detailed results analysis with the results class (:ref:`t_analyze.results_code`)
2. Visualization of the results with the ZEN-explorer visualization platform 
   (:ref:`t_analyze.visualization`), both offline and online `<https://zen-garden.ethz.ch/>`_.
3. Comparison of two different results objects (:ref:`t_analyze.compare`)



