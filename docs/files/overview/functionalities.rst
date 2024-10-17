################
Target group and functionalities
################

**Welcome to the ZEN-garden!**
ZEN-garden is an open-source optimization framework designed to model long-term energy system transition pathways.
To support current research focused on the transition of sector-coupled energy systems toward decarbonization, ZEN-garden is built upon two paradigms:
Navigating the high dimensionality of sector-coupled transition pathway models and allowing users to design small, flexible, and robust input datasets.

ZEN-garden strictly separates the codebase from the input data to allow for strongly diverse case studies.
Lightweight and intuitive input datasets and unit consistency checks reduce the possibility of user errors and facilitate the use of ZEN-garden for both new and experienced energy system modelers.

ZEN-garden is developed in Python and uses the Linopy package to formulate the optimization problem.
The optimization problem can be solved with open-source and commercial solvers, such as HiGHs and Gurobi.
ZEN-garden is licensed under the `MIT license <https://github.com/ZEN-universe/ZEN-garden/blob/main/LICENSE.txt>`_.

The key features of ZEN-garden are:

**Transition pathways**

* Greenfield and brownfield energy system transition pathways with a focus on sector-coupled systems
* Detailed investigation of time coupling of transition pathways (myopic vs. perfect foresight, technology expansion constraints, annual vs. cumulative emission targets)
* Time series aggregation to reduce model complexity while maintaining the representation of long- and short-term storage operation
* Flexible visualization of the results to support the interpretation of the results

**Input data handling**

* Flexible modification and extension of input data with intuitive structure and default values
* Unit conversion and consistency checks
* Scaling algorithm to improve the numerics of the optimization problem
* Parallelizable scenario analysis to investigate the impact of parameter changes on the results
* Detailed dataset tutorials

Target group
===============
So far, ZEN-garden has been mainly used by researchers and students at ETH Zurich, but the separation of framework and input data and the intuitive visualization platform invite less coding-experienced users.
Without constraints on the type of investigated energy systems, ZEN-garden is not restricted to be used in a specific sector, such as the power system.
Therefore, ZEN-garden can be an effective platform for energy system modelers, educators, and industrial and organizational users who are interested in planning the energy transition.

Functionalities
===============

In the following, we provide an overview of the functionalities of ZEN-garden.
For each functionality, we provide a reference to the section in the model description where the functionality is described in detail.

1. Objective function
2. Spatial domain
3. Temporal domain
4. Emission domain
5. Technology domain
6. Input data
7. Solution algorithm
8. Github integration
9. Results analysis & visualization

Objective function
-----------------------

The user can flexibly choose between the available objective functions (:ref:`objective-function`).

The available objective functions are:

* Minimization of the cumulative net-present cost of the energy system over the entire planning horizon.
* Minimization of the cumulative carbon emissions of the energy system over the entire planning horizon.

Additional objective functions can be added to the optimization problem.

Spatial domain
-----------------------

**Spatial resolution:**

The user can flexible define the spatial resolution of their model, where each geographical regions is represented by a single node. The set of nodes is defined in the input data. The :ref:`system` (``system.json``) allows users to flexibly select subsets of the set of nodes included in the input data.

**Network:**

Edges connect the nodes. Per default, edge distances are computed as the haversine distance between the nodes they connect. For each transport technology the default values can be overwritten with technology-specific edge distances.

Temporal domain
-----------------------

**Interyearly resolution:**

ZEN-garden optimizes the design and operation of energy systems over multi-year time horizons.
The reference year, the number of years, and the interyearly resolution of the model can be flexibly modified in the system configuration (``system.json``).
Additional information on the representation of the temporal domain is provided in :ref:`Time series aggregation and representation`.

**Intrayearly resolution:**

Per default, the intrayearly resolution is set at to an hourly resolution and considers 8760 h/a. Timeseries aggregation methods are available via the tsam package and allow users to flexibly reduce model complexity. Timeseries which should not impact the clustering can be excluded by the user. Moreover, a novel formulation of the constraints describing the storage levels enables users to capture both long- and short-term storage operation despite applying aggregation methods. :ref:`Time series aggregation and representation` provides a detailed description of the available parameters.

The transition pathway can be optimized with perfect foresight, i.e., all years optimized together, or myopic foresight, i.e., the optimization horizon is reduced.
The foresight and decision horizon lengths can be flexibly defined in the system configuration (``system.json``).

Emission domain
-----------------------

**Emissions:**

Emissions are determined based on the carrier- and technology-specific carbon intensities defined in the input data (:ref:`emissions_objective`).

**Decarbonization pathway:**

The decarbonization of the energy system can be modelled via annual carbon emission targets or a carbon emission budget.
It is also possible, to combine annual carbon emission targets with a carbon emission budget.
Furthermore, a carbon emission price can be introduced as a market-based instrument to reduce carbon emissions.

The annual emission target and the emission budget can be relaxed by introducing an carbon emission overshoot price for the annual carbon emissions targets, or the carbon emission budget, respectively.
The overshoot price determines the penalty term that is added to the objective function.

For more information see :ref:`emissions_constraints`.

Technology domain
-----------------------

The modular structure of ZEN-garden allows for a flexible definition of the technology-specific characteristics. General technology features are defined in the technology class.
Technology-specific characteristics are defined in the corresponding child-classes.

Some technology functionalities requires binary variables; however, if the functionalities are not selected, the binary variables are not required and the optimization problem is a linear program.
We highlight the binary variables in the following functionalities with the keyword "binary".

Three technology child classes are available to capture the behaviour of conversion, storage, and transport technologies. Conversion technologies convert 0-n input carriers into 0-m output carriers.
Conversion factors describe the conversion of the input and output carriers with respect to the technology-specific unique reference carrier.
Storage technologies store carriers over multiple time-steps; and transport technologies transport carriers between nodes via edges.
Technology retrofitting is modeled via retrofitting technologies, a child class of conversion technologies. For more detailed information on the available technology types see :ref:`technologies`.

**Technology features:**

* technology expansion constraints (minimum ("binary") and maximum capacity addition, capacity limits, etc.)
* construction times
* option to account for existing technology capacities (brownfield optimization)
* option to include technology capacities which will be available in the future

**Conversion technology features:**

* flexible definition of multiple in- and output carriers
* minimum ("binary") and maximum load behavior
* option to model the capital expenditures via a piecewise-affine approximation of non-linear cost-curves ("binary")
* retrofitting, e.g., with carbon capture units (:ref:`Conversion Technologies`)
* fuel substitution or fuel replacement (:ref:`Conversion Technologies`)

**Storage technology features:**

* natural inflow
* separate investment in power and energy capacity; option to set a fixed ratio between power and energy capacity
* time series representation of short- and long-term storage operation with self-discharge

**Transport technology features:**

* capital expenditures of transport technologies can be split into distance- and capacity-dependent components ("binary")

Input Data
-----------------------

**Input data handling:**

* human-readable csv/json structure (:ref:`input_data_handling`)
* default values for every parameter of each element (:ref:`Attribute.json files`)
* overwrite default values with values in csv file (:ref:`Overwriting default values`)
* only specify relevant dimension: if same value for all nodes, omit node index. if same value for all years/time steps, omit time/year index

**Scenario analysis:**

* overwrite values with values in scenario file (:ref:`scenario_analysis`)
* parallelizable on high-performance computing clusters

**Unit handling:**

* convert all units to base units (:ref:`Unit consistency`)
* unit consistency checks that the units are consistent throughout all parameters of all elements
* The most probable wrong unit is stated when the consistency check fails
* units of variables are inferred from parameters


Solution Algorithm
-----------------------

**Solvers:**

The optimization problem is formulated using linopy and is known to be compatible with the following solvers:

* HiGHs (open-source, ZEN-garden default)
* GLPK (open-source)
* Gurobi (commercial solver, but free academic licenses are available)

**Scaling algorithm:**

A scaling algorithm is available which can be applied to reduce the matrix range (LHS) and the parameter range (RHS).
Scaling is known to significantly reduce solution times, efficiently reducing numerical issues. Several scaling parameters are available to fine tune the algorithm and improve the algorithm performance. For more details see :ref:`Scaling`.

Github integration
-----------------------

ZEN-garden is hosted on `Github <https://github.com/ZEN-universe/ZEN-garden>`_ and is open-source.
The installation of ZEN-garden is possible via pip or by cloning and forking the repository (:ref:`installation`).
Users can contribute to the development of ZEN-garden by creating a pull request on Github.
Automated tests are implement to test the key functionalities of the code. We ask contributors to test their code locally before creating a pull request and add tests for new functionalities.

Results analysis & visualization
-----------------------

The results of the optimization can be analyzed and visualized with the following functionalities:

1. Detailed results analysis with the results class (:ref:`Accessing results`)
2. Visualization of the results with the ZEN-explorer visualization platform (:ref:`Visualization`)
3. Comparison of two different results objects (:ref:`Comparing results`)


