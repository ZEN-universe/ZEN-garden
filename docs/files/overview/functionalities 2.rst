################
Target group and functionalities
################

ZEN-garden is an open source energy system optimization model that can be used to investigate the optimal transition pathway of energy systems.
ZEN-garden is developed in Python and uses the Linopy library to formulate the optimization problem.
The optimization problem can be solved with open-source and commercial solvers, such as HiGHs and Gurobi.

The key features of ZEN-garden are:
* Flexible modification of input data
* Unit handling: Unit conversion and consistency checks
* Scaling algorithm to improve the numerics of the optimization problem
* Modular code structure allowing for a high level of flexibility
* Easy modification and extension of existing problem formulation
* Detailed dataset tutorials
* Greenfield vs. brownfield optimization considering existing technology capacities

Target group
------------
The target group of ZEN-garden are researchers, policy makers, and other stakeholders who are interested in planning the energy transition.

Functionalities
---------------

In the following, we provide an overview of the functionalities of ZEN-garden.
For each functionality, we provide a reference to the Section in the model description, where the functionality is described in detail.

1. Objective function
2. Emission domain
3. Spatial domain
4. Temporal domain
5. Technology domain
6. Input data
7. Solution algorithm
8. Automated testing
9. Results analysis & visualization

**Objective function**

The user can flexibly choose between the available objective functions (:ref:`objective-function`).

The available objective functions are:
* Minimization of the net-present of the net-present cost of the energy system over the entire planning horizon.
* Minimization of the greenhouse gas emissions of the energy system over the entire planning horizon.

Additional objective functions can be added to the optimization problem (ref).

**Emission domain**

Emissions: Emissions are determined based on the carrier- and technology-specific carbon intensities defined in the input data (:ref:`emissions_objective`).

Decarbonization pathway: The decarbonization of the energy system can be modelled via annual carbon emission targets or a carbon emission budget.
It is also possible, to combine annual carbon emission targets with a carbon emission budget.
The annual emission target and the emission budget can be relaxed by introducing an carbon emission overshoot price for the annual carbon emissions targets, or the carbon emission budget, respectively.
The overshoot price determines the penalty term that is added to the objective function (ref).

**Spatial domain**

Spatial resolution: The user can flexible define the spatial resolution of their model, where each geographical regions is represented by a single node. The set of nodes is defined in the input data. The :ref:`system` (``system.json``) allows users to flexibly select subsets of the set of nodes included in the input data.

Network: Edges connect the nodes. Per default, edge distances are computed as the Haversine distance between the nodes they connect. For each transport technology the default values can be overwritten with technology-specific edge distances.

**Temporal domain**

*Interyearly resolution:*
ZEN-garden optimizes the design and operation of energy systems over multi-year time horizons. The reference year, the number of years, and the interyearly resolution of the model can be flexibly modified in the system configuration (``system.json``). Additional information on the representation of the temporal domain is provided in :ref:`Time series aggregation and representation`.

*Intrayearly resolution:* 
Per default, the interyearly resolution is set at to an hourly resolution and considers 8760 h/a. Timeseries aggregation methods are available via the tsam package and allow users to flexibly reduce model complexity. Timeseries which should not impact the clustering can be excluded by the user. Moreover, a novel forumlation of the constraints describing the storage levels enables users to capture both, long- and short-term storage operation despite applying aggregation methods. :ref:`Time series aggregation and representation` provides a detailed description of the available parameters.

**Carrier domain**

Feedstocks and energy carriers are modeled as carriers.

**Technology domain**

The modular structure of ZEN-gardens allows for a flexible definition of the technology-specific characteristics. General technology features are defined in the technology class. Technology-specific characteristics are defined in the corresponding child-classes.
Three technology child-classess are available to capture the behaviour of conversion, storage, and transport technologies. Conversion technologies convert 0-n input carriers into 0-m output carriers. Conversion factors describe the conversion of the input and output carriers with respect to the technology-specific unique reference carrier. Storage technologies store carriers over multiple time-steps; and transport technologies transport carriers between nodes via edges. Technology retrofitting is modeled via retrofitting technologies, a child class of conversion technologies. For more detailed information on the available technology types see :ref:`technologies`.

Technology features:
* technology expansion constraints (minimum and maximum capacity, capacity limits, etc.)
* construction times
* option to account for existing technology capacities (brownfield optimization)
* option to include technology capacities which will be available in the future

Conversion technology features:
* flexible definition of multiple in- and output carriers
* min- and max load behavior
* option to model the capital expenditures via a piecewise-affine approximation of non-linear cost-curves
* retrofitting e.g. with carbon and capture units (ref)
* fuel substitution or fuel replacement (ref)

Storage technology features:
* Natural inflow

Transport technology features:
* capital expenditures of transport technologies can be split into distance- and capacity-dependent components
* default edge distances can be replaced by transport-technology-specific transport distances

**Input Data**

* Human-readable csv/json structure (ref)
* Paradigm: only specify the input data that you need to specify (ref)
    * default values for every parameter of each element (technology, carrier)
    * overwrite default values with values in csv file
    * only specify relevant dimension: if same value for all nodes, omit node index. if same value for all years/time steps, omit time/year index
* unit handling (ref)
    * associated unit string for each parameter of each element
    * convert to base units through linear combination
    * unit consistency checks that the units are consistent throughout all parameters of all elements
    * The most probable wrong unit is stated when the consistency check fails
    * units of variables are inferred from parameters → construction of energy_quantity units
* Option to linear interpolation of annual parameter values (ref)


**Solution Algorithm**

*Solvers:*
 The optimization problem is formulated using linopy and is known to be compatible with the following solvers:
* HiGHs (open-source)
* GLPK (open-source)
* Gurobi (commercial solver, but free academic licenses are available)

*Solution strategies:*
* Perfect foresight (ref)
* Rolling horizon with flexible definition of foresight and decision horizon (ref)

*Scaling algorithm*
A scaling algorithm is available which can be applied to reduce the matrix range (LHS) and the parameter range (RHS). Scaling is known to significantly reduce solution times, efficiently reducing numerical issues. Several scaling parameters are avilable to fine tune the algorithm and improve the algorithm performance. For more details see (ref).

**Automated testing**

Automated tests are implement to test the key-functionalities of the code.


**Results analysis & visualization**




