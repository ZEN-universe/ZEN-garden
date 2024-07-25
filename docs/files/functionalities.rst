################
Target group and functionalities
################

ZEN-garden is an open source energy system optimization model that can be used to investigate the optimal transition pathway of energy systems.
The model is developed in Python and uses the Linopy library to formulate the optimization problem.
The optimization problem can be solved with open-source and commercial solvers, including HiGHs and Gurobi.

Target group
------------
The target group of ZEN-garden are researchers, policy makers, and other stakeholders who are interested in planning the energy transition.

Functionalities
---------------
ZEN-garden is flexible and easily applicable to a variety of use cases. The following provides an overview of all currently available functionalities of ZEN-garden.
The functionalities are divided into the following categories:

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

The user can flexibly choose between the available objective functions, but users can also flexibly add problem-specific objective functions, and flexibly switch between the available options.

The available objective functions are:
- Minimization of the net-present of the net-present cost of the energy system over the entire planning horizon.
- Minimization of the greenhouse gas emissions of the energy system over the entire planning horizon.

**Emission domain**

- Decarbonization pathways: The user can introduce annual emission targets as well as an emission budget. The annual emission target and the emission budget can be relaxed by introducing an overshoot price. The overshoot price determines the penalty term that is added to the objective function if the annual emission target or the emission budget is exceeded.
The overshoot price for the annual emission target and the emission budget can be set independently from each other.
- Modelling emissions: Emissions can be defined for both, energy carrier consumption and technology operation.

**Spatial domain**

- Spatial resolution: The user can flexible define the spatial resolution of their model, where geographical regions are generally represented by nodes, which are connected by edges.
- Network: Per default, the distance between two nodes is computed as the Haversine distance, but it is also possible to define distances depending on the transport technology.

**Temporal domain**

- Intrayearly resolution: The user can define the temporal resolution of the model, which can be hourly, daily, weekly, monthly, or yearly.
- Interyearly resolution: The user can define the planning horizon of the model, which can be 1 year, 5 years, 10 years, 20 years, 30 years, 40 years, or 50 years.




