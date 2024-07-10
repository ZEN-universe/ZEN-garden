# Modelling an Energy System - Tutorial
This tutorial demonstrates ZEN garden's functionalities by applying them to simple energy systems. To execute the datasets presented in this tutorial, you first need to install ZEN garden by following the instructions stated in the ["How to ZEN garden" manual](https://github.com/ZEN-universe/ZEN-garden/blob/main/documentation/how_to_ZEN_garden.md). Once the installation is completed, the [_dataset\_examples_ folder](https://github.com/RRE-ETH/ZEN-garden/tree/main/documentation/dataset_examples) must be copied into the _data_ folder of your ZEN garden python project (to run a dataset it must be located in the _data_ folder). If you have not yet copied the _config.py_ file from the [_testcases_ folder](https://github.com/RRE-ETH/ZEN-garden/tree/main/tests/testcases) into your _data_ directory, this must be done to select the individual dataset which should be executed.

The following dataset examples are sorted by increasing complexity of the functionalities, starting with the most basic features. In the following datasets, the newly introduced features are applied to the previous dataset such that the change caused by the added adaption can be seen in the results as clearly as possible.

## Flow Chart Conventions
The flow charts describe the energy system setup for each dataset by visualising all technologies defined in the dataset. Transparent technology symbols were not built by the optimizer. The numbers above the arrows represent the flow rates of the carriers (e.g., heat, electricity, etc.). For datasets with multiple time steps per year, the corresponding time step of the flow rates stated in the flow chart is included in the title (e.g., TS 50).

## Content
1. [1\_base\_case:](#1_base_case) most basic features to define an energy system
2. [2\_add\_photovoltaics:](#2_add_photovoltaics) add an additional conversion technology to the energy system
3. [3\_multi\_year\_optimization:](#3_multi_year_optimization) change basic time parameters such as the number of optimized years
4. [4\_variable\_demand:](#4_variable_demand) define time-/space-dependent parameters
5. [5\_reduced\_import\_availability:](#5_reduced_import_availability) additional example of defining time-/space-dependent parameters, resulting in the need for a transport technology
6. [6\_PWA\_nonlinear\_capex:](#6_PWA_nonlinear_capex) approximate non-linear capex by piece-wise affine (PWA) approximation
7. [7\_multiple\_time\_steps\_per\_year:](#7_multiple_time_steps_per_year) optimize multiple time steps per year
8. [8\_reduced\_import\_availability\_yearly:](#8_reduced_import_availability_yearly) additional example of defining time-/space-dependent parameters, resulting in the need for a storage technology
9. [9\_time\_series\_aggregation:](#9_time_series_aggregation) using time series aggregation (TSA) to reduce computational complexity of optimizing larger multi-time step problems
10. [10\_full\_year:](#10_full_year) optimization of an hourly resolved full year
11. [11\_yearly\_variation:](#11_yearly_variation) specifying yearly parameter changes more easily
12. [12\_myopic\_foresight:](#12_myopic_foresight) optimize using myopic foresight instead of perfect foresight
13. [13\_brown\_field:](#13_brown_field) include existing capacities from before the start of the optimization
14. [14\_multi\_scenario:](#14_multi_scenario) optimize several scenarios with differences in parameter values more efficiently
15. [15\_multiple\_output\_carriers\_conversion\_techs:](#15_multiple_output_carriers_conversion_techs) define conversion technologies with multiple output carriers
16. [16\_multiple\_input\_carriers\_conversion\_techs:](#16_multiple_input_carriers_conversion_techs) define conversion technologies with multiple input carriers
17. [17\_yearly\_parameter\_missing\_values:](#17_yearly_parameter_missing_values) define yearly parameter and use interpolation feature to get values at time indices without specified value
18. [18\_interpolation\_off:](#18_interpolation_off) switch off the interpolation feature in order to use the default values at time indices without specified value
19. 

## 1\_base\_case
In this example, we will investigate a simple energy system to supply the heat demand in Switzerland (CH) and Germany (DE). Only natural gas (NG) boilers are available, which consume natural gas, imported at each node (country). Furthermore, the optimizer can build NG storages and pipelines (Figure 1).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/1fde883f-aa58-4039-81e0-31228913f3b6)

_Figure 1: Energy system "base\_case" generating heat by burning natural gas in boilers._

### Input data structure
To define an energy system in ZEN garden, the available technologies and the corresponding energy carriers must be specified. Since all technologies can be allocated to either conversion, transport or storage technologies, these three technology class folders must be provided. Additionally, a directory to define the carriers corresponding to the technologies and a directory for system-wide specifications must be part of each dataset (Figure 2).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/657f37f8-c705-41c6-a97f-f2945784f695)

_Figure 2: Dataset folder structure._

As a next step, the technologies of the introduced energy system (Figure 1) must be defined by creating the three folders named after the technologies in the corresponding technology classes (Figure 3 left). The individual technology folders must contain the so-called _attributes.json_ file which specifies the technology's **default parameter values** (can be copied from other technologies of the same type) (Figure 3 right). A default value is always defined as _**parameter name, default value, unit**_. If the parameter is unitless, the unit must be the number "1" (to ensure that no units are missed unintenionally) (Figure 3 right).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/0e6a2245-2d13-46a7-b625-ec3e81dd59ca)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/0de3759e-2b37-40f8-80d2-6ba9919d1c8c)

_Figure 3: (f.l.t.r.) Folder structure for technology definition; attributes file needed to define parameters of natural gas boiler; attributes file needed to define parameters of natural gas storage._

As the three different technology classes convert, store or transport energy carriers, all the carriers utilised by the technologies must be defined by creating a folder named after the carrier (Figure 4 left). The carrier names must be consistent with the names used in the technologies' attribute files (Figure 4 middle). The carrier folders must contain the _attributes.json_ file which specifies the default values of the carrier parameters (Figure 4 right).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/2e09fa10-78af-4959-92c3-66e96681ab8d)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/07042c62-94df-4236-a0f2-9acd3e4efd8f)
<img src="https://github.com/ZEN-universe/ZEN-garden/assets/114185605/1253c573-96b9-4417-8f53-9bd3d67f760a" width="300">

_Figure 4: (f.l.t.r.) Carrier definition ; carrier names attributes file; attributes file heat._

The directory "_energy\_system_" defines system-wide parameters. Its _attributes.json_ file specifies default values for the system parameters. The _base\_unit.csv_ file is needed to convert the units used to quantify the default values in the attribute files to a set of matching base units (the base unit file can be copied from other datasets as well). The _nodes.csv_ file must include all the nodes selected in the _system.py_ along with their coordinates which are used to compute the distance between the nodes (length of edges). The distances are needed to compute distance-dependent variables such as the costs for building a natural gas pipeline.

To create your own dataset, all the directories and files the "1\_base\_case" dataset consists of must be included (of course, the technologies can be substituted by other ones).

### Technology and Node Selection
The selection of nodes and technologies that are allowed to be built during the optimization is done in the _system.py_ file, each dataset must have in its top-level directory (Figure 5 left). Therefore, the technologies natural gas boiler, natural gas storage and natural gas pipeline must be selected in the _system.py_ file in order to represent the energy system in Figure 1 (Figure 5 right). Since the energy system consists of the countries Switzerland and Germany, they must be selected as nodes.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/e1c55ee0-3a71-4a8c-bd3e-1fa51f663d1d)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/979f9b08-1cb0-45a8-b7f0-ecf2525db9fa)

_Figure 5: (f.l.t.r.)Location in folder structure of system.py file; system.py file where nodes and technologies must be selected._

Whereas it is allowed to define more technologies than selected in the _system.py_ file, it is not allowed to select technologies which are not defined in the corresponding technology class folder.

### Results
In this example, the optimizer only installs NG boilers to supply the heat demand (Figure 6). No storage or transport capacities are built since the heat demand in the two countries can be met by directly importing and converting natural gas.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/a6c90821-49e0-4f1a-ae35-ffdc24693c90)

_Figure 6: Installed technology capacities of natural gas boilers at the nodes Germany and Switzerland._

## 2\_add\_photovoltaics
In the second example, we extend the system by an electricity demand that can be supplied by photovoltaics (PV) (Figure 7).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/c4366bc1-94db-41f3-8720-507a1dd675a4)

_Figure 7: Energy system with newly introduced photovoltaics technology to satisfy electricity demand._

Therefore, the corresponding photovoltaics folder containing its attributes file must be created (Figure 8 left) to define the new technology. As the added technology generates electricity, it has to be defined as a carrier by creating a directory named electricity containing its _attribute.json_ file (Figure 8 middle). To introduce an electricity demand, the _default_value_ of the _demand_ parameter is set to 50 GW in electricity's _attributes.json_ file (Figure 8 right).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/c40db2b2-9b88-426a-a58b-e05193a89797)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/80998888-932a-4f9b-9fc9-6286562d84f1)

_Figure 8 : (f.l.t.r.) Creation of photovoltaics folder with attributes file to define PV; introduction of electricity carrier; introduction of electricity demand._

After specifying photovoltaics as a technology and defining the carrier electricity, PV can be selected to be built by the optimizer by adding PV as a conversion technology in the _system.py_ file (Figure 9).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/d2e22bad-0652-473f-8ad3-7b5a775068ce)

_Figure 9: Selection of photovoltaics in system.py file._

### Results

As a non-zero default electricity demand has been introduced in the _attributes.json_ file of electricity, PV is built at each node to meet the demand (Figure 10).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/6bf1d168-36c5-4e02-91ad-196309f20766)


_Figure 10: Installed technology capacities of PV and NG boilers at the nodes Germany and Switzerland._

# 3\_multi\_year\_optimization
To modify time-related parameters of the optimization the values of the system's time variables can be adjusted (Figure 11). The most basic parameters are thereby the _reference\_year_, the starting year of the optimization, the _optimized\_years_, the number of years being optimized, and the _interval\_between\_years_ which allows optimizing every n-th year (e.g., every second year for an interval of 2).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/a0811c90-dd2e-4ece-b942-7d0e5232819d)


_Figure 11: Modification of time-related parameters in the system.py file._

The way the parameters are defined in the screenshot's _system.py_ file the years 2023, 2025 and 2027 will be optimized.

### Results
The capacity plot now shows the technology capacities for the three years (Figure 12). Since our input data is constant in time, we cannot see any changes over the years (and the adjustment of the reference year and the interval between the years does not show an effect).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/0e8987f8-4cda-4794-92a0-3dc72d7955d6)


_Figure 12: Installed technology capacities of PV and NG boilers at the nodes Germany and Switzerland._

## 4\_variable\_demand
In this example a first variable parameter is introduced by specifying a larger heat demand for Germany and a smaller one for Switzerland compared to the previous datasets (Figure 13).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/0e6782a0-4c93-4443-b375-a6b8659bb78b)

_Figure 13: Energy system with different heat demands at the nodes Germany and Switzerland._

To introduce parameters which are variable in time or space, additional files are used. To do so, a _.csv_ file named after the parameter must be created, e.g., _demand.csv_ (Figure 14). The file is then used to define values for the parameter's index sets, e.g., introduce different heat demands for the nodes Switzerland and Germany. The exact procedure of creating such files is described in the "How to ZEN garden".

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/b19ff358-bb5b-4333-a3e7-16cd51969cb3)
 ![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/94f29dca-688f-4e59-98b3-87415725fe0d)

_Figure 14: Definition of an inconstant heat demand by using the additional demand.csv file._

### Results
The introduction of the individual heat demands of Germany and Switzerland effects the amount of built capacity at the corresponding nodes (Figure 15). As Germany's heat demand is now triple that of Switzerland, the capacity of German NG boilers is also triple that of Switzerland.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/e4d909cb-7436-4d0f-9a52-260167286411)

_Figure 15: Installed technology capacities of PV and NG boilers at the nodes Germany and Switzerland._

## 5\_reduced\_import\_availability

In this dataset the import availability of natural gas is constrained at the node Germany which results in the construction of a NG pipeline to compensate the missing natural gas (Figure 16).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/e95192d1-1831-460c-91ab-e39c4b88a5d1)

_Figure 16: Energy system with reduced natural gas import availability at the node Germany which enforces the construction of an NG pipeline to transport the missing NG from Switzerland to Germany_

Therefore, the import availabilities of the carrier natural gas are defined differently from the default value (Figure 17). In contrast to the default value which was set to infinity there now exists an import limitation. Whereas the one of Germany is set to 59.0, Switzerland can still import as much gas as needed since no value is assigned in the input file and thus gets the default value (inf). All the values of the other countries will not be used in the optimization but are allowed to be defined in the file anyway (this is generally the case). In this example, the import availability is constant in time.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/479afe46-d372-451b-a67a-6c94e0594e74)


_Figure 17: Additional file to define variable import availabilities of natural gas._

### Results
As a result of the import limitation of natural gas in Germany, a pipeline must be built such that Germany has as much gas as needed to satisfy its heat demand (Figure 18). Figure 19 shows that the reduced import in Germany is substituted by Swiss imports (flows of carriers from outside the network) which are then transported through the pipeline to Germany.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/e0f43d5e-18a8-46f1-8ec1-c392e6cf822e)

_Figure 18: Installed technology capacities of PV, NG boilers and NG pipelines at the nodes Germany and Switzerland._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/2a6393d5-f371-4c5a-93df-b5ac22c7d9a3)


_Figure 19: Import flow rates of natural gas at the nodes Germany and Switzerland._

## 6\_PWA\_nonlinear\_capex
To account for the nonlinearities between price and capacity of capital expenditures, while keeping computational costs low, the piecewise affine approximation of non-linear functions can be used by defining two additional files in the corresponding folder (Figure 20 left). In this dataset, the nonlinearity of the natural gas boiler's capex is defined in the _nonliner\_conversion\_factor.csv_ file (boiler becomes cheaper for larger capacities) (Figure 20 right). To conduct the PWA, the _breakpoints\_pwa\_capex.csv_ file specifies the number and the values of the breakpoints, i.e., in how many sections of linear approximations the non-linear function is divided (e.g., non-linear function should be approximated by five linear functions as in this dataset) (Figure 20 middle).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/487e8723-0e80-4372-9750-970b0bcd425a)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/86ef2bd9-7a67-4172-a29b-8f90a3a59e89)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/5597d148-f6ee-4b78-9fed-0a13539c1738)

_Figure 20: (f.l.t.r.) Additional input files needed to define the PWA; file including the PWA breakpoints; file specifying the non-linear relation between the capex and the capacity._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/2d59ac15-4eb6-4a08-b8f4-e960b5d8e6a9)

_Figure 21: Non-linear capex of the natural gas boiler._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/8ebb2cff-b213-4dc6-b6e0-1ed91d6d4816)

_Figure 22: PWA approximation of the NG boiler’s non-linear capex by five linear functions._

## Results

Since boiler capacity becomes cheaper for larger capacities, the fraction of capactty and capex does not increase linearly. Therefore, Switzerland's boiler has a proportionally larger capex per installed unit c than Germany does since the Swiss heat demand is lower than the one of Germany (Figure 23).

![](RackMultipart20240122-1-34rhms_html_c8640d60fb062fda.png) ![](RackMultipart20240122-1-34rhms_html_7c8b15a75584229e.png)

_Figure 23: Installed natural gas boiler capacities at the nodes Germany and Switzerland; capex of the installed natural gas boilers at the nodes Germany and Switzerland._

# 7\_multiple\_time\_steps\_per\_year

So far, the example datasets did only account for a single time step per year. In this dataset, multiple time steps are optimized per year, of which the first one is represented in Figure 24.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/25f2edbf-9831-431d-a504-7f6ee7bb5d8e)

_Figure 24: Energy system at the specific time step 0._

To optimize multiple time steps per year, e.g., 96 time steps to resolve the first four days of the year on an hourly basis, the system file must be modified as follows (Figure 25).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/362998d7-268d-4b47-8d90-05e6c3bc7143)

_Figure 25: The number of time steps per year can be modified in the system.py file._

To make the dataset and the optimization's results more interesting, the electricity and heat demands are now defined on an hourly basis (operational time step) for the whole year, of which the first 96 will be used (Figure 26).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/721d1d3a-5c9e-4aa1-bf8e-8365de5469fe)

_Figure 26: Hourly electricity demand defined at different nodes and at each hourly time step of the year._

## Results

The built technologies differ in comparison with the previous datasets due to the changes in heat and electricity demand (Figure 27).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/60dcba6e-9c44-4e58-827e-042e4b7b59fe)

_Figure 27: Installed technology capacities of PV, NG boilers and NG pipelines at the nodes Germany and Switzerland._

The output flow rates of the conversion technologies now change hourly since the heat and electricity demand are defined on an hourly basis (Figure 28 and Figure 29).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/9f131700-8546-4bde-9dbd-f0a2ea40b420)

_Figure 28: Output flow rates of the conversion technologies PV and NG boilers at the nodes Germany and Switzerland._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/f835fc95-b26e-41fe-a8a4-26ba4e8a2d5d)

_Figure 29: Heat and electricity demands at the nodes Germany and Switzerland._

# 8\_reduced\_import\_availability\_yearly

So far storage technologies are never built during the optimization. To enforce a situation in which natural gas storages become financially attractive, the yearly import availability of natural gas is constrained in this dataset. Additionally, heat pumps are introduced to provide an alternative way of generating heat without using natural gas (Figure 30).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/6edb8417-8a79-460f-974e-0399a78ff04d)

_Figure 30: Energy system with constrained yearly import availability of natural gas enforcing the construction of a NG storage and newly introduced heat pumps at time step 35._

The yearly import availability of natural gas is constrained for Switzerland in the years 2024 and 2025 by creating an additional input file for the _availability\_import\_yearly.csv_ parameter of the carrier natural gas (Figure 31 left). Since heat pumps use electricity to transfer heat from a lower temperature to a higher temperature, they must be defined in the conversion technology folder (Figure 31 right).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/0325ebae-dfde-4684-87c4-85760e0b145b)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/c5f970d4-db04-4323-b0a7-92b9d57e3c61)

_Figure 31: csv file containing yearly import availability of natural gas in Switzerland; definition of the heat pumps technology._

## Results

Due to the yearly limits in importing natural gas in Switzerland, Germany can no longer compensate its needed natural gas by pipeline transport from Switzerland and thus builds NG storage capacities (Figure 33). In addition, some heat pumps are installed (Figure 33).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/4a4f6bc2-96d5-419a-8bb3-9b804c07ab25)

_Figure 32: Installed technology capacities of PV, NG boilers and heat pumps at the nodes Germany and Switzerland_

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/96b21014-ebaa-4724-b558-2c3ada9fbfa9)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/d3969458-fcdf-4548-99e7-1c3236174b91)

_Figure 33: Installed technology power capacities of NG storages; installed technology energy capacities of NG storages._

Figure 34 shows the energy balance of natural gas in Germany at all 96 time steps.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/23ff23f1-7fb7-4ace-a226-4e1101d2cad8)

_Figure 34: Energy balance of natural gas at the node Germany showing the different parts of import, consumption, storage, transport and demand._

In addition, Figure 35 shows the energy balance of heat in Germany.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/3860dfcc-8b40-4dfa-9d74-b2142437bd09)

_Figure 35: Energy balance of heat at the node Germany showing the different parts of production and demand._

# 9\_time\_series\_aggregation

In this example the usage of the time series aggregation (TSA) is shown by aggregating the 96 time steps of the previous dataset example to 5 time steps (Figure 36).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/39b5817e-4af6-4d22-94ce-e63ab60d0f25)

_Figure 36: Energy system optimized using timely aggregated parameter data._

Since the complexity of solving an optimization problem increases significantly with the number of time steps, computational efforts can explode quite fast. Therefore, time series aggregation is used to overcome long runtimes while keeping the precision of the optimization problem's results accurate. To specify how many distinctive time steps of the underlying input data should be used to define the optimization problem, the following adjustments have to be made. Here, the same 96 values as in the previous dataset are considered, but the time series aggregation reduces the problem to five distinctive time steps before the optimization problem is solved (Figure 37).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/3af5530d-faf4-42f9-9760-5d5002a91d8a)

_Figure 37: Activation of time series aggregation to 5 time steps in system.py file._

## Results

By comparing the energy balance of natural gas for the unaggregated case (Figure 38) and the aggregated case (Figure 39) a relatively large difference caused by the time series aggregation can be seen.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/c3d598a9-77a3-483e-adf8-9a7a7328d74b)

_Figure 38: Unaggregated energy balance of natural gas at the node Germany._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/c1ca3d0b-bda5-4171-9dc3-d996808bbfde)

_Figure 39: Aggregated energy balance of natural gas at the node Germany (5 time steps)._

To get a result which is more similar to the unaggregated case, the dataset is optimized again with ten aggregated time steps per year (Figure 40).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/22de0978-5ce1-4854-aa36-e2473667460b)

_Figure 40: Aggregated energy balance of natural gas at the node Germany (10 time steps)._

While the run time to solve dataset 9 does not differ significantly in comparison to the run time of dataset 10, the effect can be experienced by increasing the number of unaggregated time steps per year to, e.g., 8760 (full year) and solving the problem with and without time series aggregation.

# 10\_full\_year

In this example a full year is optimized in hourly time resolution (Figure 41).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/9d6c3c54-4b85-4965-b335-f2ce998f304d)

_Figure 41: Energy system optimized for a full year at time step 53._

To solve the optimization problem with a complete hourly year resolution, i.e., 8760 time steps per year, time series aggregation is used again to keep computational costs low (to experience the difference in runtime, the time series aggregation can be switched off) (Figure 42).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/1a2ddff2-e2dd-4412-a0a9-dc59f80d9ad2)

_Figure 42: System.py file adaptions for full year optimization with TSA._

## Results

The energy balances now show the yearly changes in heat (Figure 43) and electricity (Figure 44) demand (here for the first 8760 hours, i.e., the first year).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/64183c58-acca-4c16-9bc0-9d31a6dac70d)

_Figure 43: Energy balance of the carrier heat at the node Germany in year 0 (2023)._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/4b4a6d6f-d227-45dc-879c-34119aba6e3d)

_Figure 44: Energy balance of the carrier electricity at the node Germany in year 0 (2023)._

# 11\_yearly\_variation

This dataset demonstrates how yearly changing parameters can be defined more easily.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/940251aa-bdee-4e68-8b8f-1d362e18269f)

_Figure 45: Resulting flow rates of energy system with yearly changing parameters._

To facilitate the consideration of changes in parameters with hourly time steps over several years (e.g., carrier prices), their yearly percentual change can be defined in additional input files. Therefore, the hourly input data must be provided for a single year only and not for every upcoming year. In this dataset, the yearly change in the import price of natural gas is defined by the file _price\_import\_yearly\_variation.csv_ (add _\_yearly\_variation_ ending to parameter name) (Figure 46). As there are no time- or node-dependent import prices for natural gas defined in this dataset, the default value from the attributes file is scaled according to the factors specified in the yearly variation file. To get the import price of a specific year, the default value will be multiplied by the factor of the corresponding year (factors are not the percentual increase between the years).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/ae03c906-d02f-4230-8cd8-fd3a47c2fb9b)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/7737ac84-80d3-4f42-bbaf-b2b1cee664a6)

_Figure 46: Definition of yearly changing parameters by creating an additional input file._

Additionally, the yearly change in electricity demand is regarded as well. In contrast to the natural gas prices, the electricity demand is specified for time and nodes (again the input data for the countries not regarded in the optimization is allowed to be stated although it is not used) (Figure 47). Since there is no yearly change defined for Switzerland, its electricity demand will stay the same.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/9bfff84c-c858-4be7-a669-0518cf039406)

_Figure 47: Yearly variation in electricity demand at different nodes._

## Results

To demonstrate the effect of the yearly variation in natural gas import prices, ten years are optimized so that the change in the carrier cost can be seen more clearly (Figure 48).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/11d79167-9c63-4dba-81fd-b5994524b9bd)

_Figure 48: The yearly carrier costs of natural gas increase over the years._

Figure 49 shows the yearly increase in electricity production by photovoltaics due to increasing electricity demand (left) since the electricity input flows of the heat pumps (only other electricity consumer in the energy system) remain constant over the years (right).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/fda338aa-2d5f-4b0d-83be-6834601febf1)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/a196a7aa-2781-4a25-987a-cf58d819200d)

_Figure 49: Total yearly output flows of photovoltaics increase over years (left) whereas total yearly input flows of electricity consuming conversion technologies (heat pumps) remain constant._

# 12\_myopic\_foresight

All the previous datasets are optimized using so-called perfect foresight, i.e., all years are optimized at once with the assumption that all the future parameter data are known at the time the optimization is conducted. In this example, however, myopic foresight is demonstrated, where the knowledge of future parameter data, the foresight horizon, is limited.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/bb33351c-637f-4b93-a129-bd45893e399a)

_Figure 50: Energy system optimized with myopic foresight at time step 53._

So far, the defined energy system has been optimized for all its years simultaneously (perfect foresight = assumption that all the future parameter data are known already in the starting year). In contrast, the so-called rolling horizon (myopic foresight, see [myopic foresight discussion](https://github.com/RRE-ETH/ZEN-garden/discussions/143)) optimization approach can be activated as it is done in Figure 51.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/7812dae4-7946-4aef-bea1-aa3c59922245)

_Figure 51: Activation of myopic foresight in the system.py file._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/98079549-5e1f-4dd9-9319-d3b8eb1c236b)

_Figure 52: Whereas the decision and the foresight horizon are equal for perfect foresight, they differ for myopic foresight._

By activating myopic foresight, the ten years are all optimized separately and since the number of years in the horizon is set to one, the optimizer only regards the parameter data of the underlying year (single-step foresight) (Figure 52).

## Results

The yearly newly added transport (Figure 53), storage (Figure 54) and conversion (Figure 55) technology capacities do differ quite heavily between the myopic foresight approach and the perfect foresight method. Whereas the myopic case installs significantly more transport capacity, the perfect case installs more storage power capacity to compensate. Additionally, the myopic foresight optimization doesn't install any heat pump capacities in year 0, but the perfect foresight optimization does.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/253d3d87-1598-4c1b-bbb4-d709ccee9df5)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/baf92b9d-e858-499a-aeee-bb151c2954ac)

_Figure 53:Yearly installed transport capacity for myopic foresight (left) and for perfect foresight (right)._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/7f7f270d-6ad4-43ce-84e9-626055089d23)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/aa0c4230-c579-42fd-acd0-c0222ceaef3a)

_Figure 54: Yearly installed storage power capacity for myopic foresight (left) and for perfect foresight (right)._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/6c3f07b0-c014-4253-a49e-604c9f96c333)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/75143fb1-4f73-4a6e-89cf-b14a92001c00)

_Figure 55: Yearly installed conversion capacity for myopic foresight (left) and for perfect foresight (right)._

# 13\_brown\_field

This dataset shows how technology capacities which already exist before the optimization horizon starts (brown field) can be specified.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/e8499948-1bd8-4a41-a006-3d9ea83eba34)

_Figure 56: Energy system modelled with the brown field approach (existing capacities) at time step 53._

By defining customized values for the _capacity\_existing_ parameter of a technology, the capacities of e.g., existing photovoltaics can be considered (Figure 57).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/7431ec04-593a-40eb-be36-a9b866fe2ec8)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/a47285ca-4861-4218-b914-8fcda85e5ba7)

_Figure 57: Definition of existing photovoltaic capacities by creating an additional file in the photovoltaics folder._

## Results

By comparing the yearly installed conversion technology additions using existing photovoltaic capacities (brown field) with a dataset without existing photovoltaic capacities (green field) (Figure 58), it can be seen that the additions are smaller for brownfield although the overall total capacity is the same for both cases (Figure 59).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/849aa805-83e6-4538-b308-25b83f5a7d53)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/9db8177d-2d06-453b-ac4d-13677c28a56f)

_Figure 58: Yearly conversion technology capacity installation for brown field (left) and for green field (right)._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/10e005e8-d94c-49e5-85e8-dba8aaae5d47)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/24b028bf-99f0-486b-bd6d-5649f11c8c2c)

_Figure 59: Total conversion technology capacity for brown field (left) and for green field (right)._

# 14\_multi\_scenario

This dataset introduces the functionality of scenarios, a method to modify individual parameter values during a single framework execution.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/46256f4e-4693-4af8-b7d6-b78df5addd23)

_Figure 60: Multi-scenario energy system at time step 53 and the basic scenario, scenario 0._

To optimize several datasets with minor differences at once, the multi-scenario functionality can facilitate the workflow. In this example, the same dataset as in the previous example is optimized three times for different carbon emission prices. Instead of defining and optimizing three individual datasets, the scenario approach is used ([scenario discussion](https://github.com/RRE-ETH/ZEN-garden/discussions/294)). To activate the scenario method the "conduct\_scenario\_analysis" value must be set to true in the system file (Figure 61).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/23b9b469-4ca4-4c51-8418-55d83c72f268)

_Figure 61: Activation of the scenario analysis in the system.py file._

Additionally, the _scenarios.py_ file must be provided in the most outer folder of the dataset (Figure 62). Inside this file, the different scenarios must be specified by stating which parameter should be changed with respect to the base configuration.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/941d410f-743b-4e8d-b8cf-43afc3dc4f8a)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/f61c8964-808c-428e-baaf-d3411da8424c)

_Figure 62: Definition of the scenarios by creating the scenarios.py file_

In this example, three different carbon emission price scenarios are defined (Figure 63). The first file acts as the base case and the files with the "1" and "2" appendices specify the carbon prices for scenarios one and two, respectively. To minimize the run time needed to solve the optimization problem only three years are optimized again.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/b6d3dea8-bc21-42f3-bdfc-35716d18682b)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/ead5ea37-ceff-4242-8487-fbe95069f08a)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/bd94f618-5511-4727-9317-2f0655e2791b)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/52af2135-bf6b-4d2b-8ebf-b5f18cb8875a)

_Figure 63: The three scenarios differ in their carbon emission prices._

## Results

The first row of plots shows the carbon emissions caused by the usage of natural gas (Figure 64). It can be clearly seen that the amount of produced carbon decreases for higher carbon emission prices (the y-axis of scenario 2 is scaled differently). In the second row of plots, the built conversion technology capacities can be seen (Figure 65). Again, the increase in carbon prices introduces a shift from natural gas boilers to heat pumps since they become more cost-effective at higher carbon emission prices.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/59dbfcee-0544-403e-8489-bca93f57897c)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/ec5544be-e48c-429f-ae26-d225fba75d77)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/14a79026-123b-420f-a86b-9d0087f67f25)

_Figure 64: Emitted carbon of energy system for the three different scenarios._

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/e38d6534-939b-4d9c-b158-70aeb88020c4)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/a3cb332f-5c4f-439d-9b1c-29d60bcf573e)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/8f9b4a9e-3de3-4142-b6f9-55487208fd31)

_Figure 65: Installed conversion technology capacities for the three different scenarios._

# 15\_multiple\_output\_carriers\_conversion\_tech

This dataset shows how conversion technology producing multiple output carriers can be defined, e.g., combined heat and power (CHP) plants (Figure 66).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/36558c7c-1859-404e-aa20-b1f3dea26f72)

_Figure 66: Energy system including CHP plants._

ZEN garden provides the functionality to model conversion technologies with multiple output/input carriers. In this example, a CHP plant (combined heat and power) is introduced, which converts natural gas to electricity and heat. For sake of simplicity, the conversion technology natural gas boiler is no longer included. To define the two output carriers of the CHP technology they need to be defined as output carriers in the attributes file (Figure 67). Assuming a CHP plant converts about 35% of the burnt fuel to electricity and 44% to heat, the conversion factors needed for the file can be computed. Since they are defined with respect to the reference carrier (electricity), the conversion factor of natural gas to electricity is 2.875 (how much gas is needed to produce one unit of electricity (1/0.35)) and the conversion factor of heat to electricity is 1.257 (how much heat is produced per unit of electricity (0.44/0.35)).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/a9467b37-1b52-410a-bdaf-1562a68ea79a)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/ce7361bf-4e60-4a0f-8956-c067a7732aa9)

_Figure 67: Definition of the multiple output carrier conversion technology CHP plant._

## Results

The plot of the conversion technologies' output flows shows that the CHP plants produce electricity and heat (Figure 68).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/cde60577-45c9-4267-b4af-933889ec184c)

_Figure 68: Output flow rates of the conversion technologies showing the different output carriers of the CHP plants heat and electricity._

# 16\_multiple\_input\_carriers\_conversion\_tech

This dataset shows how a conversion technology with multiple input carriers can be specified.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/b5ad1da6-d64e-496b-9aa2-a009be9f7f30)

_Figure 69: Energy system with multiple input carrier CHP plant._

In the same way as multiple output carriers can be defined, a conversion technology with multiple input carriers can be defined. For this purpose, it is assumed that the CHP plant burns a mixture of natural gas and biogas. Again, the biogas consumed by the CHP plants needs to be related to the reference carrier electricity. Assuming a mixture of 50% natural gas and 50% biogas, the conversion factor defined in the file is about 1.427 GWh of biogas per GWh of produced electricity (Figure 70).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/9d4403f2-cece-449d-93b0-1923d6d32494)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/738427c7-e106-4d10-a25a-888771002f34)

_Figure 70: Definition of second input carrier bio gas._

## Results

The plot of the conversion technologies' input flow rates shows how natural gas and biogas are consumed (Figure 71).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/bea87d62-5871-480d-af84-adf316a6f2c7)

_Figure 71: Input flow rates of the conversion technologies showing the multiple input carriers of the CHP plant burning natural gas and biogas._

# 17\_yearly\_parameter\_missing\_values

For hourly time-dependent parameters, data values at missing time indices are set to the default value. In contrast, for year-dependent parameters, data values at missing year indices are interpolated by using the data specified in the corresponding input data file (Figure 72). Therefore, defining customized values for, e.g., the carbon emissions limit for the years 2023 and 2025 automatically specifies non-default values for the years between those years. The right screenshot shows the carbon emissions limit for the years 2023,2024 and 2025 (units did change during optimization from gigatons to megatons) which were interpolated using the data from the left screenshot. This feature can be handy to define yearly parameters which follow a linear evolution.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/b6fa904c-d64d-4d44-8cd3-0cb2786db8ed)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/ccc17774-15e5-4bac-bd57-173479b8ceaf)

_Figure 72: Since the carbon emissions limit is not specified for the year 2024, it is interpolated using the specified data._

## Results

Since the carbon emissions limit is reduced over the three years, the total carbon emissions emitted by the energy system decrease (Figure 73). Note that not the entire carbon emissions limit is used in the first time step (total carbon emissions = 40 megatons).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/567aa95a-ed71-41da-af62-bf4c7f72e304)

_Figure 73: Decrease in total carbon emissions emitted by energy system due to carbon emissions limit._

Similarly, the effect of the introduced carbon emissions limit can be seen in the total yearly output flows of the conversion technologies (Figure 74). The heat and electricity produced by CHP plants must be shifted to carbon-neutral technologies to fulfil the carbon emissions limit.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/bd248712-ac07-478e-99ae-7830ecaa731d)

_Figure 74: Total yearly output flows of conversion technologies shift from carbon-emitting to carbon-neutral technologies over years._

# 18\_interpolation\_offhttps://github.com/ZEN-universe/ZEN-garden/blob/development_ZENx_LK/documentation/dataset_creation_tutorial.md#8_reduced_import_availability_yearl

To switch off the interpolation feature described in the previous example (needed if the years without specified input data should use the default value from the attributes file), the _parameters\_interpolation\_off.csv_ file must be used. By stating the parameter names whose values should not be interpolated at missing years the feature can be turned off (Figure 75).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/eef0f5c2-de41-4ade-a64c-36c647aca2ec)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/25b6601e-af44-4a4e-857d-992b9d6d4f7e)

_Figure 75: Definition of yearly parameters which should not be interpolated._

Consequently, the carbon emissions limit at year 1 (2024) is set to the default parameter value which is defined as infinity (Figure 76).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/b59b9b4b-0526-46ea-b7d4-4f06c43a9942)

_Figure 76: Non-interpolated carbon emissions limit values with default value for year 2024._

## Results

By comparing the total carbon emissions of this dataset (where the interpolation feature is switched off and thus the limit for year 2024 is set to infinity) with the carbon emissions of the previous dataset (interpolated case), a difference in carbon emissions in year 1 (2024) can be seen (Figure 77).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/15d6551c-0035-4bdc-92d2-aa3b72abe40a)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/eb557d1d-7cac-4816-9edc-2d6190486648)

_Figure 77: Total carbon emissions without interpolation (left) in year 1 differ from those with interpolation (right)_

# 19\_retrofitting\_technologies
Retrofitting technologies are a child class of the conversion technologies since they can be used to retrofit conversion technologies to account for additional input/output carriers not yet considered. The main purpose is to model carbon caputre units which are retrofitted onto existing carbon-emitting conversion technologies. In this example dataset, a carbon capure (CC) device is retrofitted to absorb the CO2 emitted by the CHP plant. The capured CO2 is then stored in an underground carbon storage (CS) location.
![19_retrofitting_technologies](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/fc50b6c9-bea6-46c4-b0db-023e8a5f5837)

The retrofitting technology carbon capture must be defined inside the extra directory _set_retrofitting_technologies_ located in _set_conversion_technologies_. To allow the optimizer to build carbon capture units, the technology must be selected in the _system_ file.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/2ff64af5-2265-446b-b359-5b2f629b5f65)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/4ff61d21-cc83-451b-8fde-81da23ae3e9a)

Retrofitting technologies have two additional attributes compared to their parent class conversion technologies. For each retrofit technology a retrofit_base_technology has to be specified. Furthermore, the retrofit_flow_coupling_factor links the reference carrier flows of the retrofit technologies to the reference carrier flow of its corresponding base technology (reference carrier flow retrofitting technology over reference carrier flow base technology). In this case, the CHP plant (reference carrier electricity) is retrofitted with a carbon capture unit (reference carrier carbon) and therefore the retrofit_base_technology is the CHP plant and the retrofit_flow_coupling_factor defines how many units of carbon are captured per unit of electricity produced.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/25e3ce3a-48d1-4b23-8357-769688f346b2)


# 20\_unit\_consistency
To get a better feeling for the [unit consistency feature](https://github.com/ZEN-universe/ZEN-garden/blob/main/documentation/how_to_ZEN_garden.md#unit-consistency) of ZEN garden, this dataset contains some inconsistent units which must be corrected by you. In comparison to the previous datasets, the _energy_quantity_ of the carrier _natural_gas_ is now given in kilotons instead of MWh to increase complexity. Subsequently, the _energy_quantity_ units of technologies with _reference_carrier_ natural gas must now be specified in kilotons as well. The goal is to correct the inconsistent units with the help of the error messages thrown while running the dataset. In case you don't manage to find the correct units, the solution is: CHP_plant/conversion_factor/natural_gas: kilotons/GWh, natural_gas_pipeline/capacity_investment_existing: kilotons/hour, natural_gas_storage: change energy_quantity of all attributes to _kilotons_ instead of _MWh_.
