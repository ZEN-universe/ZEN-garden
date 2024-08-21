# How to ZEN-garden
## Setup
### Needed Installations

- PyCharm (IDE, you can use other IDEs as well, but most users of ZEN-garden use PyCharm) [Install PyCharm](https://www.jetbrains.com/pycharm/download/)
- Anaconda (Needed for ZEN-garden environment creation) [Install Anaconda](https://docs.anaconda.com/anaconda/install/)
- Gurobi (Optimization Software) [Install Gurobi](https://www.gurobi.com/downloads/) (not necessary but recommended)
- (GitHub Desktop) [Install GitHub Desktop](https://desktop.github.com/)

### Steps

1. GitHub registration: If you don't have a GitHub account yet register at: [GitHub](https://github.com/)
2. Join ZEN-garden repository: If you didn't receive a GitHub invitation, ask your supervisor to invite you to the repository (write them your GitHub email address) [ZEN-Garden Repository](https://github.com/RRE-ETH/ZEN-garden)
3. Create your own branch: In the ZEN-garden repository click on "branches" and then "new branch", choose "main" as the branch source and "development\_ZENx\_NS" (NS= name, surname) as its name
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/77b44be5-5b11-4c6d-a431-e1a41cb99a14)


![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/59a8ed6f-e97b-4d6e-8826-dc01787216e5)


4. Cloning the repository: To create a local copy of your branch on your computer, you must clone the remote repository from GitHub. **It is important that you clone the repository to a path which doesn't contain any spaces!** (Don't clone to e.g. ./Users/Name Surname, otherwise you'll have issues while executing the framework). To clone your branch there's a more beginner-friendly way using GitHub Desktop and a more advanced way using Git Bash for example.

GitHub Desktop: [Clone Repository with GitHub Desktop](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/adding-and-cloning-repositories/cloning-a-repository-from-github-to-github-desktop)

To clone the repository by using Git Bash, two methods are available: [HTTPS](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository?tool=webui) or [SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)

5. ZEN-garden environment creation: Open PyCharm to view the _zen\_garden\_env.yml_ file (contained in the ZEN-garden folder), copy ```conda env create -f zen_garden_env.yml``` and run the command in your Anaconda prompt (takes several minutes), if the installation was successful, you can see the environment at _C:\Users\username \anaconda3\envs_ or wherever Anaconda is installed
6. Gurobi license: To use all of Gurobi's functionalities, you need to obtain a free academic license: [Get your Gurobi license](https://www.gurobi.com/features/academic-named-user-license/)
 Following these instructions, you'll get a Gurobi license key which you have to run in your command prompt to activate the license for your computer
7. Create PyCharm Configurations: To execute ZEN-garden's different functionalities configurations are used. To add them, follow the steps at "PyCharm Setup": [Create Configurations](https://github.com/RRE-ETH/ZEN-garden/discussions/183)

## ZEN-garden configurations
### Run ZEN-garden module

The ZEN-garden module can be executed in several ways as well as on ETH's EULER cluster. To check if the setup was successful, you can run one of the standardized test cases. To do so,

- open _ZEN-garden\tests\testcases_ and copy paste all the tests to _ZEN-garden\data_
- copy _config.py_ to the _data_ directory
- choose "test\_1a" as the dataset and execute it using one of the following methods

#### Run ZEN-garden using the "Run Module" configuration

Executing ZEN-garden with the created configuration "Run Module" (created in [setup step 7](#steps)) is the most forward way if you use PyCharm. Simply adjust the path in the analysis attribute "dataset" in the config.py file to one of the desired datasets and click the green run-button (have a look at the [config options](#config-settings) to get an overview of all the config settings).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/86782722-a227-4320-a58a-469b4c1464e3)

#### Run ZEN-garden using a terminal
How to run the ZEN-garden package in a terminal is described in [ZEN-garden as a package](https://github.com/RRE-ETH/ZEN-garden/discussions/183). Depending on the terminal you want to use, the procedure differs slightly. Before entering the module's execution command, ensure that the _data_ folder is your working directory. To change the ${\textsf{\color{brown}working directory}}$ from, e.g., _ZEN-garden_ to _ZEN-garden/data_, simply run ```cd data```.

**PyCharm Power Shell (Terminal in PyCharm):** As the ${\textsf{\color{orange}zen-garden conda environment}}$ is activated by default, you can simply enter the following ${\textsf{\color{blue}command}}$ followed by a chosen ${\textsf{\color{green}dataset name}}$:

${\textsf{\color{orange}(zen-garden)}}$ PS ${\textsf{\color{brown}<path_to_zen_garden>/ZEN-garden/data/>}}$  ${\textsf{\color{blue}python -m zen-garden --dataset=}}$ ${\textsf{\color{green}"test 1b"}}$

To copy: ```(zen-garden) PS <path_to_zen_garden>\ZEN-garden\data> python -m zen_garden --dataset=“test_1b“```

**Anaconda Prompt:** The only difference when using the Anaconda prompt is that you have to activate the zen-garden environment manually before you can run the package execution command. This can be done by running ```conda activate zen-garden```.

If your console looks something like the screenshot below, the ZEN-garden module works fine on your computer, and you can run all the data sets located in the _data_ folder by choosing one of the two methods. Otherwise, revisit the setup steps according to the occurred error.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/fd38a757-11bc-4955-8aae-727479a8ed70)

#### Run ZEN-garden on EULER

To run computational more expensive optimization problems, ETH's EULER cluster can be accessed as described at [ZEN-garden on EULER](https://github.com/RRE-ETH/ZEN-garden/discussions/186).

### Read Results
After a dataset's optimization problem has been executed, its results can be accessed and visualized with help of the _results.py_ script. To get a first impression of the available results processing functionalities, the Jupyter Notebook _postprocess\_results.ipynb_ can help a lot. It can be found in ZEN-garden's _notebooks_ directory. 

The most efficient way to integrate the results script into your result processing script, is to import the `Results` class into your script:
``from zen_garden.postprocess.results.results import Results``
Then, you can simply read a result in your script by writing
``r = Results(path=<path_to_results>)``

Another way to access your results is to use the "Read Results" configuration. By running the _results.py_ script, the different member functions of the contained ```Results``` class can be applied to the ```Results``` object to extract and plot the data of your optimization problem. Since the "Read Results" configuration creates an instance of the ```Results``` class, the object can be accessed by "self". By setting a break point at the end of the file, the debugger console can be used to apply the class's functions to the ```Results``` instance.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/85764bc6-2999-4f83-91c6-1d49243a1d8d)

#### Accessing your results data
To access the data frames containing the raw optimization results of the variables and parameters, the following member functions of the ```Results``` class can be used:

1. ```r.get_total()```,
2. ```r.get_full_ts()```,
3. ```r.get_df()```,

where "r" is an instance of the ```Results``` class. ```r.get_total()``` returns the aggregated values of the variable and parameter values for each year. For hourly resolved variables, such as "flow\_conversion\_input", this is the sum over all hours of the year for each year. Yearly resolved variables, such as "capacity", remain unaltered because they are already yearly aggregates.

```r.get_full_ts()``` returns the hourly evolution of hourly resolved variables. This is especially useful when using the time series aggregation, where the hours of the year are aggregated by representative time steps. ```r.get_full_ts()``` disaggregates the time series back to full hourly representation. Yearly values remain in yearly resolution, thus for these components ```r.get_total()``` and ```r.get_full_ts()``` return the same result.

Under the hood of ```r.get_total()``` and ```r.get_full_ts()```, we use the ```r.get_df()``` function to extract the raw variable and parameter values. If these are of interest, you can use ```r.get_df()```, otherwise ```r.get_total()``` and ```r.get_full_ts()``` will be more useful.

##### self.get_df()
The most fundamental function to access the data of a specific variable such as, e.g., "flow_conversion_input" is ```self.get_df("flow_conversion_input")```. It returns a Pandas series containing all the "flow_conversion_input" values of the different technologies at the individual nodes at every time step.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/071ac7df-1ba0-42fe-9f2c-89b77a63b30a)

##### self.get_full_ts()
A more convenient way to access the same data is offered by ```self.get_full_ts("flow_conversion_input")```, a function which creates a data frame of the variable's full time series.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/71779981-de3b-48cd-855f-4955588a093f)

##### self.get_total()
If you're not interested in the hourly resolution of the variable values, ```self.get_total("flow_conversion_input")``` can be used to obtain the yearly sums of the hourly data.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/37abaf36-02f3-47e0-9d2b-94f23695d782)

#### Compare two datasets
You can compare two ```Results``` objects by using the following class methods. They can help you to get a fast overview of two datasets' differences which facilitates spotting the reasons for errors. Again, the Jupyter Notebook shows some practical examples, but the functionalities can be used in the debug console as well by creating the desired ```Results``` objects the way it is done in the beginning of the notebook. All the functions take a list of two ```Results``` instances as their input argument.

- relate the configs of two datasets:
```Results.compare_configs([result_instance_1, result_instance_2])```
- relate model parameters:
```Results.compare_model_parameters([result_instance_1, result_instance_2])```
- relate model variables:
```Results.compare_model_variables([result_instance_1, result_instance_2])```

### Run Tests
The main purpose of the test files is their usage for the automated testing functionality of ZEN-garden. By comparing the variables' values gathered by simulating the testcases with some reference values, the correctness of the current framework code can be proved. Whenever you adapted some framework code, you can use the run test configuration to ensure that ZEN-garden does still function properly.

#### How to define new test cases
All testcases listed as functions in the run_tests.py script will be executed if a dataset with identical name is provided in the _ZEN-garden/tests/testcases_ directory. Besides the dataset itself, the values which serve as the reference for the variable comparison must be defiend in the _test_variables_readable.csv_ file which is also located in the _testcases_ folder.

## Parameters, variables, and constraints
An important concept in ZEN-garden, or for optimization problems in general, is the definition of parameters, variables, and constraints. Parameters are used to store data that is immutable, meaning once a parameter's values are specified, they stay the same for the whole optimization (e.g., the hourly electricity demand per country). On the other hand, variables represent quantities whose values are computed by solving the optimization problem (e.g., the hourly electricity output flow of a gas turbine). By defining constraints, the parameters and variables can be related to each other such that they follow the rules of physical properties etc. (e.g., energy conservation at nodes). In the example optimization problem below, $c^Tx$ is the so-called objective function whose value is optimized (mostly minimizing the net present cost of the entire system), $x$ and $b$ are vectors containing all the variables and parameters, respectively, which are related by constraints of the form $Ax \leq b$. Additionally, some variables are defined as non-negative numbers, i.e., $x \geq 0$, as physical metrics like costs, power flows and energy etc. can only be positive.

$$
\begin{equation}
\begin{aligned}
\min_{x} \quad & c^Tx\\
\textrm{s.t.} \quad & Ax \leq b\\
  &x\geq0    \\
\end{aligned}
\end{equation}
$$

To get an overview of all the existing parameters, variables and constraints, have a look at these [tables](#parameter,-variable-and-constraint-overview).

To find the definitions of all the parameters, variables and constraints you can look up every appearance of _add\_parameter_/_add\_variable/add\_constraint_ in all of ZEN-garden's files by using CTRL+Shift+F. Assessing the definitions can be quite helpful to get a better understanding as they include the _doc_ strings, a brief explanation of the underlying parameter, variable or constraint. In addition, it can be seen in which file _(technology.py_, _carrier.py_, etc.) the definition is located, revealing some extra information. Since this method takes some time to find the desired doc string, the ```Results``` class contains the function ```r.get_doc("component")``` which returns the doc string of the corresponding component.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/ca16be5b-4a9f-4fba-ae28-4613c3f7c564)

### Unit Consistency
Since parameters and variables are related by constraints, their units must follow a certain consistency pattern. For example, the constraint _availability\_import_ relates the variable _flow\_import_ with the parameter _availability\_import_ as

$$flow\textunderscore import \leq availability\textunderscore import$$

which implies that the units of the two terms must be identical. Additionally, the constraint _nodal\_energy\_balance_ relates the variable _flow\_import_ with the parameter _demand_ as

$$demand = flow\textunderscore import - flow\textunderscore export + flow\textunderscore conversion\textunderscore output + ...$$

which implies that _flow\_import_ and _demand_ must have identical units as well. Therefore, it follows that the two parameters _availability_import_ and _demand_ of a carrier must have the same units in order to be consistent.

More generally, the five unit dimensions _energy_quantity_, _time_, _money_, _distance_ and _emissions_ are used to define individual unit categories, the parameters and variables can be assigned to. For example, the parameter _demand_ belongs to the group [energy\_quantity]/ [time] since a certain amount of energy is consumed per time step whereas the parameter _availability\_export\_yearly_ is assigned to the group [energy\_quantity] as it represents the total available energy amount per year. Therefore, the energy\_quantity term of the two parameters must not be chosen individually for the same carrier. If e.g. the units of _demand_ are specified as GW (i.e. GWh/hour) for e.g. the carrier heat, the units of _availability\_export\_yearly_ must be defined as GWh for heat automatically. Of course, this unit consistency for the energy\_quantity term must only be fulfilled carrier-wise (for a second carrier the energy term could be deinfed as e.g. ton).

Equivalently, the energy\_quanity must be consistent per technology element as well.

If there are inconsistent units in a dataset, an error is thrown indicating which units most probably are wrong.

### How to define the unit dimensions when adding a new parameter/variable to the framework
#### Parameters
The argument ```unit_category``` specifies the unit dimensions of the parameter and must be passed to the ```extrect_input_data``` function, e.g., for _capacity_addition_min_ the ```unit_category``` is defined as ```{"energy_quantity": 1, "time": -1}``` since a technology capacity is per definition given as energy_quantity (e.g. MWh) per time (hour), i.e. e.g. MW
```self.capacity_addition_min = self.data_input.extract_input_data("capacity_addition_min", index_sets=[], unit_category={"energy_quantity": 1, "time": -1})```

#### Variables
Since the units of variables are not defined by the user but are a consequence of the parameter units as explained above, their unit dimensions are specified in the ```add_variable``` functions of the class ```Variable```. Again, the argument ```unit_category``` is used to define the unit dimensionality.
```variables.add_variable(model, name="capacity", index_sets=cls.create_custom_set(["set_technologies", "set_capacity_types", "set_location", "set_time_steps_yearly"], optimization_setup), bounds=capacity_bounds, doc='size of installed technology at location l and time t', unit_category={"energy_quantity": 1, "time": -1})```

## Input data structure
The input data of a dataset must be composed of the _system.py_ file and the three folders _set\_carriers_,  _set\_technologies_ and _energy\_system_.

### system.py
The _system.py_ file must contain the sets of technologies that constitute the energy system, i.e. , that take part in supplying the final energy demands. You can have technologies in your input data folder but not list them in the system.py. In this case, they are excluded from the optimization. Additionally, a subset of nodes (from _system\_specification/set\_nodes.csv_), the starting year of the optimization (_reference\_year_) and a lot of other time related specifications can be defined. The time step parameters are discussed [in this Git Discussion](https://github.com/RRE-ETH/ZEN-garden/discussions/143). To get an overview of how to define the different properties, have a look at the [system settings](#system).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/0ce6f2e6-646a-42d6-bce9-93cd142f00f2)

### set\_carriers
The _set\_carriers_ folder contains the energy carrier types such as electricity, heat, biomass, natural gas, etc. All the carriers that are needed by the technologies specified in the _system.py_ file must be contained in this directory; additional carriers are allowed as well. You do not need to specifically list the carriers in system.py as they are implied by the included technologies.

To define a specific carrier, a folder named after the carrier containing the attributes file must be created.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/1f575676-9a9f-4474-83af-02bc5bed8a28)

The attributes file contains all the default values of the parameters' needed to describe the carrier([structure attributes file](https://github.com/ZEN-universe/ZEN-garden/discussions/351)). As the parameters' values can differ along the energy systems' nodes and the simulated time steps, the variations can be described by creating additional input files having the following name structure (parameter name without the "default" ending):

- **demand.csv:** If there exists a demand for a carrier, it can be described in the demand file.
- **availability\_import.csv:** This file can be used to specify different values of a carrier's import availability as it may differ for the nodes, time steps etc.
- **availability\_export.csv:** As for the import availability the export availability can be customised.
- …

Examples of existing parameters can be assessed in the attribute files of the test datasets (for completion, the whole set of parameters can be found in the [appendices](#parameter-variable-and-constraint-overview). To get a better understanding of how to structure these additional input files, have a look at the [spreadsheet structure section](#spreadsheet-structure).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/42409664-58fc-435f-8a9f-44e682487ce9)

### set_technologies
The _set_technologies_ folder contains the sub folders _set_conversion_technologies_, _set_transport_technologies_ and _set_storage_technologies_.

#### set\_conversion\_technologies
The _set\_conversion\_technologies_ folder contains the energy conversion technologies such as boilers, power plants (e.g., lignite coal plants), or renewables. All the conversion technologies that are specified in the system file's technology sets must be contained in this directory; additional conversion technologies are allowed. The procedure of defining a specific conversion technology is the very same as for energy carriers, described in the previous section. Again, a folder with the conversion technology's name must be created, including the attributes file for conversion technologies and variations in space and time can be specified with additional input data files.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/13061320-754a-4107-88a1-c0cb163718c5)

##### set_retrofitting_technologies
The _set_retrofitting_technologies_ folder contains the retrofitting technologies such as carbon capture technolgies. The retrofitting technologies are specified in _system.py_. The retrofitting technologies are a child class of the conversion technologies and have two additional attributes. For each retrofit technology a _retrofit_base_technology_ has to be specified. Furthermore, _retrofit_flow_coupling_factor_ links the reference carrier flows of the retrofit technologies to the reference carrier flow of its corresponding base technology (reference carrier flow retrofitting technology over reference carrier flow base technology). 
For instance, if a coal fired power plant (reference carrier electricity) is retrofitted with a carbon capture unit (reference carrier carbon), the coal fired power plant would be the _retrofit_base_technology_ of the carbon capture unit, and the _retrofit_flow_coupling_factor_ would define how many units of carbon can be captured per unit of electricity produced.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/9a54ad87-4db8-4c25-86e0-87ec53e959d0)


#### set\_transport\_technologies
The _set\_transport\_technologies_ folder contains the energy transport technologies such as natural gas pipelines or power lines. All the transport technologies that are specified in the system file's technology sets must be contained in this directory; additional transport technologies are allowed. Once more, the individual transport technologies must be defined the same way as carriers and other technologies.

##### opex_specific_fixed vs opex_specific_fixed_per_distance
The specific opex of transport technologies can either be defined as money per capacity or money per capacity per length, by providing the attribute opex_specific_fixed or opex_specific_fixed_per_distance, respectively. If both attributes are defined, the distance-dependent value is used.

#### set\_storage\_technologies
The _set\_storage\_technologies_ folder contains the energy storage technologies such as pumped hydro, natural gas storages, batteries, etc. All the storage technologies that are specified in the system file's technology sets must be contained in this directory; additional storage technologies are allowed. Again, the procedure of defining them is equivalent as before.

### energy\_system
The _energy\_system_ folder contains additional input data that is needed to define the energy system as a whole. Other than the carrier and technology folders, this folder must contain more files than just the attributes file:

- attributes.json: carbon emissions related information etc.
- base\_units.csv: definition of base units to which input data units are converted
- set\_edges.csv: definition of existing connections (edges) between node pairs
- set\_nodes.csv: set of all nodes along with its longitude and latitude, which are used to compute the distances between the nodes (=length of edges). These distances are needed to compute distance-dependent variables such as the cost of building transport technologies.
- unit\_definitions.txt: definition of units not contained in the unit handling package

### Spreadsheet structure
The individual values at different nodes and time steps can be entered into the input files by using the column headers "node" and "time"/"year" as it is done in the pictures.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/7939b5bf-fb50-4662-98c1-5b18f8072953)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/658e49ed-ba41-4164-a04a-db2bdfc85fc5)

The header "time" is used to represent hourly time steps, whereas the "year" header serves for yearly time steps ([time step discussion](https://github.com/RRE-ETH/ZEN-garden/discussions/143)). An overview of the parameters' time step types is given in the appendices. In addition to the one-dimensional input structure above, data varying in space (nodes) and time can be structured by stating the nodes and time steps explicitly:

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/ebd6d909-ed63-4285-9bad-42cc4becbae6)


Thanks to ZEN garden's capability of completing required parameter values which are not stated in the input files explicitly, the user doesn't have to specify the values for all indices of the parameter. For example, the parameter _availability\_import_ is defined by the _index\_sets_ "set\_nodes" and "set\_time\_steps", however it is possible to only provide data for the individual nodes and none for the different time steps (see screen shot above). The input data handling will then complete the "missing" values by using the nodes' individual import availabilities for all the time steps (time independent parameter). Therefore, the framework user can choose for each parameter, if the default value (specified in the attributes file) or individual values for the parameter's index sets should be used.

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/ff361138-cbef-4200-bab7-3c638bd3fbac)

### Additional methods to enter input data
#### PWA
To approximate the nonlinearities of a technology's capex (capex vs. capacity) the so-called piecewise affine (PWA) approximation can be used. In this method, the nonlinear function is divided into several regions where it can be approximated as a linear function. To specify such PWA input data the following two files are needed (in the folder of the corresponding technology):

- _nonlinear\_capex.csv_ 
- _breakpoints\_pwa\_capex.csv_ 

The "nonlinear" file must thereby contain the values for the nonlinear relation between the two metrics (e.g., heat output vs. natural gas input) followed by a pair of units whereas the "breakpoints" file is used to divide the nonlinear function into several intervals (#intervals = #breakpoints - 1).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/d4369f1e-3660-430a-9a74-cbfc141aea62)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/901f72a2-f499-4d0c-9e86-60c6bbdb1c89)

#### Define technology with multiple input/output carriers
To define a conversion technology with multiple input and output carriers, several carrier types can be specified as the technology's input/output carriers in its corresponding attributes file (e.g., heat and carbon as output carrier). Having several input/output carriers requires additional conversion factors, i.e., the factor relating the amount generated/consumed of a carrier with respect to the one's of the reference carrier. Therefore, a second conversion factor must be specified in the conversion technolgy's attributes file(e.g., since the reference carrier in the screenshot is heat and the two other carriers are natural gas and carbon, the conversion factors relating heat with natural gas (1.1 GWh NG/GWh heat) and heat with carbon (0.01 kt carbon/GWh heat) need to be declared).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/9fcb6eae-2303-4e13-81db-71d8c83c39ea)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/dd392616-beeb-4a97-aa33-1c3acfe56663)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/52d92618-4127-4c50-b231-f7e6730cb84e)

#### Define input data with yearly variations
To simplify the hourly dependent input data which varies each year by a specific factor the "yearly\_variation" file can be used. For example, if the heat demand is expected to increase or decrease over the years by a known percentage, the change can be specified as it is done in the following figure instead of defining all the values explicitly in the heat demand file. By doing so, the demand values will be scaled accordingly to the yearly variation factors (e.g., demand CH in year 2022 and time step 0 will be 9).

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/10a6894b-5bd1-4cbb-b711-b71d6a1916a2)
![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/874a55b6-35c1-4491-baf6-30d3b9dbf77a)

## Framework structure
<object data="https://github.com/ZEN-universe/ZEN-garden/blob/development_ZENx_LK/documentation/ZEN-garden_structure.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="http://yoursite.com/the.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/ZEN-universe/ZEN-garden/blob/development_ZENx_LK/documentation/ZEN-garden_structure.pdf">Download PDF</a>.</p>
    </embed>
</object>

ZEN-garden is structured into the three building blocks _preprocess_, _model_ and _postprocess_. _model_ is used to describe the optimization problem of the energy system containing the different technologies and carriers, _preprocess_ extracts the provided input data and _postprocess_ saves and visualizes the simulation results. To get a better understanding of the general order of ZEN-gardens execution steps and the package levels, have a look at the flowchart.

### Preprocess
#### Functions
##### extract\_input\_data.py
The functions to extract the data from the differently structured spreadsheets and store the information in data frames are in this script. As there are a lot of different ways in which the input data itself or the description of specific parameters can be specified (linear/PWA, from attributes/extra file, etc.), the data extraction process is quite complicated, leading to the large number of functions.

##### time\_series\_aggregation.py
To reduce the complexity and thus the computational cost of optimizing energy systems with hourly data resolution, the time series of the underlying data can be aggregated to decrease the number of time steps with individual data values. For example, _test\_4b_ aggregates the 8760 hourly time steps of a year to ten representative time steps, thus reducing the computational effort heavily while approximating the original data still well. The time series aggregation parameters are defined in the system file and further details can be found at: [TSA discussion](https://github.com/RRE-ETH/ZEN-garden/discussions/114)

##### unit\_handling.py
To ensure that all the units of the input data are consistent, a unit-handling is implemented. By specifying a set of base units (_system\_specification/base\_units.csv_), the units used in the input data files do not have to be consistent as the unit handling will transform the units according to the chosen base units ([unit handling discussion](https://github.com/RRE-ETH/ZEN-garden/discussions/113)).

### Model
#### default\_config.py
The default configuration is defined in the _Config_ class which describes all the relevant specifications for ZEN-garden with the four dictionaries:

- **analysis** : describes the desired analysis such as the objective (e.g., total cost), if the problem should be minimized or maximized, the discount rate, etc.
- **solver** : contains the information regarding the solver used to solve the optimization problem such as its name (e.g. glpk (free) or gurobi\_persistent (commercial; free with academic license)) and different corresponding solver specifications ([Gurobi documentation](https://www.gurobi.com/documentation/9.5/refman/parameters.html))
- **system** : describes the energy system;used to select technologies which are included in the optimization problem and to define a subset of nodes which should exist (more technologies and nodes allowed in input data; see above), contains time related parameters to specify time series aggregation ([TSA discussion](https://github.com/RRE-ETH/ZEN-garden/discussions/114))
- **scenarios** : used to define the individual settings of multi-scenario simulations (see _test\_6a_)

To specify changes from the default configuration the files _config.py_, _system.py_ and _scenarios.py_ are used.

- **config.py:** The config.py file is in the _data_ and the _testcases_ folder. The changes made in these files apply to all the datasets contained in the corresponding folder. For example, if you specify a solver name in the _data_ config, it affects all the datasets located in the _data_ directory. As described in the "Run ZEN-garden module" section, the config can be used to select which dataset should be executed.
- **system.py:** Parameters that describe the energy system are changed in the system.py file and only apply to the specific dataset in which the system.py file is located. Each dataset must contain its own system file which should look similar as:

![image](https://github.com/ZEN-universe/ZEN-garden/assets/114185605/22268e1c-24f6-4ffe-8890-bb969990ca9e)

- **scenarios.py:** To specify the individual scenarios of a multi-scenario simulation, this file is used similarly as in _test\_6a_. By defining additional input data files, individual parameter values can be modified with respect to the default dataset, thus allowing a more efficient way than running a completely new dataset ([Scenario Analysis](https://github.com/RRE-ETH/ZEN-garden/discussions/294)).

#### optimization\_setup.py
The _OptimizationSetup_ class defines the optimization model by saving the properties of the _analysis_ and _system_ dictionaries. Using this information, the class adds the specified carriers and technologies to the optimization model such that it can be solved with its built-in solving method afterwards.

#### Objects
##### component.py
The _Component_ class is used to add the parameters, variables and constraints to the optimization model represented by a linopy model. Since the linopy modelling language requires specific ways of how these parameters, variables and constraints must be constructed to suit its properties, _component.py_ is needed. Component.py is therefore needed to adapt the parameters constructed via element.py such that they can be added to the linopy optimization model.

##### element.py
The _Element_ class serves as the incremental building block of all technologies and carriers by defining all the necessary methods to define specific carriers and technologies. Therefore, it is the parent class of the _Carrier_ and the _Technology_ class.

##### energy\_system.py
The ```EnergySystem``` class contains methods to add parameters, variables and constraints, to the optimization problem. In general, these components concern system-wide properties such as carbon or cost metrics. Additionally, the connections between the individual nodes are calculated and the objective function is created and passed to the concrete model.

##### time\_steps.py
Contains a helper class containing methods to deal with timesteps.

##### Carrier
###### carrier.py
This script defines the class _Carrier_, which describes the individual energy carriers such as electricity, heat, biomass, etc. By extracting the corresponding input files, the carrier-related parameters, variables, and constraints are created.

##### Technology
###### technology.py
Contains the ```Technology``` class which defines all the technology-related parameters, variables and constraints that hold for all the existing technologies.

###### conversion\_technology.py
Creates parameters, variables and constraints specifically needed for conversion technologies (e.g., natural gas boiler).

###### storage\_technology.py
Creates parameters, variables and constraints specifically needed for storage technologies (e.g., pumped hydro storage).

###### transport\_technology.py
Creates parameters, variables and constraints specifically needed for transport technologies (e.g., power lines).

### Postprocess
#### postprocess.py
The ```Postprocess``` class saves all the information contained in the optimization problem and all the system configurations such that the gathered solution and the whole setup can be accessed without running the optimization again.

#### results.py
The ```Results``` class can read the files created with the Postprocess class and contains methods to visualize these results.

## Appendices
### Config Settings
#### System
system[Dictionary Key] = Value Type(Value Options)
 e.g.:

system['set\_transport\_technologies'] = ['power\_line','natural\_gas\_pipeline','carbon\_pipeline']

| **Dictionary Key:** | **Value Options/Meaning:** | **Value Type:** |
| --- | --- | --- |
| "set\_conversion\_technologies" | Subset of the names of the folders located in the _set\_conversion\_technologies_ directory of the dataset | List of strings |
| "set\_storage\_technologies" | Subset of the names of the folders located in the _set\_storage\_technologies_ directory of the dataset | List of strings |
| "set\_transport\_technologies" | Subset of the names of the folders located in the _set\_transport\_technologies_ directory of the dataset | List of strings |
| "set\_nodes" | Nodes defined in _set\_nodes.csv_ | List of strings |
| "reference\_year" | Starting year of optimization | Positive integer \>=1900 |
| "unaggregated\_time\_steps\_per\_year" | First x hours of year (normally 8760 for full year) | Positive integer |
| "aggregated\_time\_steps\_per\_year" | Number of representative time steps for time series aggregation | Positive integer |
| "conduct\_time\_series\_aggregation" | True/False | Bool (default: False) |
| "optimized\_years" | Number of optimized years | Positive integer |
| "interval\_between\_years" | Every "i-th" year is optimized (e.g. every second year for interval of 2) | Positive integer |
| "use\_rolling\_horizon" | Switch between perfect- and myopic foresight | Bool (default: False) |
| "years\_in\_rolling\_horizon" | Number of years in foresight horizon | Positive integer |
| "double\_capex\_transport" | Account transport technology capex twice: per installed capacity (€/W) and per installed capacity per length (€/(W\*m)) | Bool (default: False) |
| "exclude\_parameter\_from\_TSA" | Choose if parameters defined in the optional file _exclude\_parameter\_from\_TSA.csv_ should be excluded from TSA | Bool (default: True) |
| "conduct\_scenario\_analysis" | Choose if scenario analysis should be conducted | bool (default: False) |
| "run\_default\_scenario" | Choose if default scenario should be executed or not | Bool (default: True) |
| "clean\_sub\_scenarios" | Choose if result files of scenarios that are not in the current scenario dict should be deleted | Bool (default: False) |
| "knowledge\_depreciation\_rate" | Rate at which knowledge stock of existing capacities is depreciated annually | Positive float ϵ [0,1] |

#### Analysis

| **Dictionary Key:** | **Value Options/Meaning:** | **Value Type:** |
| --- | --- | --- |
| "objective" | "total\_cost" or "total\_carbon\_emissions" | String |
| "sense" | "minimize" or "maximize" | String |
| "time\_series\_aggregation" | [See tsam package](https://tsam.readthedocs.io/en/latest/) | Dictionary |
| "folder\_output" | Specify path of output folder to store optimization results (e.g. "./outputs/") | String |
| "output\_format" | "h5", "json" or "gzip" | String |
| "max\_output\_size\_mb" | Limit the maximum file size of json result files | Positive Integer (default: 500) |
| "use\_capacities\_existing" | Choose if existing capacity should be considered (brownfield vs. greenfield) | Bool (default: False) |

#### Solver

| **Dictionary Key:** | **Value Options/Meaning:** | **Value Type:** |
| --- | --- | --- |
| "name" | Choose solver: "glpk" or "gurobi" | String (default: "glpk") |
| "solver\_options" | Additional solver options (see _default\_config.py_) | Dictionary |
| "keep\_files" | Whether to keep temporary solver files | Bool (default: False) |
| "io\_api" | Solver option: [See linopy package](https://linopy.readthedocs.io/en/latest/generated/linopy.model.Model.solve.html#linopy.model.Model.solve) | String |
| "analyze\_numerics" | Whether details about the optimization problem's numerics should be printed in the console | Bool (default: False) |
| "recommend\_base\_units" | Check if there is a better set of base units in terms of the optimizations numerics | Bool (default: False) |
| "immutable\_unit" | Define base units which must not be changed to find a better set of base units | List of strings |
| "range\_of\_exponents" | Range of exponents the base units are allowed to be shifted to become "better" e.g. {"min": -2, "max": 3} base units can be decreased by 10^-2 or increased by 10^3 | Dictionary (default: {"min": -3, "max": 3} |
| "define\_ton\_as\_metric\_ton" | Whether the unit "ton" should be a metric ton (True) or an imperial ton (False) | Bool (default: True) |
| "rounding\_decimal\_points" | Rounding tolerance for new capacity and unit multipliers | Positive integer (default: 5) |
| "rounding\_decimal\_points\_ts" | Rounding tolerance for time series after TSA | Positive integer (default: 5) |
| "linear\_regression\_check" | Tolerances to determine if PWA-modelled x-y relationships can be represented by a linear slope while meeting these tolerances | Dictionary (default: {"eps\_intercept": 0.1, "epsRvalue": 1-1E-5} |

### Parameter, Variable and Constraint Overview

 **Parameters** 

| **Name:** | **Time Step Type:** | **Doc String:** | **Scope** : | **Unit Category:** |
| --- | --- | --- | --- | --- |
| carbon\_emissions\_annual\_limit | set\_time\_steps\_yearly | Parameter which specifies the total limit on carbon emissions | energy system | {"emissions": 1}|
| carbon\_emissions\_budget | temporal immutable | Parameter which specifies the total budget of carbon emissions until the end of the entire time horizon | energy system | {"emissions": 1}|
| carbon\_emissions\_cumulative\_existing | temporal immutable | Parameter which specifies the total previous carbon emissions | energy system | {"emissions": 1}|
| price\_carbon\_emissions | set\_time\_steps\_yearly | Parameter which specifies the yearly carbon price | energy system | {"money": 1, "emissions": -1}|
| price\_carbon\_emissions\_budget\_overshoot | temporal immutable | Parameter which specifies the carbon price for budget overshoot | energy system | {"money": 1, "emissions": -1}|
| price\_carbon\_emissions\_annual\_overshoot | temporal immutable | Parameter which specifies the carbon price for annual overshoot | energy system | {"money": 1, "emissions": -1}|
| market\_share\_unbounded | temporal immutable | Parameter which specifies the unbounded market share | energy system | {}|
| knowledge\_spillover\_rate | temporal immutable | Parameter which specifies the knowledge spillover rate | energy system |{}|
| time\_steps\_operation\_duration | set\_time\_steps\_operation | Parameter which specifies the time step duration in operation for all technologies | energy system | {"time": 1}|
| demand | set\_time\_steps\_operation | Parameter which specifies the carrier demand | carrier | {"energy_quantity": 1, "time": -1}|
| availability\_import | set\_time\_steps\_operation | Parameter which specifies the maximum energy that can be imported from outside the system boundaries | carrier | {"energy_quantity": 1, "time": -1}|
| availability\_export | set\_time\_steps\_operation | Parameter which specifies the maximum energy that can be exported to outside the system boundaries | carrier | {"energy_quantity": 1, "time": -1}|
| availability\_import\_yearly | set\_time\_steps\_yearly | Parameter which specifies the maximum energy that can be imported from outside the system boundaries for the entire year | carrier | {"energy_quantity": 1}|
| availability\_export\_yearly | set\_time\_steps\_yearly | Parameter which specifies the maximum energy that can be exported to outside the system boundaries for the entire year | carrier | {"energy_quantity": 1}|
| price\_import | set\_time\_steps\_operation | Parameter which specifies the import carrier price | carrier | {"money": 1, "energy_quantity": -1}|
| price\_export | set\_time\_steps\_operation | Parameter which specifies the export carrier price | carrier | {"money": 1, "energy_quantity": -1}|
| price\_shed\_demand | temporal immutable | Parameter which specifies the price to shed demand | carrier | {"money": 1, "energy_quantity": -1}|
| carbon\_intensity\_carrier | set\_time\_steps\_yearly | Parameter which specifies the carbon intensity of carrier | carrier | {"emissions": 1, "energy_quantity": -1}|
| capacity\_existing | temporal immutable | Parameter which specifies the existing technology size | technology | {"energy_quantity": 1, "time": -1}|
| capacity\_investment\_existing | set\_time\_steps\_yearly\_entire\_horizon | Parameter which specifies the size of the previously invested capacities | technology | {"energy_quantity": 1, "time": -1}|
| capacity\_addition\_min | temporal immutable | Parameter which specifies the minimum capacity addition that can be installed | technology | {"energy_quantity": 1, "time": -1}|
| capacity\_addition\_max | temporal immutable | Parameter which specifies the maximum capacity addition that can be installed | technology | {"energy_quantity": 1, "time": -1}|
| capacity\_addition\_unbounded | temporal immutable | Parameter which specifies the unbounded capacity addition that can be added each year (only for delayed technology deployment) | technology | {"energy_quantity": 1, "time": -1}|
| lifetime\_existing | temporal immutable | Parameter which specifies the remaining lifetime of an existing technology | technology | {}|
| capex\_capacity\_existing | temporal immutable | Parameter which specifies the total capex of an existing technology which still has to be paid | technology | {"money": 1, "energy_quantity": -1}|
| opex\_specific\_variable | set\_time\_steps\_operation | Parameter which specifies the variable specific opex | technology | {"money": 1, "energy_quantity": -1}|
| opex\_specific\_fixed | set\_time\_steps\_yearly | Parameter which specifies the fixed annual specific opex | technology | {"money": 1, "energy_quantity": -1, "time": 1}|
| lifetime | temporal immutable | Parameter which specifies the lifetime of a newly built technology | technology | {}|
| construction\_time | temporal immutable | Parameter which specifies the construction time of a newly built technology | technology | {}|
| max\_diffusion\_rate | set\_time\_steps\_yearly | Parameter which specifies the maximum diffusion rate which is the maximum increase in capacity between investment steps | technology | {}|
| capacity\_limit | temporal immutable | Parameter which specifies the capacity limit of technologies | technology | {"energy_quantity": 1, "time": -1}|
| min\_load | set\_time\_steps\_operation | Parameter which specifies the minimum load of technology relative to installed capacity | technology | {}|
| max\_load | set\_time\_steps\_operation | Parameter which specifies the maximum load of technology relative to installed capacity | technology | {}|
| carbon\_intensity\_technology | temporal immutable | Parameter which specifies the carbon intensity of each technology | technology | {"emissions": 1, "energy_quantity": -1}|
| retrofit\_flow\_coupling\_factor | set\_time\_steps\_operation | Parameter which specifies the flow coupling between the retrofitting technologies and its base technology | technology| {"energy_quantity": 1, "energy_quantity": -1}|
| capex\_specific\_conversion | set\_time\_steps\_yearly | Parameter which specifies the slope of the capex if approximated linearly | conversion technology | {"money": 1, "energy_quantity": -1, "time": 1}|
| conversion\_factor | set\_time\_steps\_yearly | Parameter which specifies the slope of the conversion efficiency if approximated linearly | conversion technology | {"energy_quantity": 1, "energy_quantity": -1}|
| time\_steps\_storage\_level\_duration | set\_time\_steps\_storage\_level | Parameter which specifies the time step duration in StorageLevel for all technologies | storage technology | {"time": 1}|
| efficiency\_charge | set\_time\_steps\_yearly | efficiency during charging for storage technologies | storage technology | {}|
| efficiency\_discharge | set\_time\_steps\_yearly | efficiency during discharging for storage technologies | storage technology | {}|
| self\_discharge | temporal immutable | self-discharge of storage technologies | storage technology | {}|
| capex\_specific\_storage | set\_time\_steps\_yearly | specific capex of storage technologies | storage technology | {"money": 1, "energy_quantity": -1, "time": 1}|
| distance | temporal immutable | distance between two nodes for transport technologies | transport technology | {"distance": 1}|
| capex\_specific\_transport | set\_time\_steps\_yearly | capex per unit for transport technologies | transport technology | {"money": 1, "energy_quantity": -1, "time": 1}|
| capex\_per\_distance\_transport | set\_time\_steps\_yearly | capex per distance for transport technologies | transport technology | {"money": 1, "distance": -1, "energy_quantity": -1, "time": 1}|
| transport\_loss\_factor | temporal immutable | carrier losses due to transport with transport technologies | transport technology | {"distance": -1}|
| transport\_loss\_factor\_exponential | temporal immutable | exponential carrier losses due to transport with transport technologies | transport technology | {"distance": -1}|


 **Variables** 

| **Name:**                            | **Time Step Type:** | **Doc String:** | **Scope:** | **Unit Category:** |
|--------------------------------------| --- | --- | --- | --- |
| carbon\_emissions\_annual            | set\_time\_steps\_yearly | annual carbon emissions of energy system | energy system | {"emissions": 1}|
| carbon\_emissions\_cumulative        | set\_time\_steps\_yearly | cumulative carbon emissions of energy system over time for each year | energy system | {"emissions": 1}|
| carbon\_emissions\_budget\_overshoot | set\_time\_steps\_yearly | overshoot carbon emissions of energy system at the end of the time horizon | energy system | {"emissions": 1}|
| carbon\_emissions\_annual\_overshoot | set\_time\_steps\_yearly | overshoot of the annual carbon emissions limit of energy system | energy system | {"emissions": 1}|
| cost\_carbon\_emissions\_total       | set\_time\_steps\_yearly | total cost of carbon emissions of energy system | energy system | {"money": 1}|
| cost\_total                          | set\_time\_steps\_yearly | total cost of energy system | energy system | {"money": 1}|
| net\_present\_cost                   | set\_time\_steps\_yearly | net\_present\_cost of energy system | energy system | {"money": 1}|
| flow\_import                         | set\_time\_steps\_operation | node- and time-dependent carrier import from the grid | carrier | {"energy_quantity": 1, "time": -1}|
| flow\_export                         | set\_time\_steps\_operation | node- and time-dependent carrier export from the grid | carrier | {"energy_quantity": 1, "time": -1}|
| cost\_carrier                        | set\_time\_steps\_operation | node- and time-dependent carrier cost due to import and export | carrier | {"money": 1, "time": -1}|
| cost\_carrier\_total                 | set\_time\_steps\_yearly | total carrier cost due to import and export | carrier | {"money": 1}|
| carbon\_emissions\_carrier           | set\_time\_steps\_operation | carbon emissions of importing and exporting carrier | carrier |{"emissions": 1, "time": -1}|
| carbon\_emissions\_carrier\_total    | set\_time\_steps\_yearly | total carbon emissions of importing and exporting carrier | carrier | {"emissions": 1}|
| shed\_demand                         | set\_time\_steps\_operation | shed demand of carrier | carrier | {"energy_quantity": 1, "time": -1}|
| cost\_shed\_demand                   | set\_time\_steps\_operation | shed demand of carrier | carrier | {"money": 1, "time": -1}|
| technology\_installation             | set\_time\_steps\_yearly | installment of a technology at location l and time t | technology | {}|
| capacity                             | set\_time\_steps\_yearly | size of installed technology at location l and time t | technology | {"energy_quantity": 1, "time": -1}|
| capacity\_addition                   | set\_time\_steps\_yearly | size of built technology (invested capacity after construction) at location l and time t | technology | {"energy_quantity": 1, "time": -1}|
| capacity\_investment                 | set\_time\_steps\_yearly | size of invested technology at location l and time t | technology | {"energy_quantity": 1, "time": -1}|
| cost\_capex                          | set\_time\_steps\_yearly | capex for building technology at location l and time t | technology | {"money": 1}|
| capex\_yearly                        | set\_time\_steps\_yearly | annual capex for having technology at location l | technology | {"money": 1}|
| cost\_capex\_total                   | set\_time\_steps\_yearly | total capex for installing all technologies in all locations at all times | technology | {"money": 1}|
| cost\_opex                           | set\_time\_steps\_operation | opex for operating technology at location l and time t | technology |{"money": 1, "time": -1}|
| cost\_opex\_yearly                   | set\_time\_steps\_yearly | yearly opex for operating technology at location l and year y | technology | {"money": 1}|
| cost\_opex\_total                    | set\_time\_steps\_yearly | total opex all technologies and locations in year y | technology | {"money": 1}|
| carbon\_emissions\_technology        | set\_time\_steps\_operation | carbon emissions for operating technology at location l and time t | technology | {"emissions": 1, "time": -1}|
| carbon\_emissions\_technology\_total | set\_time\_steps\_yearly | total carbon emissions for operating technology at location l and time t | technology | {"emissions": 1}|
| flow\_conversion\_input              | set\_time\_steps\_operation | Carrier input of conversion technologies | conversion technology | {"energy_quantity": 1, "time": -1}|
| flow\_conversion\_output             | set\_time\_steps\_operation | Carrier output of conversion technologies | conversion technology | {"energy_quantity": 1, "time": -1}|
| capacity\_approximation              | set\_time\_steps\_yearly | pwa variable for size of installed technology on edge i and time t | technology | {"energy_quantity": 1, "time": -1}|
| capex\_approximation                 | set\_time\_steps\_yearly | pwa variable for capex for installing technology on edge i and time t | technology | {"money": 1}|
| flow\_approximation\_reference       | set\_time\_steps\_operation | pwa of flow of reference carrier of conversion technologies | conversion technology | {"energy_quantity": 1, "time": -1}|
| flow\_approximation\_dependent       | set\_time\_steps\_operation | pwa of flow of dependent carriers of conversion technologies | conversion technology | {"energy_quantity": 1, "time": -1}|
| flow\_storage\_charge                | set\_time\_steps\_operation | carrier flow into storage technology on node i and time t | storage technology | {"energy_quantity": 1, "time": -1}|
| flow\_storage\_discharge             | set\_time\_steps\_operation | carrier flow out of storage technology on node i and time t | storage technology | {"energy_quantity": 1, "time": -1}|
| storage\_level                       | set\_time\_steps\_storage\_level | storage level of storage technology on node in each storage time step | storage technology | {"energy_quantity": 1}|
| flow\_transport                      | set\_time\_steps\_operation | carrier flow through transport technology on edge i and time t | transport technology | {"energy_quantity": 1, "time": -1}|
| flow\_transport\_loss                | set\_time\_steps\_operation | carrier flow through transport technology on edge i and time t | transport technology | {"energy_quantity": 1, "time": -1}|
| tech\_on\_var                        | set\_time\_steps\_operation | Binary variable which equals 1 when technology is switched on at location l and time t, else 0 | technology | {}|
| tech\_off\_var                       | set\_time\_steps\_operation | Binary variable which equals 1 when technology is switched off at location l and time t, else 0 | technology | {}|

 |

 **Constraints** 

| **Name:**                                        | **Time Step Type:** | **Doc String:** | **Scope:** |
|--------------------------------------------------| --- | --- | --- |
| constraint\_carbon\_emissions\_annual            | set\_time\_steps\_yearly | total annual carbon emissions of energy system | energy system |
| constraint\_carbon\_emissions\_cumulative        | set\_time\_steps\_yearly | cumulative carbon emissions of energy system over time | energy system |
| constraint\_cost\_carbon\_emissions\_total       | set\_time\_steps\_yearly | total carbon emissions cost of energy system | energy system |
| constraint\_carbon\_emissions\_annual\_limit     | set\_time\_steps\_yearly | limit of total annual carbon emissions of energy system | energy system |
| constraint\_carbon\_emissions\_budget            | set\_time\_steps\_yearly | Budget of total carbon emissions of energy system | energy system |
| constraint\_carbon\_emissions\_budget\_overshoot | set\_time\_steps\_yearly | Disable carbon emissions budget overshoot if carbon emissions budget overshoot price is inf | energy system |
| constraint\_carbon\_emissions\_annual\_overshoot | set\_time\_steps\_yearly | Disable annual carbon emissions overshoot if annual carbon emissions overshoot price is inf | energy system |
| constraint\_carbon\_emissions\_overshoot\_limit  | set\_time\_steps\_yearly | Limit of overshot carbon emissions of energy system | energy system |
| constraint\_cost\_total                          | set\_time\_steps\_yearly | total cost of energy system | energy system |
| constraint\_net\_present\_cost                   | set\_time\_steps\_yearly | net\_present\_cost of energy system | energy system |
| constraint\_availability\_import                 | set\_time\_steps\_operation | node- and time-dependent carrier availability to import from outside the system boundaries | carrier |
| constraint\_availability\_export                 | set\_time\_steps\_operation | node- and time-dependent carrier availability to export to outside the system boundaries | carrier |
| constraint\_availability\_import\_yearly         | set\_time\_steps\_yearly | node- and time-dependent carrier availability to import from outside the system boundaries summed over entire year | carrier |
| constraint\_availability\_export\_yearly         | set\_time\_steps\_yearly | node- and time-dependent carrier availability to export to outside the system boundaries summed over entire year | carrier |
| constraint\_cost\_carrier                        | set\_time\_steps\_operation | cost of importing and exporting carrier | carrier |
| constraint\_cost\_shed\_demand                   | set\_time\_steps\_operation | cost of shedding carrier demand | carrier |
| constraint\_limit\_shed\_demand                  | set\_time\_steps\_operation | limit of shedding carrier demand | carrier |
| constraint\_cost\_carrier\_total                 | set\_time\_steps\_yearly | total cost of importing and exporting carriers | carrier |
| constraint\_carbon\_emissions\_carrier           | set\_time\_steps\_operation | carbon emissions of importing and exporting carrier | carrier |
| constraint\_carbon\_emissions\_carrier\_total    | set\_time\_steps\_yearly | total carbon emissions of importing and exporting carriers | carrier |
| constraint\_nodal\_energy\_balance               | set\_time\_steps\_operation | node- and time-dependent energy balance for each carrier | carrier |
| constraint\_technology\_capacity\_limit          | set\_time\_steps\_yearly | limited capacity of technology depending on loc and time | technology |
| constraint\_technology\_min\_capacity            | set\_time\_steps\_yearly | min capacity of technology that can be installed | technology |
| constraint\_technology\_max\_capacity            | set\_time\_steps\_yearly | max capacity of technology that can be installed | technology |
| constraint\_technology\_construction\_time       | set\_time\_steps\_yearly | lead time in which invested technology is constructed | technology |
| constraint\_technology\_lifetime                 | set\_time\_steps\_yearly | max capacity of technology that can be installed | technology |
| constraint\_technology\_diffusion\_limit         | set\_time\_steps\_yearly | Limits the newly built capacity by the existing knowledge stock | technology |
| constraint\_capacity\_factor                     | set\_time\_steps\_operation | limit max load by installed capacity | technology |
| constraint\_capex\_yearly                        | set\_time\_steps\_yearly | annual capex of having capacity of technology. | technology |
| constraint\_cost\_capex\_total                   | set\_time\_steps\_yearly | total capex of all technology that can be installed. | technology |
| constraint\_opex\_technology                     | set\_time\_steps\_operation | opex for each technology at each location and time step | technology |
| constraint\_cost\_opex\_yearly                   | set\_time\_steps\_yearly | total opex of all technology that are operated. | technology |
| constraint\_cost\_opex\_total                    | set\_time\_steps\_yearly | total opex of all technology that are operated. | technology |
| constraint\_carbon\_emissions\_technology        | set\_time\_steps\_operation | carbon emissions for each technology at each location and time step | technology |
| constraint\_carbon\_emissions\_technology\_total | set\_time\_steps\_yearly | total carbon emissions for each technology at each location and time step | technology |
| disjunct\_on\_technology                         | set\_time\_steps\_operation | disjunct to indicate that technology is on | technology |
| disjunct\_off\_technology                        | set\_time\_steps\_operation | disjunct to indicate that technology is off | technology |
| disjunction\_decision\_on\_off\_technology       | set\_time\_steps\_operation | disjunction to link the on off disjuncts | technology |
| constraint\_linear\_capex                        | set\_time\_steps\_yearly | Linear relationship in capex | technology |
| constraint\_linear\_conversion\_factor           | set\_time\_steps\_operation | Linear relationship in conversion\_factor | conversion technology |
| constraint\_capex\_coupling                      | set\_time\_steps\_yearly | couples the real capex variables with the approximated variables | conversion technology |
| constraint\_capacity\_coupling                   | set\_time\_steps\_yearly | couples the real capacity variables with the approximated variables | conversion technology |
| constraint\_reference\_flow\_coupling            | set\_time\_steps\_operation | couples the real reference flow variables with the approximated variables | conversion technology |
| constraint\_dependent\_flow\_coupling            | set\_time\_steps\_operation | couples the real dependent flow variables with the approximated variables | conversion technology |
| constraint\_storage\_level\_max                  | set\_time\_steps\_storage\_level | limit maximum storage level to capacity | storage technology |
| constraint\_couple\_storage\_level               | set\_time\_steps\_storage\_level | couple subsequent storage levels (time coupling constraints) | storage technology |
| constraint\_storage\_technology\_capex           | set\_time\_steps\_yearly | Capital expenditures for installing storage technology | storage technology |
| constraint\_transport\_technology\_losses\_flow  | set\_time\_steps\_operation | Carrier loss due to transport with through transport technology | transport technology |
| constraint\_transport\_technology\_capex         | set\_time\_steps\_yearly | Capital expenditures for installing transport technology | transport technology |
| constraint\_transport\_technology\_bidirectional | set\_time\_steps\_yearly | Forces that transport technology capacity must be equal in both directions | transport technology |

 |

9
