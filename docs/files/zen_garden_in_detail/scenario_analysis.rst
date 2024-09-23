.. _scenario_analysis:
################
Scenario analysis
################

The scenario tool allows the user to overwrite parameter values and run a given model multiple times with slight variations to the input data, the system or analysis settings. In the following we discuss the features of the scenario tool. Specifically, 

* How to define scenarios
* Hierarchical expansion of sets to quickly define parameter changes for multiple elements
* Defining parameter values in lists to avoid lengthy manual definitions
* How to overwrite system and analysis settings

.. _scenario_definition:
Scenario definition 
=====================
Scenarios are defined in the ``scenarios.json``::

    {"scenario_name_1":
        {"element_1": 
            {"param_1": {
                "default": "<attribute_file_name>",
                "default_op": <float>,  
                "file": "<file_name>",
                "file_op": <float>
                }            
            }
        }
    }

Each scenario has a unique name. For each element of the ``energy_system``, as well as the ``energy_system`` itself, the parameters can be overwritten to perform a different analysis. Four options are available:

* ``default``: Change the filename from which the default value is taken
* ``default_op``: Multiply default value by a constant factor 
* ``file``: Change the file name from which the values are taken to overwrite the default values
* ``file_op``: Multiply the parameter values after reading the default value and overwriting the default values with the file values by a constant factor

It is also possible to combine the four options. For example, if you would like to change the import price for the element ``natural_gas``the ``scenario.json``would look like this::

    {"high_gas_price":
        {"natural_gas": 
            {"price_import": {
                "default": "attributes_high",
                "default_op": 1.5,  
                "file": "price_import_high",
                "file_op": 0.9
                }            
            }
        }
    }

In this example, first the default value would be read from ``attributes_high.json``. Thereafter, the default value would be multiplied by 1.5. Now, the values specified in the file ``price_import_high.csv`` are read and overwrite the corresponding default values. Lastly, the parameter values are multiplied by 0.9.

.. note:: 
    ``file_op`` is applied after the file values have replaced the default values and will therefore be applied to **all** parameter values, the default values as well. Thus, setting both ``default_op`` and ``file_op`` will change the default values twice.

.. note::
    If you want to change the yearly variation of a time-dependent parameter, i.e., adding a file for demand_yearly_variation, please use ``demand_yearly_variation`` directly.

    .. code-block::

        {"example": {
            "electricity": {
                "demand_yearly_variation":{
                    "file":"demand_yearly_variation_high"
                    }
                }
            }

Note that you overwrite the demand_yearly_variation parameter, not demand.

.. _overwriting_sets:
Overwriting entire sets or subsets
==================================

In some cases, we would like to change a parameter for all elements of a set. To do this, we use the same syntax, but use the set name instead of the element name::

    {"example": {
        "set_technologies": {
            "max_load": {
                "file": "max_load_5",
                "file_op": 1.5,
                "default": "attributes_v2", 
                "default_op": 0.25,
                "exclude": ["tech1", "tech2"]
                }
            }
        }
    }

For sets, an additional key ``"exclude"`` is allowed, which allows us to define a list of set-elements that should not be overwritten. The set expansion works hierarchical, meaning that if we define the same parameter for an element of the set, this parameter will not be touched at all. For example, let's say we have ``set_technologies = ["tech1", "tech2"]``::

    {"new_example": {
        "set_technologies": {
            "max_load": {
                "file": "max_load_5"
                }
            },
        "tech1": {
            "max_load": {
                "default": 3
                }
            }
        }
    }

after expansion the final scenarios dictionary would be::

    {"new_example": {
        "tech1": {
            "max_load": {
                "default": 3
                }
            },
        "tech2": {
            "max_load": {
                "file": "max_load_5"
                }
            }
        }
    }

This hierarchy is continued for smaller sets, e.g. defining ``set_transport_technologies`` takes precedence to ``set_technologies``, etc.

.. _defining_scenario_params_with_lists:
 Defining parameters with lists
 ==============================

 It is also to define parameters in lists::

    {"price_range": {
        "natural_gas": {
            "import_price": {
                "default": "attributes_high",
                "default_op": [0.25, 0.3, 0.35]
                }
            }
        }
    }

Will create 3 new scenarios for all values specified in ``default_op``. All keys support the option to pass lists instead of strings or floats, however, it is important that the value is a proper Python list, not an array or something else. To avoid errors, we recommend wrapping your values in ``list(...)``, especially if you generate the iterable with ``np.linspace()``, ``range()`` or similar. If multiple lists are defined within the same scenario, all possible combinations (cartesian product) are investigated, so watch out for combinatorial explosions.

Per default, the names for the generated scenarios are "p{i:02d}_{j:03d}", where i is an int referring to the expanded parameter name (e.g. ``natural_gas``, ``import_price``, ``file``, ``default_op``) and j to its value in the list (e.g. ``[0.25, 0.3, 0.35]``). The mappings of ``i`` and ``j`` to the parameter names and values are written to  ``param_map.json`` in the root directory of the corresponding scenario (see below). It is possible to overwrite this default naming with a formatting key::

    {"price_range": {
        "natural_gas": {
            "import_price": {
                "default": "attributes_high",
                "default_op": [0.25, 0.3, 0.35],
                "default_op_fmt": "high_gas_price_{}"
                }
            }
        }

The formatting key is the original key containing the list followed by "_fmt". The value of the formatting key has to be a string containing the format literal "{}". The formatting string "{}" will then be replaced by each of the values of the list. For example here, we would generate the three scenarios ``high_gas_price_0.25``, ``high_gas_price_0.3`` and ``high_gas_price_0.35``.

When a scenario contains one or multiple lists, all sub-scenarios are also in a subfolder, for example, the output structure could look something like this::

    dataset_1/
        scenario_1/
        scenario_2/
            scenario_p00_000_p001_000/
            scenario_p01_000_p001_000/
            ...
            param_map.json
        scenario_3/
    ...

Here, ``scenario_2`` was defined via lists and its sub-scenarios are now in subfolders with the definitions of the parameters in the ``param_map.json``. 

.. _scenarios_using_sets_and_lists:
Using both, sets and lists
==============================

When using both, set and list expansion, list expansion is done first. For example::

    {"example": {
        "set_carriers": {
            "price_import": {
                "file_op": [1.5, 2.5, 3.5],
                "exclude": ["carrier1", "carrier2"]
                }
            }
        }
    }

will only generate 3 scenarios where the ``file_op`` for all technologies (except ``["carrier1", "carrier2"]``) are set to the values in the lists simultaneously.

.. _scenarios_analysis_system:
Overwriting Analysis and System
==============================

It is also possible to overwrite entries in the system and analysis settings. The syntax is as follows::

    {"example": {
        "system": {
            key: value
            },
        "natural_gas": {
            "price_import": {
                "file": "import_price_high",
                "file_op": 1.5
                }
            }
        }
    }

Note that there is a strict type check when overwriting the system or analysis, i.e. the value used for ``value`` must have the same type as the value already in the dictionary.

.. _scenarios_running_the_analysis:
Running the analysis
=====================

Per default, all scenarios are run sequentially, as before. Additionally, one can specify a subset of scenarios to run with the --job_index argument. For example::

    python -m zen_garden --job_index 1,4,7

will run scenarios 1,4,7, where the number is the index of the key (starting with 0), not the key itself (no explicit scenario names).

.. note::

    When submitting a job on the cluster per default all scenarios are run sequentially. However, you can also run jobs in parallel by specifying the scenarios via the ``--array=start-stop:step%Nmax`` argument (start and stop are inclusive, Nmax is the max number of concurrent jobs). Other ``--array`` options are e.g. ``--array=1,4,7``, which will run only the specified jobs. Note that the indices start with 0, so running the first four scenarios would be ``--array=0-3`` (per default the step is 1 and Nmax default to the number of submitted jobs). 