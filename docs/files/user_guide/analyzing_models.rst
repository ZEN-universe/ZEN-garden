################
Analyzing a model
################

.. _Accessing results:
Accessing results
=================
The results of ZEN-garden are stored in the ``output`` folder in the ``data`` directory in a folder with the name of the dataset.

.. note::
    The results are self-contained, i.e., you can copy the ``<dataset_name>`` folder to another location and still access the results.

The easiest way to access the results values is to import the ``Results`` class from the ``zen_garden`` package in, e.g., Pycharm or Jupyter Notebook, with your ``zen-garden`` environment activated and load the results with the following code::

    from zen_garden.postprocess.results.results import Results
    r = Results(path='<result_folder>')

The path ``<result_folder>`` is the path to the results folder of the dataset, e.g., ``data/output/<dataset_name>``.
To access the data frames containing the raw optimization results of the variables and parameters (called component in the following), the following member functions for the instance ``r`` of the ``Results`` class can be used:

1. ``r.get_total()``: Returns annual total values of the component.
2. ``r.get_full_ts()``: Returns the time series of the component. In case of hourly-resolved data, the time series has a length of 8760 times the number of years simulated.
3. ``r.get_dual()``: Returns the dual values of the constraints.
4. ``r.get_unit()``: Returns the units of the component.
5. ``r.get_doc()``: Returns the documentation string of the component.

The user must pass the name of the ``component`` to the member functions, e.g., ``r.get_total('capacity')`` to access the annual capacity values for all technologies.
Optional arguments can be passed to the member functions to filter the results. The optional arguments are:

1. ``element_name``: A single element name that is in the first index of the component (e.g., "wind_onshore" for "capacity", if the technology "wind_onshore" is modeled). Not available for ``r.get_unit()``.
2. ``year``: A single optimization period for which the results should be returned (0, 1, 2, ...). Not available for ``r.get_unit()``.
3. ``scenario_name``: A single scenario name for which the results should be returned.

To return the names of all components, the following member function can be used::

    r.get_component_names(<component_type>)

The argument ``<component_type>`` can be one of the following: ``'parameter'``, ``'variable'``, ``'dual'``, ``'sets'``.

.. _Visualization:
User guide for visualization
=================

If you have followed the steps of chapter :ref:`installation`, you should have a conda environment or a virtual environment that contains the necessary python packages to run the visualization suite.

After successfully running an optimization. you can start the visualization with ``python -m zen_garden.visualization``.

.. note::

    By default, the suite looks for solutions that are contained in the folder ``./outputs``, relatively to where you run the command. If you are copying results from somewhere else, make sure to create a folder called ``outputs`` and copy the results there.
    Alternatively, you can pass an arbitrary folder with ``python -m zen_garden.visualization <path to your solutions folder>`` to change the solutions folder.

This command will open a new tab in your default browser with the correct URL.
If the tab does not open automatically, you can open http://localhost:8000/explorer in any browser of your choice.

To interrupt the visualization, you can press ``Ctrl+C`` in the terminal where you started the visualization.

You can investigate precomputed results online with the visualization suite by visiting the following link: https://zen-garden.ethz.ch/explorer/

.. _Comparing results:
Comparing results
=================
ZEN-garden provides methods to compare two different result objects. This can be helpful to understand why two results differ.
Furthermore, it allows for a fast way to spot errors in the datasets.
The most useful application is to compare the configuration (:ref:`System, analysis, solver settings`) of two datasets and the parameter values.
Comparing variable values is often not very informative, as the results mostly differ in a large variety of variables.
Let's assume you have the following two result objects::

    from zen_garden.postprocess.results.results import Results
    r1 = Results(path='<result_folder_1>')
    r2 = Results(path='<result_folder_2>')

Then you can compare the two result objects with the following code::

    from zen_garden.postprocess.results.comparisons import compare_model_values, compare_configs
    compare_parameters = compare_model_values([r1, r2], component_type = 'parameter')
    compare_variables = compare_model_values([r1, r2], component_type = 'variable')
    compare_config = compare_configs([r1, r2])

Per default, ``compare_model_values`` compares the total annual values of components (:ref:`Accessing results`). If the user wants to compare the full time series, the optional argument ``compare_total=False`` can be passed to the function.
``compare_model_values`` also accepts ``component_type = "dual"`` and ``component_type = "sets"``.

``compare_configs`` compares the configurations of the two datasets.
