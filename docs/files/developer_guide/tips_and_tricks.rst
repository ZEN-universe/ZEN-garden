.. _tips.tips:

############################
Tips and Tricks
############################


.. _tips.call_graph:

Creating a call graph
======================

A call graph shows the methods/functions that are called in the ZEN-garden
and the order in which they are called. It reveals structure of the code.

The packages `pycallgraph2 <https://anaconda.org/conda-forge/pycallgraph2>`_ and 
`graphviz <https://graphviz.org/>`_ can be used to create a call graph of 
ZEN-garden (or sections thereof). To add these packages to the 
ZEN-garden environment, open the Anaconda Prompt app and :ref:`activate the 
ZEN-garden environment <installation.activate>`. Then run the following 
commands:

.. code::

    conda install -c conda-forge graphviz
    conda install conda-forge::pycallgraph2

Once these packages are installed, the following syntax can be used to 
create a call graph:

.. code:: python

    from pycallgraph2 import PyCallGraph, Config, GlobbingFilter
    from pycallgraph2.output import GraphvizOutput

    # set parameters
    max_depth = 10
    include_pattern = 'zen_garden.*'
    output_file = 'output.png'

    # set configurations (specified by the above parameters)
    config = Config(max_depth = max_depth)
    config.trace_filter = GlobbingFilter(include=[
        include_pattern  # only include functions which match the pattern
    ])
    graphviz = GraphvizOutput(output_file = output_file) # set output file

    # run pycallgraph
    with PyCallGraph(output=graphviz, config = config):
        function_to_profile()

In this script, replace ``function_to_profile()`` with code to 
analyze. Additionally, several parameters control the output:

* ``max_depth`` - sets the maximum depth of function calls which to show in the 
  call graph.
* ``include`` - pattern that function names must contain to be included.
  This filters the included functions. For example, the string ``"zen_garden.*"``
  specifies that only methods of the zen_garden package should be included.
* ``output_file`` - specifies the output file name where to save the call graph.
  By default, the file gets saved to the current working directory.

