.. _plugins.intro:

############################
Overview
############################

The ZEN-garden plugin system lets you extend or modify the behaviour of a model run
without changing the ZEN-garden source code.
A plugin is a small, self-contained Python package that you develop and install
separately.  ZEN-garden detects installed plugins automatically and runs them at
well-defined points during execution.

How it works
------------

ZEN-garden defines a set of **events** — specific points in the workflow.
A plugin registers one or more **functions** that should be called when a particular
event occurs.  When ZEN-garden reaches that event, it calls all registered functions
in the order they were registered.

.. _plugins.available_events:

Available events
-----------------
At the moment no events are available.


Plugin discovery
----------------

Plugins are discovered through Python's standard
`entry point <https://packaging.python.org/en/latest/specifications/entry-points/>`_
mechanism.  When you install a plugin package (``pip install``), it advertises
itself under the ``zen_garden.plugins`` group.  ZEN-garden then finds and loads it
automatically — no changes to ZEN-garden are required.

Activating a plugin
-------------------

Add the plugin's name to ``config.json`` under the ``"plugins"`` key:

.. code-block:: json

    {
        "plugins": {
            "my_plugin_name": {
                "some_setting": 42
            }
        }
    }

The settings you provide are merged into the plugin's ``config`` dictionary before
any of its functions are called.

See also
--------

- :ref:`Implementing plugins <dev_guide.implementing_plugins>`
