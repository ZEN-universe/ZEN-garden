.. _plugins.intro:

############################
Plugin System Overview
############################

What are plugins?
-----------------

Plugins extend ZEN-garden's functionality without modifying the core codebase.
They are Python packages that "hook into" ZEN-garden at specific points during execution.

Think of it like this: ZEN-garden has a workflow. At certain moments (called "events"),
it pauses and asks: *"Does anyone want to do something here?"* Plugins raise their hand
and say: *"Yes, I can add a new constraint"* or *"I can modify the results"*.

Plugins live in the
`ZEN-garden plugins repository <https://github.com/ZEN-universe/ZEN-garden-plugins>`_
or be developed privately. You can install them like any other Python package and
activate them in your ``config.yaml``.

**Key idea:** You write a function, decorate it with the event you at which the
function should run, package it as a Python package, install it, add the plugin to
your ``config.yaml``, and ZEN-garden will call your function at the respective event.


Using an existing plugin
------------------------

**Installation**

Install a plugin package from a local path, like this (set the -e flag for editable mode):

.. code-block:: shell

     uv pip install -e path\to\zen_garden_plugins --no-deps


**Activation**

Add the plugin to your ZEN-garden ``config.yaml`` under the ``plugins`` section:

.. code-block:: yaml

    plugins:
      myplugin:
        my_setting: 123
        another_setting: "value"

The settings you provide here are passed to the plugin so it knows how to behave.
Each plugin's documentation explains what settings it accepts.

**How it works**

When ZEN-garden starts:

1. It reads your ``config.yaml`` and sees which plugins are listed
2. It locates and imports each plugin
3. It passes your settings to the plugin's configuration
4. During execution, whenever an event occurs, your plugin's functions are called automatically

Creating a plugin
-----------------

To create your own plugin, see the full implementation guide in the documentation of the
`ZEN-garden plugins repository <https://github.com/ZEN-universe/ZEN-garden-plugins>`_.

See also
--------

- Plugin configuration: :ref:`Configuration options for plugins <configuration.plugins>`
