.. _dev_guide.implementing_plugins:

############################
Implementing plugins
############################

This page walks you through creating a plugin for ZEN-garden from scratch.
A plugin lives in its own repository and is installed as a regular Python package.
ZEN-garden picks it up automatically once it is installed.

.. note::

   Plugins are intentionally kept separate from ZEN-garden. You should never need
   to modify ZEN-garden itself to add a plugin.


Step 1 — Create a repository
-------------------------------

Fork the Zen garden plugin template repository on GitHub (add link here).

The directory (``plugin_template/``) is the Python package.
All plugin logic goes into ``plugin.py``. You can define helper modules if you like,
but the loader only looks for ``plugin.py``.


Step 2 — Adapt ``pyproject.toml``
----------------------------------

The plugin must be a valid Python package with a ``pyproject.toml`` that advertises
itself as a plugin for ZEN-garden. You can adapt the template's ``pyproject.toml``. Essentially you only need
to change the name. In case your plugin relies on additional packages, you also need to specify the
dependencies

Step 3 — Develop your plugin
-----------------------------

Every ``plugin.py`` must contain:

1. A ``config`` dictionary at the top level.  ZEN-garden will fill this with the
   settings you specify in ``config.json`` under the respective plugin name.
2. One or more functions decorated with
   ``@EventPublisher.register(Event.<event_name>)``.  These functions are called
   automatically when ZEN-garden reaches the corresponding event.
3. Testing
4. Documentation

Install the plugin package into the same Python environment as ZEN-garden:

.. code-block:: shell

    pip install -e path/to/my_plugin

The ``-e`` flag installs it in *editable* mode, which means changes to your files
take effect immediately without reinstalling.


Available events
~~~~~~~~~~~~~~~~

Events are defined in ``zen_garden.plugin_system.events.Event``.
Each event corresponds to a specific point in the workflow and passes relevant
objects as keyword arguments. A current list of available event can be found
:ref:`here <plugins.available_events>`.

Step 5 — Activate in ``config.json``
--------------------------------------

Add the plugin's name to ``config.json``:

.. code-block:: json

    {
        "plugins": {
            "my_plugin_name": {
                "my_setting": 123
            }
        }
    }

The dictionary under ``"my_plugin_name"`` is merged into the plugin's ``config``
before any of its functions are called.  Use it to pass numerical parameters,
file paths, flags, and so on.


How the loader works
---------------------

When ZEN-garden starts, the loader scans all installed packages for entry points
in the ``zen_garden.plugins`` group.  For each plugin listed in ``config.json`` it:

1. Locates the installed entry point with that name.
2. Imports the corresponding ``plugin.py`` module.
3. Merges the user-provided settings into the module's ``config`` dictionary.

If a plugin name appears in ``config.json`` but is not installed, ZEN-garden raises
a ``ModuleNotFoundError`` with a clear message listing the available plugins.
