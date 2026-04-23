.. _dev_guide.implementing_plugins:

############################
Implementing plugins
############################

This page explains how to implement a plugin for ZEN-garden.
Keep plugins small and focused: they should register callbacks for
events as defined in ZEN-garden main module.

The callbacks are executed at predefined locations (via events)
in the main module.


Plugin contract
---------------

- Location: place your plugin as a package under `zen_garden.plugins`. The loader
  imports the module `...<plugin>.plugin`. You can find a template in
  `zen_garden.plugins._template`
- Registration: use the `Event` helper to attach functions to events
  (see `zen_garden.events.Event` for available events). The function will
  be called at the respective event.
- Config: use the dictonary at `config.plugins` to specify a config to
  pass to the plugin.

Minimal example
----------------

The tests include a minimal plugin used as an example. A shortened version:

.. code-block:: python

    # tests/unit_tests/fake_plugin/plugin.py
    from zen_garden.plugin_system.events import EventPublisher
    from tests.unit_tests.test_events import TestEvent

    config = {}


    @EventPublisher.register(TestEvent.test_event1)
    def first_method(**kwargs):
        pass

    @EventPublisher.register(TestEvent.test_event2)
    def first_method(**kwargs):
        pass

How the loader uses your plugin
-------------------------------

The loader function `zen_garden.plugins.loader.register_plugins`
imports `zen_garden.plugins.<name>.plugin` and updates the plugin module's
`config` dict with the configuration provided in `config.json`. The loader
registers the plugin's callbacks to the events they are decorated with, so
that are executed during runtime when the respective event is triggered.

