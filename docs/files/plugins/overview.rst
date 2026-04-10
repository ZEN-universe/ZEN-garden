.. _plugins.intro:

############################
Overview
############################

The ZEN-garden plugin system lets you extend core behaviour without altering the
main codebase. Plugins are regular Python packages that register callback
functions for events triggered `zen_garden.events.EventPublisher`.

- Plugins live under `zen_garden.plugins.<plugin_name>.plugin` and expose a
  `config` dictionary for configuration.
- The loader `register_plugins` imports selected plugins and merges
  user-provided settings into each plugin's `config` before execution.
- Use the `EventPublisher.register(Event.<name>)` decorator to attach functions to
  events that will be called by the framework at defined points (for example, 
  before/after model construction). 

See also
--------

- Developer guide: :ref:`Implementing plugins <dev_guide.implementing_plugins>`
- Available plugins: :ref:`Available plugins <plugins.available_plugins>`
- Configurations: :ref:`Configuration options for plugins <configuration.plugins>`
