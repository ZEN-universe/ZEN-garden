"""
A template for a plugin.

Write functions that subscribe to an event if decorated.These functions are executed
when the execution reaches the trigger to the respective event.
The config dictionary will be filled by plugins.loader.register_plugins()
"""

from zen_garden.plugin_system.events import Event, EventPublisher

config = {}


# Choose the event that will trigger the function call
@EventPublisher.register(Event.test_event1)
def this_will_be_called_first(*args, **kwargs):
    """You can name this function as you wish."""
    pass
